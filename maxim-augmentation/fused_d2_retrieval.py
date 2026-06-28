from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform, map_coordinates, sobel, zoom
from scipy.optimize import linear_sum_assignment, minimize
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize


DEFAULT_DATA_ROOT = Path("/root/.cache/kagglehub/competitions/ehl-paris-medical-image-retrieval")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "target_id_ranking"])
        writer.writeheader()
        writer.writerows(rows)


def resolve_image_path(data_root: Path, image_path: str) -> Path:
    path = Path(image_path)
    if path.exists():
        return path
    if not path.is_absolute():
        path = data_root / path
        if path.exists():
            return path
    if path.name.endswith(".nii.gz"):
        fallback = path.with_name(path.name[:-3])
        if fallback.exists():
            return fallback
    raise FileNotFoundError(path)


def load_normalized(path: Path) -> np.ndarray:
    image = nib.load(str(path))
    volume = np.asanyarray(image.dataobj).astype(np.float32)
    if volume.ndim > 3:
        volume = volume[..., 0]
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    mask = np.abs(volume) > 1e-6
    values = volume[mask]
    if values.size < 256:
        values = volume.reshape(-1)
    lo, hi = np.percentile(values, [1.0, 99.0]).astype(np.float32)
    scale = float(hi - lo) if np.isfinite(hi - lo) and float(hi - lo) > 1e-6 else 1.0
    volume = np.clip((volume - lo) / scale, 0.0, 1.0)
    volume[~mask] = 0.0
    return volume.astype(np.float32, copy=False)


def downsample(volume: np.ndarray, size: int) -> np.ndarray:
    shape = np.asarray(volume.shape, dtype=np.float64)
    grid = np.stack(np.meshgrid(*[np.linspace(0, s - 1, size) for s in shape], indexing="ij"), axis=0)
    return map_coordinates(volume, grid, order=1, mode="constant", cval=0.0).astype(np.float32, copy=False)


def load_grid(data_root: Path, image_path: str, grid: int) -> np.ndarray:
    return downsample(load_normalized(resolve_image_path(data_root, image_path)), grid)


def l2(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.mean(axis=-1, keepdims=True)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)


def flat_feature(vol: np.ndarray) -> np.ndarray:
    return l2(vol.reshape(1, -1))[0]


def fit_pca_ridge(qf: np.ndarray, tf: np.ndarray, components: int, alpha: float):
    n = min(components, qf.shape[0] - 1, qf.shape[1], tf.shape[1])
    q_pca = PCA(n_components=n, whiten=True, random_state=20260627).fit(qf)
    t_pca = PCA(n_components=n, whiten=True, random_state=20260627).fit(tf)
    ridge = Ridge(alpha=alpha).fit(q_pca.transform(qf), t_pca.transform(tf))
    return q_pca, t_pca, ridge


def rigid_matrix(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ax, ay, az = params[:3]
    rot = np.eye(3)
    for axis, angle in enumerate((ax, ay, az)):
        c, s = np.cos(angle), np.sin(angle)
        r = np.eye(3)
        a, b = [i for i in range(3) if i != axis]
        r[a, a], r[a, b], r[b, a], r[b, b] = c, -s, s, c
        rot = rot @ r
    return rot, params[3:6]


def apply_rigid(vol: np.ndarray, params: np.ndarray) -> np.ndarray:
    rot, shift = rigid_matrix(params)
    center = (np.asarray(vol.shape) - 1) / 2.0
    offset = center - rot @ center + shift
    return affine_transform(vol, rot, offset=offset, order=1, mode="constant", cval=0.0)


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    fg = (a > 0.02) | (b > 0.02)
    if int(fg.sum()) < 16:
        return 0.0
    x, y = a[fg], b[fg]
    x = x - x.mean()
    y = y - y.mean()
    return float((x @ y) / ((np.linalg.norm(x) * np.linalg.norm(y)) + 1e-6))


def register_to_template(vol: np.ndarray, template: np.ndarray, opt_size: int) -> np.ndarray:
    g = vol.shape[0]
    zt = zoom(template, opt_size / template.shape[0], order=1)
    zv = zoom(vol, opt_size / g, order=1)

    def neg(params: np.ndarray) -> float:
        return -ncc(zt, apply_rigid(zv, params))

    starts = (
        np.zeros(6),
        np.array([0.25, 0, 0, 0, 0, 0], dtype=np.float64),
        np.array([0, 0.25, 0, 0, 0, 0], dtype=np.float64),
        np.array([0, 0, 0.25, 0, 0, 0], dtype=np.float64),
    )
    best_p = np.zeros(6)
    best_f = neg(best_p)
    for start in starts:
        res = minimize(neg, start, method="Powell", options={"maxiter": 80, "xtol": 0.03, "ftol": 0.005})
        if res.fun < best_f:
            best_f = float(res.fun)
            best_p = np.asarray(res.x, dtype=np.float64)
    scaled = best_p.copy()
    scaled[3:6] *= g / opt_size
    return apply_rigid(vol, scaled).astype(np.float32, copy=False)


def canonical_feature(vol: np.ndarray, size: int) -> np.ndarray:
    mask = vol > 0.05
    coords = np.column_stack(np.where(mask))
    if len(coords) < 32:
        coords = np.column_stack(np.where(np.ones_like(vol, dtype=bool)))
    center = coords.mean(axis=0)
    cov = np.cov((coords - center).T)
    values, vectors = np.linalg.eigh(cov)
    vectors = vectors[:, np.argsort(values)[::-1]]
    projections = (coords - center) @ vectors
    scales = np.maximum(np.percentile(np.abs(projections), 99.0, axis=0), 1.0)
    for axis in range(3):
        if float(np.mean(projections[:, axis] ** 3)) < 0:
            vectors[:, axis] *= -1
    grid = np.stack(np.meshgrid(*[np.linspace(-1.0, 1.0, size) for _ in range(3)], indexing="ij"), axis=-1)
    physical = center + (grid.reshape(-1, 3) * scales) @ vectors.T
    sampled = map_coordinates(
        vol,
        [physical[:, 0], physical[:, 1], physical[:, 2]],
        order=1,
        mode="constant",
        cval=0.0,
    ).reshape(size, size, size)
    edge = np.sqrt(sum(sobel(sampled, axis=a) ** 2 for a in range(3)))
    return l2(np.concatenate([sampled.reshape(-1), edge.reshape(-1)])[None, :])[0]


def invariant_feature(vol: np.ndarray, n_int: int = 48, n_rad: int = 24, n_grad: int = 32) -> np.ndarray:
    mask = vol > 0.05
    fg = vol[mask]
    if fg.size < 32:
        fg = vol.reshape(-1)
    parts = []
    hist, _ = np.histogram(fg, bins=n_int, range=(0.0, 1.0), density=True)
    parts.append(hist.astype(np.float32))

    coords = np.column_stack(np.where(mask)).astype(np.float32)
    if len(coords) < 32:
        coords = np.column_stack(np.where(np.ones_like(vol, dtype=bool))).astype(np.float32)
    center = coords.mean(axis=0)
    rad = np.linalg.norm(coords - center, axis=1)
    bins = np.clip((rad / (float(rad.max()) + 1e-6) * n_rad).astype(int), 0, n_rad - 1)
    vals = vol[mask] if mask.sum() == len(coords) else vol.reshape(-1)
    prof = np.zeros(n_rad, dtype=np.float32)
    cnt = np.zeros(n_rad, dtype=np.float32)
    np.add.at(prof, bins, vals)
    np.add.at(cnt, bins, 1.0)
    parts.append(prof / (cnt + 1e-6))

    grad = np.sqrt(sum(sobel(vol, axis=a) ** 2 for a in range(3)))
    ghist, _ = np.histogram(grad[mask], bins=n_grad, range=(0.0, float(grad.max()) + 1e-6), density=True)
    parts.append(ghist.astype(np.float32))

    cov = np.cov((coords - center).T)
    eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
    parts.append((eig / (eig.sum() + 1e-6)).astype(np.float32))
    return l2(np.concatenate(parts)[None, :])[0]


def zscore_rows(scores: np.ndarray) -> np.ndarray:
    return (scores - scores.mean(axis=1, keepdims=True)) / (scores.std(axis=1, keepdims=True) + 1e-6)


def build_clean_model(data_root: Path, grid: int, components: int, alpha: float):
    pairs = read_csv(data_root / "dataset1" / "train_pairs.csv")
    tq = np.stack([load_grid(data_root, row["query_image"], grid) for row in pairs])
    tt = np.stack([load_grid(data_root, row["target_image"], grid) for row in pairs])
    t1_template = tq.mean(axis=0)
    t2_template = tt.mean(axis=0)
    q_pca, t_pca, ridge = fit_pca_ridge(
        np.stack([flat_feature(v) for v in tq]),
        np.stack([flat_feature(v) for v in tt]),
        components,
        alpha,
    )
    return tq, tt, t1_template, t2_template, q_pca, t_pca, ridge


def score_dataset(data_root: Path, dataset: str, split: str, args: argparse.Namespace, model) -> list[dict[str, str]]:
    train_q, train_t, t1_template, t2_template, q_pca, t_pca, ridge = model
    qrows = read_csv(data_root / dataset / f"{split}_queries.csv")
    trows = read_csv(data_root / dataset / f"{split}_gallery.csv")

    raw_q, raw_t = [], []
    template_q, template_t = [], []
    canon_q, canon_t = [], []
    inv_q, inv_t = [], []

    for i, row in enumerate(qrows, 1):
        vol = load_grid(data_root, row["query_image"], args.grid)
        raw_q.append(vol)
        template_q.append(flat_feature(register_to_template(vol, t1_template, args.opt_size)))
        if args.canonical_weight:
            canon_q.append(canonical_feature(vol, args.canonical_size))
        if args.invariant_weight:
            inv_q.append(invariant_feature(vol))
        if i % 20 == 0:
            print(f"{dataset}/{split} query {i}/{len(qrows)}", flush=True)

    for i, row in enumerate(trows, 1):
        vol = load_grid(data_root, row["target_image"], args.grid)
        raw_t.append(vol)
        template_t.append(flat_feature(register_to_template(vol, t2_template, args.opt_size)))
        if args.canonical_weight:
            canon_t.append(canonical_feature(vol, args.canonical_size))
        if args.invariant_weight:
            inv_t.append(invariant_feature(vol))
        if i % 20 == 0:
            print(f"{dataset}/{split} target {i}/{len(trows)}", flush=True)

    qz = normalize(ridge.predict(q_pca.transform(np.stack(template_q))))
    tz = normalize(t_pca.transform(np.stack(template_t)))
    scores = args.template_weight * zscore_rows((qz @ tz.T).astype(np.float64))

    if args.canonical_weight:
        train_cq = np.stack([canonical_feature(v, args.canonical_size) for v in train_q])
        train_ct = np.stack([canonical_feature(v, args.canonical_size) for v in train_t])
        cq_pca, ct_pca, c_ridge = fit_pca_ridge(train_cq, train_ct, args.canonical_components, args.alpha)
        cq = normalize(c_ridge.predict(cq_pca.transform(np.stack(canon_q))))
        ct = normalize(ct_pca.transform(np.stack(canon_t)))
        scores += args.canonical_weight * zscore_rows((cq @ ct.T).astype(np.float64))

    if args.invariant_weight:
        train_iq = np.stack([invariant_feature(v) for v in train_q])
        train_it = np.stack([invariant_feature(v) for v in train_t])
        iq_pca, it_pca, i_ridge = fit_pca_ridge(train_iq, train_it, min(args.invariant_components, train_iq.shape[1]), args.alpha)
        iq = normalize(i_ridge.predict(iq_pca.transform(np.stack(inv_q))))
        it = normalize(it_pca.transform(np.stack(inv_t)))
        scores += args.invariant_weight * zscore_rows((iq @ it.T).astype(np.float64))

    if args.assignment and scores.shape[0] == scores.shape[1]:
        row_ind, col_ind = linear_sum_assignment(-scores)
        assigned = np.empty(scores.shape[0], dtype=np.int64)
        assigned[row_ind] = col_ind
        scores[np.arange(scores.shape[0]), assigned] = scores.max() + 1e6

    target_ids = np.asarray([row["target_id"] for row in trows])
    rows = []
    for qi, row in enumerate(qrows):
        order = np.argsort(-scores[qi], kind="mergesort")
        rows.append({"query_id": row["query_id"], "target_id_ranking": " ".join(target_ids[order].tolist())})
    print(f"scored {dataset}/{split}: {len(qrows)}x{len(trows)}", flush=True)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--datasets", nargs="+", default=["dataset2"])
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--grid", type=int, default=44)
    parser.add_argument("--components", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--opt-size", type=int, default=20)
    parser.add_argument("--template-weight", type=float, default=1.0)
    parser.add_argument("--canonical-weight", type=float, default=0.20)
    parser.add_argument("--canonical-size", type=int, default=24)
    parser.add_argument("--canonical-components", type=int, default=128)
    parser.add_argument("--invariant-weight", type=float, default=0.0)
    parser.add_argument("--invariant-components", type=int, default=64)
    parser.add_argument("--assignment", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("output/d2_fused_template_canon.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = build_clean_model(args.data_root, args.grid, args.components, args.alpha)
    rows = []
    for dataset in args.datasets:
        for split in args.splits:
            rows.extend(score_dataset(args.data_root, dataset, split, args, model))
    write_csv(args.out, rows)
    print(f"wrote {len(rows)} rows to {args.out}", flush=True)


if __name__ == "__main__":
    main()
