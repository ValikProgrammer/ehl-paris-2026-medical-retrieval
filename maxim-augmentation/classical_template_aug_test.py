from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform, map_coordinates, zoom
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
    grid = np.stack(
        np.meshgrid(*[np.linspace(0, s - 1, size) for s in shape], indexing="ij"),
        axis=0,
    )
    sampled = map_coordinates(volume, grid, order=1, mode="constant", cval=0.0)
    return sampled.astype(np.float32, copy=False)


def load_grid(data_root: Path, image_path: str, grid: int) -> np.ndarray:
    return downsample(load_normalized(resolve_image_path(data_root, image_path)), grid)


def flat_feature(vol: np.ndarray) -> np.ndarray:
    x = vol.reshape(-1).astype(np.float32)
    x = x - x.mean()
    return x / (np.linalg.norm(x) + 1e-6)


def fit_pca(qf: np.ndarray, tf: np.ndarray, components: int):
    n = min(components, qf.shape[0] - 1, qf.shape[1], tf.shape[1])
    q_pca = PCA(n_components=n, whiten=True, random_state=20260627).fit(qf)
    t_pca = PCA(n_components=n, whiten=True, random_state=20260627).fit(tf)
    return q_pca, t_pca


def apply_assignment_mode(
    scores: np.ndarray,
    mode: str,
    lock_margin: float,
    assignment_topk: int,
) -> np.ndarray:
    if mode == "none":
        return scores
    if scores.shape[0] != scores.shape[1]:
        return scores
    adjusted = scores.copy()
    n = scores.shape[0]
    big = float(scores.max() + 1e6)

    if mode == "full":
        row_ind, col_ind = linear_sum_assignment(-scores)
        assigned = np.empty(n, dtype=np.int64)
        assigned[row_ind] = col_ind
        adjusted[np.arange(n), assigned] = big
        return adjusted

    if mode == "lock-confident":
        order = np.argsort(-scores, axis=1, kind="mergesort")
        top1 = order[:, 0]
        top2 = order[:, 1]
        margins = scores[np.arange(n), top1] - scores[np.arange(n), top2]
        locked_rows: list[int] = []
        locked_cols: list[int] = []
        used_cols: set[int] = set()
        for row in np.argsort(-margins):
            col = int(top1[row])
            if margins[row] >= lock_margin and col not in used_cols:
                locked_rows.append(int(row))
                locked_cols.append(col)
                used_cols.add(col)
        for row, col in zip(locked_rows, locked_cols):
            adjusted[row, col] = big
        remaining_rows = [i for i in range(n) if i not in set(locked_rows)]
        remaining_cols = [j for j in range(n) if j not in used_cols]
        if remaining_rows and remaining_cols:
            sub = scores[np.ix_(remaining_rows, remaining_cols)]
            row_ind, col_ind = linear_sum_assignment(-sub)
            for rr, cc in zip(row_ind, col_ind):
                adjusted[remaining_rows[rr], remaining_cols[cc]] = big
        return adjusted

    if mode == "topk":
        k = max(1, min(assignment_topk, n))
        allowed = np.zeros_like(scores, dtype=bool)
        order = np.argsort(-scores, axis=1, kind="mergesort")
        for i in range(n):
            allowed[i, order[i, :k]] = True
        penalty = float(scores.min() - 1e6)
        constrained = np.where(allowed, scores, penalty)
        row_ind, col_ind = linear_sum_assignment(-constrained)
        assigned = np.empty(n, dtype=np.int64)
        assigned[row_ind] = col_ind
        adjusted[np.arange(n), assigned] = big
        return adjusted

    raise ValueError(f"Unknown assignment mode: {mode}")


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


def registration_starts(angle_deg: float, shift_vox: float) -> tuple[np.ndarray, ...]:
    angle = np.deg2rad(angle_deg)
    starts = [np.zeros(6, dtype=np.float64)]
    for axis in range(3):
        for sign in (-1.0, 1.0):
            p = np.zeros(6, dtype=np.float64)
            p[axis] = sign * angle
            starts.append(p)
    if shift_vox > 0:
        for axis in range(3):
            for sign in (-1.0, 1.0):
                p = np.zeros(6, dtype=np.float64)
                p[3 + axis] = sign * shift_vox
                starts.append(p)
    return tuple(starts)


def register_to_template(
    vol: np.ndarray,
    template: np.ndarray,
    opt_size: int,
    start_angle_deg: float,
    start_shift_vox: float,
    maxiter: int,
) -> np.ndarray:
    g = vol.shape[0]
    zt = zoom(template, opt_size / template.shape[0], order=1)
    zv = zoom(vol, opt_size / g, order=1)

    def neg(params: np.ndarray) -> float:
        return -ncc(zt, apply_rigid(zv, params))

    best_p = np.zeros(6)
    best_f = neg(best_p)
    for start in registration_starts(start_angle_deg, start_shift_vox):
        res = minimize(
            neg,
            start,
            method="Powell",
            options={"maxiter": maxiter, "xtol": 0.025, "ftol": 0.003},
        )
        if res.fun < best_f:
            best_f = float(res.fun)
            best_p = np.asarray(res.x, dtype=np.float64)
    scaled = best_p.copy()
    scaled[3:6] *= g / opt_size
    return apply_rigid(vol, scaled).astype(np.float32, copy=False)


def load_pair_features(data_root: Path, pair_csv: Path, grid: int, limit: int | None) -> tuple[np.ndarray, np.ndarray]:
    pairs = read_csv(pair_csv)
    if limit is not None:
        pairs = pairs[:limit]
    q = np.stack([flat_feature(load_grid(data_root, row["query_image"], grid)) for row in pairs])
    t = np.stack([flat_feature(load_grid(data_root, row["target_image"], grid)) for row in pairs])
    return q, t


def load_pair_grids(data_root: Path, pair_csv: Path, grid: int, limit: int | None) -> tuple[np.ndarray, np.ndarray]:
    pairs = read_csv(pair_csv)
    if limit is not None:
        pairs = pairs[:limit]
    q = np.stack([load_grid(data_root, row["query_image"], grid) for row in pairs])
    t = np.stack([load_grid(data_root, row["target_image"], grid) for row in pairs])
    return q, t


def build_model(
    data_root: Path,
    template_pair_csv: Path,
    pca_pair_csv: Path,
    ridge_pair_csv: Path,
    grid: int,
    components: int,
    alpha: float,
    ridge_limit: int | None,
):
    template_q, template_t = load_pair_grids(data_root, template_pair_csv, grid, limit=None)
    t1_template = template_q.mean(axis=0)
    t2_template = template_t.mean(axis=0)

    pca_qf, pca_tf = load_pair_features(data_root, pca_pair_csv, grid, limit=None)
    q_pca, t_pca = fit_pca(pca_qf, pca_tf, components)

    ridge_qf, ridge_tf = load_pair_features(data_root, ridge_pair_csv, grid, limit=ridge_limit)
    ridge = Ridge(alpha=alpha).fit(q_pca.transform(ridge_qf), t_pca.transform(ridge_tf))
    print(
        f"built templates from {len(template_q)} pairs, pca from {len(pca_qf)} pairs, "
        f"ridge from {len(ridge_qf)} pairs",
        flush=True,
    )
    return t1_template, t2_template, q_pca, t_pca, ridge


def score_pool(
    data_root: Path,
    dataset: str,
    split: str,
    grid: int,
    opt_size: int,
    start_angle_deg: float,
    start_shift_vox: float,
    reg_maxiter: int,
    model,
    assignment_mode: str,
    lock_margin: float,
    assignment_topk: int,
    register: bool,
) -> list[dict[str, str]]:
    t1_template, t2_template, q_pca, t_pca, ridge = model
    root = data_root / dataset
    qrows = read_csv(root / f"{split}_queries.csv")
    trows = read_csv(root / f"{split}_gallery.csv")

    q = []
    for i, row in enumerate(qrows, 1):
        vol = load_grid(data_root, row["query_image"], grid)
        feat = flat_feature(
            register_to_template(vol, t1_template, opt_size, start_angle_deg, start_shift_vox, reg_maxiter)
            if register else vol
        )
        q.append(normalize(ridge.predict(q_pca.transform(feat[None, :])))[0])
        if i % 20 == 0:
            print(f"{dataset}/{split} query {i}/{len(qrows)}", flush=True)

    t = []
    for i, row in enumerate(trows, 1):
        vol = load_grid(data_root, row["target_image"], grid)
        feat = flat_feature(
            register_to_template(vol, t2_template, opt_size, start_angle_deg, start_shift_vox, reg_maxiter)
            if register else vol
        )
        t.append(normalize(t_pca.transform(feat[None, :]))[0])
        if i % 20 == 0:
            print(f"{dataset}/{split} target {i}/{len(trows)}", flush=True)

    q_arr = np.stack(q)
    t_arr = np.stack(t)
    scores = (q_arr @ t_arr.T).astype(np.float64)
    scores = apply_assignment_mode(scores, assignment_mode, lock_margin, assignment_topk)

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
    parser.add_argument("--train-pair-csv", type=Path, required=True,
                        help="CSV used for templates, PCA, and Ridge unless overridden.")
    parser.add_argument("--template-pair-csv", type=Path, default=None,
                        help="CSV used to build T1/T2 templates. Use clean dataset1 here.")
    parser.add_argument("--pca-pair-csv", type=Path, default=None,
                        help="CSV used to fit PCA bases. Use clean dataset1 here.")
    parser.add_argument("--ridge-pair-csv", type=Path, default=None,
                        help="CSV used to fit Ridge. Use calibrated augmentation here.")
    parser.add_argument("--ridge-limit", type=int, default=None,
                        help="Optional first-N rows from the Ridge CSV; useful for reducing augmentation weight.")
    parser.add_argument("--datasets", nargs="+", default=["dataset2"])
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--grid", type=int, default=44)
    parser.add_argument("--components", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--opt-size", type=int, default=20)
    parser.add_argument("--start-angle-deg", type=float, default=14.0,
                        help="Registration multi-start angle around each axis.")
    parser.add_argument("--start-shift-vox", type=float, default=0.0,
                        help="Registration multi-start translation at opt-size resolution.")
    parser.add_argument("--reg-maxiter", type=int, default=100)
    parser.add_argument("--assignment", action="store_true")
    parser.add_argument("--assignment-mode", choices=["none", "full", "lock-confident", "topk"], default=None)
    parser.add_argument("--lock-margin", type=float, default=0.05)
    parser.add_argument("--assignment-topk", type=int, default=10)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("output/d2_template_pseudo_calibrated_g44_hungarian.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assignment_mode = args.assignment_mode
    if assignment_mode is None:
        assignment_mode = "full" if args.assignment else "none"
    template_pair_csv = args.template_pair_csv or args.train_pair_csv
    pca_pair_csv = args.pca_pair_csv or args.train_pair_csv
    ridge_pair_csv = args.ridge_pair_csv or args.train_pair_csv
    model = build_model(
        args.data_root,
        template_pair_csv,
        pca_pair_csv,
        ridge_pair_csv,
        args.grid,
        args.components,
        args.alpha,
        args.ridge_limit,
    )
    rows: list[dict[str, str]] = []
    for dataset in args.datasets:
        for split in args.splits:
            rows.extend(
                score_pool(
                    args.data_root,
                    dataset,
                    split,
                    args.grid,
                    args.opt_size,
                    args.start_angle_deg,
                    args.start_shift_vox,
                    args.reg_maxiter,
                    model,
                    assignment_mode,
                    args.lock_margin,
                    args.assignment_topk,
                    register=not args.no_register,
                )
            )
    write_csv(args.out, rows)
    print(f"wrote {len(rows)} rows to {args.out}", flush=True)


if __name__ == "__main__":
    main()
