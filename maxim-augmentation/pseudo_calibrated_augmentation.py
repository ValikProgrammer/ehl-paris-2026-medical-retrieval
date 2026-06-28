from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter, map_coordinates, zoom
from scipy.optimize import linear_sum_assignment, minimize
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize


DEFAULT_DATA_ROOT = Path("/root/.cache/kagglehub/competitions/ehl-paris-medical-image-retrieval")


@dataclass
class TransformEstimate:
    image_id: str
    image_path: str
    dataset: str
    split: str
    side: str
    pair_query_id: str
    pair_target_id: str
    score: float
    margin: float
    mutual_nn: bool
    hungarian_top1: bool
    ncc_before: float
    ncc_after: float
    rx: float
    ry: float
    rz: float
    tx: float
    ty: float
    tz: float


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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


def register_to_template(vol: np.ndarray, template: np.ndarray, opt_size: int) -> tuple[np.ndarray, np.ndarray, float, float]:
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
        res = minimize(
            neg,
            start,
            method="Powell",
            options={"maxiter": 80, "xtol": 0.03, "ftol": 0.005},
        )
        if res.fun < best_f:
            best_f = float(res.fun)
            best_p = np.asarray(res.x, dtype=np.float64)
    scaled = best_p.copy()
    scaled[3:6] *= g / opt_size
    aligned = apply_rigid(vol, scaled)
    return aligned.astype(np.float32, copy=False), scaled, ncc(template, vol), ncc(template, aligned)


def build_model(data_root: Path, grid: int, components: int, alpha: float, opt_size: int):
    pairs = read_csv(data_root / "dataset1" / "train_pairs.csv")
    train_q = np.stack([load_grid(data_root, row["query_image"], grid) for row in pairs])
    train_t = np.stack([load_grid(data_root, row["target_image"], grid) for row in pairs])
    t1_template = train_q.mean(axis=0)
    t2_template = train_t.mean(axis=0)
    qf = np.stack([flat_feature(v) for v in train_q])
    tf = np.stack([flat_feature(v) for v in train_t])
    q_pca, t_pca, ridge = fit_pca_ridge(qf, tf, components, alpha)
    return t1_template, t2_template, q_pca, t_pca, ridge


def score_pool(data_root: Path, dataset: str, split: str, grid: int, opt_size: int, model):
    t1_template, t2_template, q_pca, t_pca, ridge = model
    root = data_root / dataset
    qrows = read_csv(root / f"{split}_queries.csv")
    trows = read_csv(root / f"{split}_gallery.csv")
    q_feats, t_feats = [], []
    q_regs, t_regs = [], []
    for i, row in enumerate(qrows, 1):
        vol = load_grid(data_root, row["query_image"], grid)
        aligned, params, before, after = register_to_template(vol, t1_template, opt_size)
        q_regs.append((params, before, after))
        q_feats.append(normalize(ridge.predict(q_pca.transform(flat_feature(aligned)[None, :])))[0])
        if i % 20 == 0:
            print(f"{dataset}/{split} query {i}/{len(qrows)}", flush=True)
    for i, row in enumerate(trows, 1):
        vol = load_grid(data_root, row["target_image"], grid)
        aligned, params, before, after = register_to_template(vol, t2_template, opt_size)
        t_regs.append((params, before, after))
        t_feats.append(normalize(t_pca.transform(flat_feature(aligned)[None, :]))[0])
        if i % 20 == 0:
            print(f"{dataset}/{split} target {i}/{len(trows)}", flush=True)
    scores = (np.stack(q_feats) @ np.stack(t_feats).T).astype(np.float64)
    return qrows, trows, scores, q_regs, t_regs


def select_pseudo_matches(
    qrows: list[dict[str, str]],
    trows: list[dict[str, str]],
    scores: np.ndarray,
    min_margin: float,
    min_score: float,
) -> list[tuple[int, int, float, float, bool, bool]]:
    top = np.argsort(-scores, axis=1, kind="mergesort")
    top1 = top[:, 0]
    top2 = top[:, 1] if scores.shape[1] > 1 else top[:, 0]
    margins = scores[np.arange(scores.shape[0]), top1] - scores[np.arange(scores.shape[0]), top2]
    target_best_query = np.argmax(scores, axis=0)
    row_ind, col_ind = linear_sum_assignment(-scores)
    assigned = np.full(scores.shape[0], -1, dtype=np.int64)
    assigned[row_ind] = col_ind

    matches = []
    for qi, ti in enumerate(top1):
        score = float(scores[qi, ti])
        margin = float(margins[qi])
        mutual = int(target_best_query[ti]) == qi
        hungarian_top1 = int(assigned[qi]) == int(ti)
        if score >= min_score and margin >= min_margin and mutual and hungarian_top1:
            matches.append((qi, int(ti), score, margin, mutual, hungarian_top1))
    return matches


def collect_estimates(
    data_root: Path,
    grid: int,
    opt_size: int,
    min_margin: float,
    min_score: float,
    model,
    out_dir: Path,
) -> list[TransformEstimate]:
    estimates: list[TransformEstimate] = []
    pseudo_rows: list[dict[str, str]] = []
    for split in ("val", "test"):
        qrows, trows, scores, q_regs, t_regs = score_pool(data_root, "dataset2", split, grid, opt_size, model)
        matches = select_pseudo_matches(qrows, trows, scores, min_margin, min_score)
        print(f"dataset2/{split}: kept {len(matches)}/{len(qrows)} pseudo-matches", flush=True)
        for qi, ti, score, margin, mutual, hungarian_top1 in matches:
            qrow, trow = qrows[qi], trows[ti]
            pseudo_rows.append(
                {
                    "split": split,
                    "query_id": qrow["query_id"],
                    "target_id": trow["target_id"],
                    "score": f"{score:.8f}",
                    "margin": f"{margin:.8f}",
                    "mutual_nn": str(mutual),
                    "hungarian_top1": str(hungarian_top1),
                }
            )
            for side, row, regs in (
                ("query", qrow, q_regs[qi]),
                ("target", trow, t_regs[ti]),
            ):
                params, before, after = regs
                estimates.append(
                    TransformEstimate(
                        image_id=row[f"{side}_id"],
                        image_path=row[f"{side}_image"],
                        dataset="dataset2",
                        split=split,
                        side=side,
                        pair_query_id=qrow["query_id"],
                        pair_target_id=trow["target_id"],
                        score=score,
                        margin=margin,
                        mutual_nn=mutual,
                        hungarian_top1=hungarian_top1,
                        ncc_before=float(before),
                        ncc_after=float(after),
                        rx=float(params[0]),
                        ry=float(params[1]),
                        rz=float(params[2]),
                        tx=float(params[3]),
                        ty=float(params[4]),
                        tz=float(params[5]),
                    )
                )
    write_csv(out_dir / "pseudo_matches.csv", pseudo_rows, list(pseudo_rows[0].keys()) if pseudo_rows else ["split", "query_id", "target_id", "score", "margin", "mutual_nn", "hungarian_top1"])
    write_csv(out_dir / "transform_estimates.csv", [{k: str(v) for k, v in asdict(e).items()} for e in estimates], list(asdict(estimates[0]).keys()) if estimates else list(TransformEstimate.__dataclass_fields__.keys()))
    return estimates


def params_array(estimates: list[TransformEstimate], side: str) -> np.ndarray:
    rows = [e for e in estimates if e.side == side]
    if not rows:
        raise ValueError(f"No transform estimates for side={side}")
    return np.asarray([[e.rx, e.ry, e.rz, e.tx, e.ty, e.tz] for e in rows], dtype=np.float64)


def robust_sample_params(values: np.ndarray, rng: np.random.Generator, jitter: float) -> np.ndarray:
    lo = np.percentile(values, 5, axis=0)
    hi = np.percentile(values, 95, axis=0)
    clipped = np.clip(values, lo, hi)
    mean = clipped.mean(axis=0)
    cov = np.cov(clipped.T)
    cov = cov + np.eye(cov.shape[0]) * 1e-6
    sample = rng.multivariate_normal(mean, cov * jitter)
    return np.clip(sample, lo, hi)


def apply_elastic(vol: np.ndarray, rng: np.random.Generator, probability: float, sigma_range: tuple[float, float], magnitude_range: tuple[float, float]) -> np.ndarray:
    if rng.random() > probability:
        return vol
    sigma = rng.uniform(*sigma_range)
    magnitude = rng.uniform(*magnitude_range)
    coords = np.stack(np.meshgrid(*[np.arange(s) for s in vol.shape], indexing="ij"))
    disp = [
        gaussian_filter(rng.standard_normal(vol.shape), sigma) * magnitude
        for _ in range(3)
    ]
    warped_coords = [coords[i] + disp[i] for i in range(3)]
    return map_coordinates(vol, warped_coords, order=1, mode="constant", cval=0.0).astype(np.float32, copy=False)


def save_grid_nifti(volume: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(volume.astype(np.float32), affine=np.eye(4))
    nib.save(image, str(path))


def generate_augmented_dataset(
    data_root: Path,
    out_dir: Path,
    grid: int,
    estimates: list[TransformEstimate],
    copies: int,
    seed: int,
    jitter: float,
    elastic_probability: float,
    elastic_sigma: tuple[float, float],
    elastic_magnitude: tuple[float, float],
) -> None:
    rng = np.random.default_rng(seed)
    q_values = params_array(estimates, "query")
    t_values = params_array(estimates, "target")
    pairs = read_csv(data_root / "dataset1" / "train_pairs.csv")
    rows: list[dict[str, str]] = [dict(row) for row in pairs]
    image_dir = out_dir / "images"
    for pair_index, row in enumerate(pairs, 1):
        q_base = load_grid(data_root, row["query_image"], grid)
        t_base = load_grid(data_root, row["target_image"], grid)
        for copy_index in range(copies):
            q_params = robust_sample_params(q_values, rng, jitter)
            t_params = robust_sample_params(t_values, rng, jitter)
            q_aug = apply_elastic(apply_rigid(q_base, q_params), rng, elastic_probability, elastic_sigma, elastic_magnitude)
            t_aug = apply_elastic(apply_rigid(t_base, t_params), rng, elastic_probability, elastic_sigma, elastic_magnitude)
            q_id = f"{row['query_id']}_pcal{copy_index:02d}"
            t_id = f"{row['target_id']}_pcal{copy_index:02d}"
            q_path = image_dir / f"{q_id}.nii.gz"
            t_path = image_dir / f"{t_id}.nii.gz"
            save_grid_nifti(q_aug, q_path)
            save_grid_nifti(t_aug, t_path)
            new_row = dict(row)
            new_row.update(
                {
                    "pair_id": f"{row['pair_id']}_pcal{copy_index:02d}",
                    "query_id": q_id,
                    "target_id": t_id,
                    "query_image": str(q_path),
                    "target_image": str(t_path),
                }
            )
            rows.append(new_row)
        if pair_index % 25 == 0:
            print(f"augmented {pair_index}/{len(pairs)} dataset1 pairs", flush=True)
    write_csv(out_dir / "train_pairs_pseudo_calibrated.csv", rows, list(rows[0].keys()))
    summary = {
        "original_pairs": len(pairs),
        "copies_per_pair": copies,
        "total_rows": len(rows),
        "grid": grid,
        "seed": seed,
        "query_param_mean": q_values.mean(axis=0).tolist(),
        "query_param_std": q_values.std(axis=0).tolist(),
        "target_param_mean": t_values.mean(axis=0).tolist(),
        "target_param_std": t_values.std(axis=0).tolist(),
        "elastic_probability": elastic_probability,
        "elastic_sigma": list(elastic_sigma),
        "elastic_magnitude": list(elastic_magnitude),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=Path("output"))
    parser.add_argument("--grid", type=int, default=44)
    parser.add_argument("--components", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--opt-size", type=int, default=20)
    parser.add_argument("--min-margin", type=float, default=0.02)
    parser.add_argument("--min-score", type=float, default=-1.0)
    parser.add_argument("--copies", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--jitter", type=float, default=1.0)
    parser.add_argument("--elastic-probability", type=float, default=0.45)
    parser.add_argument("--elastic-sigma", type=float, nargs=2, default=(5.0, 8.0))
    parser.add_argument("--elastic-magnitude", type=float, nargs=2, default=(3.0, 8.0))
    parser.add_argument("--skip-generate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"data_root={args.data_root}", flush=True)
    print(f"out_dir={args.out_dir}", flush=True)
    model = build_model(args.data_root, args.grid, args.components, args.alpha, args.opt_size)
    estimates = collect_estimates(
        args.data_root,
        args.grid,
        args.opt_size,
        args.min_margin,
        args.min_score,
        model,
        args.out_dir,
    )
    print(f"collected {len(estimates)} transform estimates", flush=True)
    if args.skip_generate:
        return
    generate_augmented_dataset(
        args.data_root,
        args.out_dir,
        args.grid,
        estimates,
        args.copies,
        args.seed,
        args.jitter,
        args.elastic_probability,
        tuple(args.elastic_sigma),
        tuple(args.elastic_magnitude),
    )


if __name__ == "__main__":
    main()
