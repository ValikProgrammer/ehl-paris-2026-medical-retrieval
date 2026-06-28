from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter, map_coordinates, zoom
from scipy.optimize import linear_sum_assignment, minimize
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize


DEFAULT_DATA_ROOT = Path("/root/.cache/kagglehub/competitions/ehl-paris-medical-image-retrieval")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


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


def apply_elastic(
    vol: np.ndarray,
    rng: np.random.Generator,
    probability: float,
    sigma_range: tuple[float, float],
    magnitude_range: tuple[float, float],
) -> np.ndarray:
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


def fit_pca_ridge(qf: np.ndarray, tf: np.ndarray, components: int, alpha: float):
    n = min(components, qf.shape[0] - 1, qf.shape[1], tf.shape[1])
    q_pca = PCA(n_components=n, whiten=True, random_state=20260627).fit(qf)
    t_pca = PCA(n_components=n, whiten=True, random_state=20260627).fit(tf)
    ridge = Ridge(alpha=alpha).fit(q_pca.transform(qf), t_pca.transform(tf))
    return q_pca, t_pca, ridge


def mrr_and_assignment(scores: np.ndarray) -> tuple[float, float]:
    n = scores.shape[0]
    ranks = []
    for i in range(n):
        order = np.argsort(-scores[i], kind="mergesort")
        ranks.append(int(np.where(order == i)[0][0]) + 1)
    row_ind, col_ind = linear_sum_assignment(-scores.astype(np.float64))
    assigned = col_ind[np.argsort(row_ind)]
    return float(np.mean(1.0 / np.asarray(ranks))), float(np.mean(assigned == np.arange(n)))


def transform_values(path: Path, side: str, min_margin: float) -> np.ndarray:
    rows = [
        row for row in read_csv(path)
        if row["side"] == side and float(row["margin"]) >= min_margin
    ]
    if not rows:
        raise ValueError(f"No transforms for side={side} at min_margin={min_margin}")
    return np.asarray(
        [[float(row[k]) for k in ("rx", "ry", "rz", "tx", "ty", "tz")] for row in rows],
        dtype=np.float64,
    )


def robust_sample(values: np.ndarray, rng: np.random.Generator, jitter: float) -> np.ndarray:
    lo = np.percentile(values, 5, axis=0)
    hi = np.percentile(values, 95, axis=0)
    clipped = np.clip(values, lo, hi)
    mean = clipped.mean(axis=0)
    cov = np.cov(clipped.T) + np.eye(values.shape[1]) * 1e-6
    sample = rng.multivariate_normal(mean, cov * jitter)
    return np.clip(sample, lo, hi)


def build_eval_set(args: argparse.Namespace):
    pairs = read_csv(args.data_root / "dataset1" / "train_pairs.csv")
    order = np.argsort([
        hashlib.sha256(f"{args.seed}:{row['pair_id']}".encode()).hexdigest()
        for row in pairs
    ])
    train_idx = order[:args.n_train]
    eval_idx = order[args.n_train: args.n_train + args.n_eval]
    train_rows = [pairs[i] for i in train_idx]
    eval_rows = [pairs[i] for i in eval_idx]

    train_q = np.stack([load_grid(args.data_root, row["query_image"], args.grid) for row in train_rows])
    train_t = np.stack([load_grid(args.data_root, row["target_image"], args.grid) for row in train_rows])
    t1_template = train_q.mean(axis=0)
    t2_template = train_t.mean(axis=0)
    q_pca, t_pca, ridge = fit_pca_ridge(
        np.stack([flat_feature(v) for v in train_q]),
        np.stack([flat_feature(v) for v in train_t]),
        args.components,
        args.alpha,
    )

    q_vals = transform_values(args.transform_estimates, "query", args.min_transform_margin)
    t_vals = transform_values(args.transform_estimates, "target", args.min_transform_margin)
    rng = np.random.default_rng(args.seed)
    eval_q, eval_t = [], []
    for i, row in enumerate(eval_rows, 1):
        q = load_grid(args.data_root, row["query_image"], args.grid)
        t = load_grid(args.data_root, row["target_image"], args.grid)
        q = apply_rigid(q, robust_sample(q_vals, rng, args.jitter))
        t = apply_rigid(t, robust_sample(t_vals, rng, args.jitter))
        q = apply_elastic(q, rng, args.elastic_probability, tuple(args.elastic_sigma), tuple(args.elastic_magnitude))
        t = apply_elastic(t, rng, args.elastic_probability, tuple(args.elastic_sigma), tuple(args.elastic_magnitude))
        eval_q.append(q)
        eval_t.append(t)
        if i % 20 == 0:
            print(f"built calibrated eval {i}/{len(eval_rows)}", flush=True)
    return t1_template, t2_template, q_pca, t_pca, ridge, np.stack(eval_q), np.stack(eval_t)


def evaluate(args: argparse.Namespace) -> tuple[float, float]:
    t1_template, t2_template, q_pca, t_pca, ridge, eval_q, eval_t = build_eval_set(args)
    qf, tf = [], []
    for i, vol in enumerate(eval_q, 1):
        aligned = register_to_template(vol, t1_template, args.opt_size, args.start_angle_deg, args.start_shift_vox, args.reg_maxiter)
        qf.append(flat_feature(aligned))
        if i % 20 == 0:
            print(f"registered eval queries {i}/{len(eval_q)}", flush=True)
    for i, vol in enumerate(eval_t, 1):
        aligned = register_to_template(vol, t2_template, args.opt_size, args.start_angle_deg, args.start_shift_vox, args.reg_maxiter)
        tf.append(flat_feature(aligned))
        if i % 20 == 0:
            print(f"registered eval targets {i}/{len(eval_t)}", flush=True)
    qz = normalize(ridge.predict(q_pca.transform(np.stack(qf))))
    tz = normalize(t_pca.transform(np.stack(tf)))
    return mrr_and_assignment((qz @ tz.T).astype(np.float64))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--transform-estimates", type=Path, default=Path("output/transform_estimates.csv"))
    parser.add_argument("--grid", type=int, default=44)
    parser.add_argument("--n-train", type=int, default=250)
    parser.add_argument("--n-eval", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--components", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--opt-size", type=int, default=20)
    parser.add_argument("--start-angle-deg", type=float, default=14.0)
    parser.add_argument("--start-shift-vox", type=float, default=0.0)
    parser.add_argument("--reg-maxiter", type=int, default=100)
    parser.add_argument("--min-transform-margin", type=float, default=0.01)
    parser.add_argument("--jitter", type=float, default=1.0)
    parser.add_argument("--elastic-probability", type=float, default=0.45)
    parser.add_argument("--elastic-sigma", type=float, nargs=2, default=(5.0, 8.0))
    parser.add_argument("--elastic-magnitude", type=float, nargs=2, default=(3.0, 8.0))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_mrr, hungarian_acc = evaluate(args)
    print(
        f"calibrated_synthetic n_train={args.n_train} n_eval={args.n_eval} "
        f"grid={args.grid} opt={args.opt_size} angle={args.start_angle_deg:g} "
        f"shift={args.start_shift_vox:g} jitter={args.jitter:g} "
        f"raw_mrr={raw_mrr:.6f} hungarian_acc={hungarian_acc:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
