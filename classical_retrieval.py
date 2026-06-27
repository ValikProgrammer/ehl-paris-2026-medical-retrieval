from __future__ import annotations
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "nibabel>=5.3",
#   "numpy>=2.0",
#   "scipy>=1.14",
#   "scikit-learn>=1.5",
# ]
# ///

"""Classical MRI retrieval experiments for the EHL Paris challenge.

This script deliberately stays independent from the provided MONAI/PyTorch
baseline. It implements low-risk baselines from the research report:

- NIfTI/header QA and path fallback for local .nii files.
- Robust intensity normalization and foreground cropping.
- Geometry/shape, raw-volume, edge, projection, mask, and histogram features.
- Dataset1 train-pair CV for measurable ablations.
- PCA + ridge cross-modal mapping trained from labelled dataset1 pairs.
- Combined Kaggle submission generation for all validation/test pools.
"""

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
from scipy.ndimage import sobel, zoom
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


SEED = 20260627
DEFAULT_DATA_ROOT = Path("ehl-paris-medical-image-retrieval")
DEFAULT_CACHE_DIR = Path(".classical_cache")
DEFAULT_OUT_DIR = Path("submissions")

FEATURES = (
    "raw_grid32",
    "raw_crop32",
    "edge_grid32",
    "edge_crop32",
    "mask_crop32",
    "proj64",
    "mask_proj64",
    "shape_moments",
    "pca_abs_mask24",
    "pca_abs_raw24",
    "hist64",
    "meta12",
)

FUSIONS: dict[str, tuple[str, ...]] = {
    "fusion_grid": ("raw_grid32", "edge_grid32", "mask_crop32"),
    "fusion_crop": ("raw_crop32", "edge_crop32", "mask_crop32", "proj64"),
    "fusion_shape": ("mask_crop32", "mask_proj64", "meta12"),
    "fusion_robust_shape": ("mask_crop32", "mask_proj64", "shape_moments"),
    "fusion_pca_abs": ("pca_abs_mask24", "pca_abs_raw24", "mask_proj64"),
    "fusion_dataset2": ("pca_abs_mask24", "pca_abs_raw24", "mask_crop32", "mask_proj64"),
    "fusion_dataset3": ("mask_crop32", "mask_proj64", "raw_crop32", "edge_crop32"),
    "fusion_all": (
        "raw_grid32",
        "raw_crop32",
        "edge_grid32",
        "edge_crop32",
        "mask_crop32",
        "proj64",
        "mask_proj64",
        "shape_moments",
        "pca_abs_mask24",
        "pca_abs_raw24",
        "hist64",
        "meta12",
    ),
    # Strong default for hidden pools: shape/mask helps deformations, edge/proj
    # helps cross-modal anatomy, raw crop helps dataset1's common grid.
    "fusion_default": (
        "raw_crop32",
        "edge_crop32",
        "mask_crop32",
        "proj64",
        "mask_proj64",
    ),
}

PCA_FEATURES = ("raw_crop32", "edge_crop32", "mask_crop32", "proj64", "mask_proj64")
UNCENTERED_FEATURES = {"hist64", "meta12", "shape_moments", "pca_abs_mask24", "pca_abs_raw24"}


@dataclass(frozen=True)
class Pool:
    dataset: str
    split: str
    query_csv: Path
    gallery_csv: Path


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
    if not path.is_absolute():
        path = data_root / path
    if path.exists():
        return path
    # The local download stores uncompressed .nii files while manifests say .nii.gz.
    if path.name.endswith(".nii.gz"):
        nii_path = path.with_name(path.name[:-3])
        if nii_path.exists():
            return nii_path
    raise FileNotFoundError(f"Image path not found: {path}")


def load_train_pairs(data_root: Path) -> list[dict[str, str]]:
    rows = read_csv(data_root / "dataset1" / "train_pairs.csv")
    for row in rows:
        row["query_path"] = str(resolve_image_path(data_root, row["query_image"]))
        row["target_path"] = str(resolve_image_path(data_root, row["target_image"]))
    return rows


def load_query_manifest(data_root: Path, path: Path) -> dict[str, Path]:
    return {row["query_id"]: resolve_image_path(data_root, row["query_image"]) for row in read_csv(path)}


def load_gallery_manifest(data_root: Path, path: Path) -> dict[str, Path]:
    return {row["target_id"]: resolve_image_path(data_root, row["target_image"]) for row in read_csv(path)}


def all_prediction_pools(
    data_root: Path,
    datasets: tuple[str, ...] = ("dataset1", "dataset2", "dataset3"),
    splits: tuple[str, ...] = ("val", "test"),
) -> list[Pool]:
    pools: list[Pool] = []
    for dataset in datasets:
        for split in splits:
            ds = data_root / dataset
            pools.append(
                Pool(
                    dataset=dataset,
                    split=split,
                    query_csv=ds / f"{split}_queries.csv",
                    gallery_csv=ds / f"{split}_gallery.csv",
                )
            )
    return pools


def safe_percentile(values: np.ndarray, q: Iterable[float]) -> np.ndarray:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.asarray([0.0 for _ in q], dtype=np.float32)
    return np.percentile(values, list(q)).astype(np.float32)


def bbox_from_mask(mask: np.ndarray, margin: int = 2) -> tuple[slice, slice, slice]:
    coords = np.where(mask)
    if len(coords[0]) == 0:
        return tuple(slice(0, s) for s in mask.shape)  # type: ignore[return-value]
    slices = []
    for axis, values in enumerate(coords):
        lo = max(0, int(values.min()) - margin)
        hi = min(mask.shape[axis], int(values.max()) + margin + 1)
        slices.append(slice(lo, hi))
    return tuple(slices)  # type: ignore[return-value]


def resize_nd(arr: np.ndarray, shape: tuple[int, ...], order: int) -> np.ndarray:
    factors = [target / max(1, current) for target, current in zip(shape, arr.shape)]
    out = zoom(arr, factors, order=order)
    # zoom can be off by one because of rounding. Pad/crop to exact shape.
    result = np.zeros(shape, dtype=np.float32)
    common = tuple(slice(0, min(a, b)) for a, b in zip(out.shape, shape))
    result[common] = out[common].astype(np.float32, copy=False)
    return result


def resize_2d(arr: np.ndarray, size: int = 64, order: int = 1) -> np.ndarray:
    return resize_nd(arr.astype(np.float32, copy=False), (size, size), order=order)


def robust_normalize(volume: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    foreground = volume[mask]
    if foreground.size < 256:
        foreground = volume[np.isfinite(volume)]
    lo, hi = safe_percentile(foreground, (1.0, 99.0))
    if not np.isfinite(hi - lo) or float(hi - lo) < 1e-6:
        lo, hi = safe_percentile(foreground, (0.0, 100.0))
    scale = float(hi - lo) if float(hi - lo) > 1e-6 else 1.0
    norm = np.clip((volume - float(lo)) / scale, 0.0, 1.0).astype(np.float32)
    return norm, np.asarray([float(lo), float(hi)], dtype=np.float32)


def gradient_magnitude(arr: np.ndarray) -> np.ndarray:
    gx = sobel(arr, axis=0)
    gy = sobel(arr, axis=1)
    gz = sobel(arr, axis=2)
    edge = np.sqrt(gx * gx + gy * gy + gz * gz).astype(np.float32)
    hi = float(np.percentile(edge[np.isfinite(edge)], 99.0)) if np.isfinite(edge).any() else 1.0
    if hi > 1e-6:
        edge = np.clip(edge / hi, 0.0, 1.0)
    return edge.astype(np.float32)


def foreground_shape_moments(mask: np.ndarray, norm: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(mask))
    if coords.shape[0] < 16:
        return np.zeros(96, dtype=np.float32)

    coords = coords.astype(np.float32)
    values = norm[mask].astype(np.float32)
    shape = np.asarray(mask.shape, dtype=np.float32)
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    cov = (centered.T @ centered) / max(1, coords.shape[0] - 1)
    eigvals = np.linalg.eigvalsh(cov).astype(np.float32)
    eigvals = np.sort(np.maximum(eigvals, 0.0))[::-1]
    eig_sqrt = np.sqrt(eigvals + 1e-6)

    dist = np.linalg.norm(centered, axis=1) / (float(np.linalg.norm(shape)) + 1e-6)
    dist_hist, _ = np.histogram(dist, bins=48, range=(0.0, 0.6), density=False)
    dist_hist = dist_hist.astype(np.float32)
    dist_hist /= float(dist_hist.sum() + 1e-6)

    scaled = centered / (eig_sqrt[None, :] + 1e-6)
    mahal = np.sqrt(np.sum(scaled * scaled, axis=1))
    mahal_hist, _ = np.histogram(mahal, bins=32, range=(0.0, 4.0), density=False)
    mahal_hist = mahal_hist.astype(np.float32)
    mahal_hist /= float(mahal_hist.sum() + 1e-6)

    radial_edges = np.linspace(0.0, 0.6, 9, dtype=np.float32)
    radial_means = []
    for lo, hi in zip(radial_edges[:-1], radial_edges[1:]):
        in_shell = (dist >= float(lo)) & (dist < float(hi))
        radial_means.append(float(values[in_shell].mean()) if np.any(in_shell) else 0.0)

    summary = np.concatenate(
        [
            np.sort(shape / 256.0),
            np.sort(eig_sqrt / (shape.max() + 1e-6)),
            np.asarray(
                [
                    float(mask.mean()),
                    float(np.percentile(dist, 50)),
                    float(np.percentile(dist, 75)),
                    float(np.percentile(dist, 90)),
                    float(np.percentile(values, 25)),
                    float(np.percentile(values, 50)),
                    float(np.percentile(values, 75)),
                    float(values.mean()),
                    float(values.std()),
                    float((centroid / (shape + 1e-6)).mean()),
                ],
                dtype=np.float32,
            ),
            np.asarray(radial_means, dtype=np.float32),
            dist_hist,
            mahal_hist,
        ]
    ).astype(np.float32)

    if summary.size < 96:
        summary = np.pad(summary, (0, 96 - summary.size))
    return summary[:96].astype(np.float32)


def foreground_pca_abs_hists(
    mask: np.ndarray,
    norm: np.ndarray,
    bins: int = 24,
    max_points: int = 250_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotation/translation/sign-invariant foreground histograms in PCA space."""
    coords = np.column_stack(np.where(mask))
    if coords.shape[0] < 64:
        zeros = np.zeros(bins**3, dtype=np.float32)
        return zeros, zeros

    coords = coords.astype(np.float32)
    values = norm[mask].astype(np.float32)
    if coords.shape[0] > max_points:
        step = int(math.ceil(coords.shape[0] / max_points))
        coords = coords[::step]
        values = values[::step]

    centroid = coords.mean(axis=0)
    centered = coords - centroid
    cov = (centered.T @ centered) / max(1, coords.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 1e-6)
    eigvecs = eigvecs[:, order]

    projected = centered @ eigvecs
    scaled = np.abs(projected / (np.sqrt(eigvals)[None, :] + 1e-6))
    scaled = np.clip(scaled, 0.0, 3.5)

    hist, _ = np.histogramdd(scaled, bins=bins, range=((0.0, 3.5), (0.0, 3.5), (0.0, 3.5)))
    raw_hist, _ = np.histogramdd(
        scaled,
        bins=bins,
        range=((0.0, 3.5), (0.0, 3.5), (0.0, 3.5)),
        weights=values,
    )

    hist = hist.astype(np.float32).ravel()
    raw_hist = raw_hist.astype(np.float32).ravel()
    hist /= float(hist.sum() + 1e-6)
    raw_hist /= float(raw_hist.sum() + 1e-6)
    return hist, raw_hist


def extract_feature_npz(path: Path) -> dict[str, np.ndarray]:
    img = nib.load(str(path))
    volume = np.asanyarray(img.dataobj).astype(np.float32)
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    if volume.ndim > 3:
        volume = volume[..., 0]
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D image, got {volume.shape} for {path}")

    finite = np.isfinite(volume)
    nonzero = finite & (np.abs(volume) > 1e-6)
    if int(nonzero.sum()) < 256:
        nonzero = finite
    bbox = bbox_from_mask(nonzero, margin=2)

    norm, lo_hi = robust_normalize(volume, nonzero)
    crop = norm[bbox]
    crop_mask = nonzero[bbox].astype(np.float32)

    raw_grid32 = resize_nd(norm, (32, 32, 32), order=1)
    raw_crop32 = resize_nd(crop, (32, 32, 32), order=1)
    mask_crop32 = resize_nd(crop_mask, (32, 32, 32), order=0)

    edge_grid32 = gradient_magnitude(raw_grid32)
    edge_crop32 = gradient_magnitude(raw_crop32)

    projections = []
    mask_projections = []
    for axis in range(3):
        projections.append(resize_2d(crop.max(axis=axis), 64, order=1))
        projections.append(resize_2d(crop.mean(axis=axis), 64, order=1))
        mask_projections.append(resize_2d(crop_mask.mean(axis=axis), 64, order=1))
    proj64 = np.concatenate([x.ravel() for x in projections]).astype(np.float32)
    mask_proj64 = np.concatenate([x.ravel() for x in mask_projections]).astype(np.float32)

    hist, _ = np.histogram(crop[crop_mask > 0.5], bins=64, range=(0.0, 1.0), density=False)
    hist64 = hist.astype(np.float32)
    hist64 /= float(hist64.sum() + 1e-6)
    shape_moments = foreground_shape_moments(nonzero, norm)
    pca_abs_mask24, pca_abs_raw24 = foreground_pca_abs_hists(nonzero, norm)

    shape = np.asarray(volume.shape, dtype=np.float32)
    crop_shape = np.asarray(crop.shape, dtype=np.float32)
    zooms = np.asarray(img.header.get_zooms()[:3], dtype=np.float32)
    affine = np.asarray(img.affine, dtype=np.float32)
    affine_diag = np.linalg.norm(affine[:3, :3], axis=0).astype(np.float32)
    fg_fraction = np.asarray([float(nonzero.mean())], dtype=np.float32)
    meta12 = np.concatenate(
        [
            shape / 256.0,
            crop_shape / 256.0,
            zooms / 4.0,
            affine_diag / 4.0,
            fg_fraction,
            lo_hi / (np.abs(lo_hi).max() + 1e-6),
        ]
    ).astype(np.float32)

    return {
        "raw_grid32": raw_grid32.ravel().astype(np.float32),
        "raw_crop32": raw_crop32.ravel().astype(np.float32),
        "edge_grid32": edge_grid32.ravel().astype(np.float32),
        "edge_crop32": edge_crop32.ravel().astype(np.float32),
        "mask_crop32": mask_crop32.ravel().astype(np.float32),
        "proj64": proj64,
        "mask_proj64": mask_proj64,
        "shape_moments": shape_moments,
        "pca_abs_mask24": pca_abs_mask24,
        "pca_abs_raw24": pca_abs_raw24,
        "hist64": hist64,
        "meta12": meta12,
    }


def cache_path(cache_dir: Path, image_id: str) -> Path:
    return cache_dir / "features" / f"{image_id}.npz"


def ensure_features(images: dict[str, Path], cache_dir: Path, force: bool = False) -> None:
    feature_dir = cache_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    ids = sorted(images)
    for index, image_id in enumerate(ids, start=1):
        out = cache_path(cache_dir, image_id)
        if out.exists() and not force:
            try:
                with np.load(out) as cached:
                    if all(feature in cached for feature in FEATURES):
                        continue
            except Exception:
                pass
        feats = extract_feature_npz(images[image_id])
        np.savez_compressed(out, **feats)
        if index == 1 or index % 25 == 0 or index == len(ids):
            print(f"cached {index}/{len(ids)}: {image_id}", flush=True)


def load_feature(image_id: str, cache_dir: Path, feature: str) -> np.ndarray:
    with np.load(cache_path(cache_dir, image_id)) as data:
        return data[feature].astype(np.float32)


def feature_matrix(ids: list[str], cache_dir: Path, feature: str) -> np.ndarray:
    return np.stack([load_feature(image_id, cache_dir, feature) for image_id in ids]).astype(np.float32)


def concat_matrix(ids: list[str], cache_dir: Path, features: tuple[str, ...]) -> np.ndarray:
    parts = [feature_matrix(ids, cache_dir, feature) for feature in features]
    return np.concatenate(parts, axis=1).astype(np.float32)


def row_normalize(x: np.ndarray, center: bool = True) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    if center:
        x = x - x.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-6
    return x / denom


def feature_centered(feature: str) -> bool:
    return feature not in UNCENTERED_FEATURES


def feature_vectors(ids: list[str], cache_dir: Path, feature: str) -> np.ndarray:
    return row_normalize(feature_matrix(ids, cache_dir, feature), center=feature_centered(feature))


def fusion_vectors(ids: list[str], cache_dir: Path, fusion: str) -> np.ndarray:
    parts = [feature_vectors(ids, cache_dir, feature) for feature in FUSIONS[fusion]]
    scale = 1.0 / math.sqrt(len(parts))
    return row_normalize(np.concatenate([part * scale for part in parts], axis=1), center=False)


def cosine_scores(q: np.ndarray, t: np.ndarray, feature: str) -> np.ndarray:
    center = feature_centered(feature)
    qn = row_normalize(q, center=center)
    tn = row_normalize(t, center=center)
    return (qn @ tn.T).astype(np.float32)


def zscore_by_query(scores: np.ndarray) -> np.ndarray:
    return (scores - scores.mean(axis=1, keepdims=True)) / (scores.std(axis=1, keepdims=True) + 1e-6)


def rank_metrics(scores: np.ndarray, query_ids: list[str], target_ids: list[str], truth: dict[str, str]) -> dict[str, float]:
    target_index = {target_id: i for i, target_id in enumerate(target_ids)}
    ranks = []
    for qi, query_id in enumerate(query_ids):
        true_target = truth[query_id]
        ti = target_index[true_target]
        true_score = scores[qi, ti]
        rank = 1 + int(np.sum(scores[qi] > true_score))
        ranks.append(rank)
    ranks_arr = np.asarray(ranks, dtype=np.float32)
    return {
        "mrr": float(np.mean(1.0 / ranks_arr)),
        "top1": float(np.mean(ranks_arr <= 1)),
        "top3": float(np.mean(ranks_arr <= 3)),
        "top5": float(np.mean(ranks_arr <= 5)),
        "median_rank": float(np.median(ranks_arr)),
        "mean_rank": float(np.mean(ranks_arr)),
    }


def score_feature(query_ids: list[str], target_ids: list[str], cache_dir: Path, feature: str) -> np.ndarray:
    return cosine_scores(feature_matrix(query_ids, cache_dir, feature), feature_matrix(target_ids, cache_dir, feature), feature)


def score_fusion(query_ids: list[str], target_ids: list[str], cache_dir: Path, fusion: str) -> np.ndarray:
    matrices = []
    for feature in FUSIONS[fusion]:
        matrices.append(zscore_by_query(score_feature(query_ids, target_ids, cache_dir, feature)))
    return np.mean(matrices, axis=0).astype(np.float32)


def fit_pca_ridge(
    train_query_ids: list[str],
    train_target_ids: list[str],
    cache_dir: Path,
    features: tuple[str, ...],
    n_components: int,
    alpha: float,
) -> tuple[PCA, PCA, Ridge]:
    x = concat_matrix(train_query_ids, cache_dir, features)
    y = concat_matrix(train_target_ids, cache_dir, features)
    components = max(2, min(n_components, len(train_query_ids) - 1, x.shape[1], y.shape[1]))
    q_pca = PCA(n_components=components, svd_solver="randomized", random_state=SEED)
    t_pca = PCA(n_components=components, svd_solver="randomized", random_state=SEED + 1)
    xp = q_pca.fit_transform(x)
    yp = t_pca.fit_transform(y)
    ridge = Ridge(alpha=alpha)
    ridge.fit(xp, yp)
    return q_pca, t_pca, ridge


def score_pca_ridge(
    query_ids: list[str],
    target_ids: list[str],
    cache_dir: Path,
    q_pca: PCA,
    t_pca: PCA,
    ridge: Ridge,
    features: tuple[str, ...],
) -> np.ndarray:
    xq = q_pca.transform(concat_matrix(query_ids, cache_dir, features))
    yt = t_pca.transform(concat_matrix(target_ids, cache_dir, features))
    pred = ridge.predict(xq)
    return (row_normalize(pred, center=True) @ row_normalize(yt, center=True).T).astype(np.float32)


def pca_ridge_vectors(
    query_ids: list[str],
    target_ids: list[str],
    cache_dir: Path,
    q_pca: PCA,
    t_pca: PCA,
    ridge: Ridge,
    features: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    xq = q_pca.transform(concat_matrix(query_ids, cache_dir, features))
    yt = t_pca.transform(concat_matrix(target_ids, cache_dir, features))
    pred = ridge.predict(xq)
    return row_normalize(pred, center=True), row_normalize(yt, center=True)


def build_submission_vectors(
    query_ids: list[str],
    target_ids: list[str],
    cache_dir: Path,
    classical_fusion: str,
    classical_weight: float,
    pca_weight: float,
    pca_model: tuple[PCA, PCA, Ridge] | None,
) -> tuple[np.ndarray, np.ndarray]:
    query_parts = [fusion_vectors(query_ids, cache_dir, classical_fusion) * classical_weight]
    target_parts = [fusion_vectors(target_ids, cache_dir, classical_fusion) * classical_weight]
    if pca_model is not None and pca_weight > 0.0:
        q_pca, t_pca, ridge = pca_model
        query_pca, target_pca = pca_ridge_vectors(query_ids, target_ids, cache_dir, q_pca, t_pca, ridge, PCA_FEATURES)
        query_parts.append(query_pca * pca_weight)
        target_parts.append(target_pca * pca_weight)
    query_vecs = row_normalize(np.concatenate(query_parts, axis=1), center=False)
    target_vecs = row_normalize(np.concatenate(target_parts, axis=1), center=False)
    return query_vecs, target_vecs


def save_pool_vectors(
    path: Path,
    pool: Pool,
    query_ids: list[str],
    target_ids: list[str],
    query_vectors: np.ndarray,
    target_vectors: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        dataset=np.asarray(pool.dataset),
        split=np.asarray(pool.split),
        query_ids=np.asarray(query_ids),
        target_ids=np.asarray(target_ids),
        query_vectors=query_vectors.astype(np.float32),
        target_vectors=target_vectors.astype(np.float32),
    )


def cv_pca_ridge(
    pairs: list[dict[str, str]],
    target_ids: list[str],
    cache_dir: Path,
    n_splits: int,
    n_components: int,
    alpha: float,
) -> tuple[np.ndarray, list[str]]:
    pair_query_ids = [row["query_id"] for row in pairs]
    pair_target_ids = [row["target_id"] for row in pairs]
    all_scores = np.zeros((len(pair_query_ids), len(target_ids)), dtype=np.float32)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    indices = np.arange(len(pairs))
    for fold, (train_idx, test_idx) in enumerate(kfold.split(indices), start=1):
        train_query_ids = [pair_query_ids[i] for i in train_idx]
        train_target_ids = [pair_target_ids[i] for i in train_idx]
        test_query_ids = [pair_query_ids[i] for i in test_idx]
        q_pca, t_pca, ridge = fit_pca_ridge(
            train_query_ids,
            train_target_ids,
            cache_dir,
            PCA_FEATURES,
            n_components=n_components,
            alpha=alpha,
        )
        fold_scores = score_pca_ridge(test_query_ids, target_ids, cache_dir, q_pca, t_pca, ridge, PCA_FEATURES)
        all_scores[test_idx] = fold_scores
        print(f"pca_ridge fold {fold}/{n_splits} done", flush=True)
    return all_scores, pair_query_ids


def collect_train_images(pairs: list[dict[str, str]]) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for row in pairs:
        images[row["query_id"]] = Path(row["query_path"])
        images[row["target_id"]] = Path(row["target_path"])
    return images


def command_inspect(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    pairs = load_train_pairs(data_root)
    images = collect_train_images(pairs)
    sample_ids = sorted(images)[: int(args.sample)]
    report = []
    for image_id in sample_ids:
        path = images[image_id]
        img = nib.load(str(path))
        report.append(
            {
                "id": image_id,
                "path": str(path.relative_to(data_root)),
                "shape": [int(x) for x in img.shape[:3]],
                "zooms": [float(x) for x in img.header.get_zooms()[:3]],
                "qform_code": int(img.header["qform_code"]),
                "sform_code": int(img.header["sform_code"]),
                "affine_diag_norm": [
                    float(x) for x in np.linalg.norm(np.asarray(img.affine)[:3, :3], axis=0).round(4)
                ],
            }
        )
    print(json.dumps({"num_train_pairs": len(pairs), "sample": report}, indent=2))


def command_cache(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    cache_dir = args.cache_dir.resolve()
    images: dict[str, Path] = {}
    if args.scope in {"train", "all"}:
        images.update(collect_train_images(load_train_pairs(data_root)))
    if args.scope in {"predict", "all"}:
        for pool in all_prediction_pools(data_root, tuple(args.datasets), tuple(args.splits)):
            images.update(load_query_manifest(data_root, pool.query_csv))
            images.update(load_gallery_manifest(data_root, pool.gallery_csv))
    print(f"caching {len(images)} images into {cache_dir}")
    ensure_features(images, cache_dir, force=args.force)


def command_cv(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    cache_dir = args.cache_dir.resolve()
    pairs = load_train_pairs(data_root)
    images = collect_train_images(pairs)
    ensure_features(images, cache_dir, force=False)

    query_ids = [row["query_id"] for row in pairs]
    target_ids = [row["target_id"] for row in pairs]
    truth = {row["query_id"]: row["target_id"] for row in pairs}

    rows = []
    for feature in FEATURES:
        scores = score_feature(query_ids, target_ids, cache_dir, feature)
        metrics = rank_metrics(scores, query_ids, target_ids, truth)
        row = {"method": feature, **metrics}
        rows.append(row)
        print(json.dumps(row), flush=True)

    for fusion in FUSIONS:
        scores = score_fusion(query_ids, target_ids, cache_dir, fusion)
        metrics = rank_metrics(scores, query_ids, target_ids, truth)
        row = {"method": fusion, **metrics}
        rows.append(row)
        print(json.dumps(row), flush=True)

    for components in args.pca_components:
        for alpha in args.pca_alpha:
            scores, cv_query_ids = cv_pca_ridge(
                pairs,
                target_ids,
                cache_dir,
                n_splits=args.splits,
                n_components=components,
                alpha=alpha,
            )
            metrics = rank_metrics(scores, cv_query_ids, target_ids, truth)
            row = {"method": f"pca_ridge_c{components}_a{alpha:g}", **metrics}
            rows.append(row)
            print(json.dumps(row), flush=True)

    rows = sorted(rows, key=lambda r: (-float(r["mrr"]), float(r["mean_rank"])))
    out = args.out or (cache_dir / "cv_results.csv")
    write_csv(
        out,
        [{k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()} for row in rows],
        ["method", "mrr", "top1", "top3", "top5", "median_rank", "mean_rank"],
    )
    print(f"wrote {out}")
    print("best methods:")
    for row in rows[:10]:
        print(json.dumps(row), flush=True)


def ranking_rows_from_scores(query_ids: list[str], target_ids: list[str], scores: np.ndarray) -> list[dict[str, str]]:
    rows = []
    target_ids_arr = np.asarray(target_ids)
    for qi, query_id in enumerate(query_ids):
        order = np.argsort(-scores[qi], kind="mergesort")
        rows.append({"query_id": query_id, "target_id_ranking": " ".join(target_ids_arr[order].tolist())})
    return rows


def score_method(
    method: str,
    query_ids: list[str],
    target_ids: list[str],
    cache_dir: Path,
    pca_model: tuple[PCA, PCA, Ridge] | None,
) -> np.ndarray:
    if method in FEATURES:
        return score_feature(query_ids, target_ids, cache_dir, method)
    if method in FUSIONS:
        return score_fusion(query_ids, target_ids, cache_dir, method)
    if method == "pca_ridge":
        if pca_model is None:
            raise ValueError("pca_ridge requires a fitted model")
        q_pca, t_pca, ridge = pca_model
        return score_pca_ridge(query_ids, target_ids, cache_dir, q_pca, t_pca, ridge, PCA_FEATURES)
    if method == "fusion_default_plus_pca":
        if pca_model is None:
            raise ValueError("fusion_default_plus_pca requires a fitted model")
        classical = zscore_by_query(score_fusion(query_ids, target_ids, cache_dir, "fusion_default"))
        pca_scores = zscore_by_query(score_method("pca_ridge", query_ids, target_ids, cache_dir, pca_model))
        return (0.65 * classical + 0.35 * pca_scores).astype(np.float32)
    raise ValueError(f"Unknown method: {method}")


def command_predict(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    cache_dir = args.cache_dir.resolve()
    train_pairs = load_train_pairs(data_root)

    images = collect_train_images(train_pairs)
    pools = all_prediction_pools(data_root, tuple(args.datasets), tuple(args.splits))
    pool_manifests = []
    for pool in pools:
        queries = load_query_manifest(data_root, pool.query_csv)
        targets = load_gallery_manifest(data_root, pool.gallery_csv)
        images.update(queries)
        images.update(targets)
        pool_manifests.append((pool, queries, targets))
    ensure_features(images, cache_dir, force=False)

    pca_model = None
    if args.method in {"pca_ridge", "fusion_default_plus_pca"}:
        pca_model = fit_pca_ridge(
            [row["query_id"] for row in train_pairs],
            [row["target_id"] for row in train_pairs],
            cache_dir,
            PCA_FEATURES,
            n_components=args.pca_components,
            alpha=args.pca_alpha,
        )

    submission_rows: list[dict[str, str]] = []
    for pool, queries, targets in pool_manifests:
        query_ids = sorted(queries)
        target_ids = sorted(targets)
        scores = score_method(args.method, query_ids, target_ids, cache_dir, pca_model)
        submission_rows.extend(ranking_rows_from_scores(query_ids, target_ids, scores))
        print(f"ranked {pool.dataset}/{pool.split}: {len(query_ids)}x{len(target_ids)}", flush=True)

    out = args.out or (DEFAULT_OUT_DIR / f"{args.method}.csv")
    write_csv(out, submission_rows, ["query_id", "target_id_ranking"])
    print(f"wrote {len(submission_rows)} rows to {out}")


def command_predict_mix(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    cache_dir = args.cache_dir.resolve()
    train_pairs = load_train_pairs(data_root)
    method_by_dataset = {
        "dataset1": args.dataset1_method,
        "dataset2": args.dataset2_method,
        "dataset3": args.dataset3_method,
    }

    images = collect_train_images(train_pairs)
    pool_manifests = []
    for pool in all_prediction_pools(data_root, tuple(args.datasets), tuple(args.splits)):
        queries = load_query_manifest(data_root, pool.query_csv)
        targets = load_gallery_manifest(data_root, pool.gallery_csv)
        images.update(queries)
        images.update(targets)
        pool_manifests.append((pool, queries, targets))
    ensure_features(images, cache_dir, force=False)

    pca_model = None
    if any(method in {"pca_ridge", "fusion_default_plus_pca"} for method in method_by_dataset.values()):
        pca_model = fit_pca_ridge(
            [row["query_id"] for row in train_pairs],
            [row["target_id"] for row in train_pairs],
            cache_dir,
            PCA_FEATURES,
            n_components=args.pca_components,
            alpha=args.pca_alpha,
        )

    submission_rows: list[dict[str, str]] = []
    for pool, queries, targets in pool_manifests:
        method = method_by_dataset[pool.dataset]
        query_ids = sorted(queries)
        target_ids = sorted(targets)
        scores = score_method(method, query_ids, target_ids, cache_dir, pca_model)
        submission_rows.extend(ranking_rows_from_scores(query_ids, target_ids, scores))
        print(f"ranked {pool.dataset}/{pool.split} with {method}: {len(query_ids)}x{len(target_ids)}", flush=True)

    write_csv(args.out, submission_rows, ["query_id", "target_id_ranking"])
    print(f"wrote {len(submission_rows)} rows to {args.out}")


def command_embed_submit(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    cache_dir = args.cache_dir.resolve()
    vectors_dir = args.vectors_dir.resolve()
    train_pairs = load_train_pairs(data_root)

    images = collect_train_images(train_pairs)
    pool_manifests = []
    for pool in all_prediction_pools(data_root, tuple(args.datasets), tuple(args.splits)):
        queries = load_query_manifest(data_root, pool.query_csv)
        targets = load_gallery_manifest(data_root, pool.gallery_csv)
        images.update(queries)
        images.update(targets)
        pool_manifests.append((pool, queries, targets))
    ensure_features(images, cache_dir, force=False)

    pca_model = None
    if args.pca_weight > 0.0:
        pca_model = fit_pca_ridge(
            [row["query_id"] for row in train_pairs],
            [row["target_id"] for row in train_pairs],
            cache_dir,
            PCA_FEATURES,
            n_components=args.pca_components,
            alpha=args.pca_alpha,
        )

    submission_rows: list[dict[str, str]] = []
    for pool, queries, targets in pool_manifests:
        query_ids = sorted(queries)
        target_ids = sorted(targets)
        query_vectors, target_vectors = build_submission_vectors(
            query_ids,
            target_ids,
            cache_dir,
            args.classical_fusion,
            args.classical_weight,
            args.pca_weight,
            pca_model,
        )
        scores = (query_vectors @ target_vectors.T).astype(np.float32)
        submission_rows.extend(ranking_rows_from_scores(query_ids, target_ids, scores))
        save_pool_vectors(
            vectors_dir / f"{pool.dataset}_{pool.split}_vectors.npz",
            pool,
            query_ids,
            target_ids,
            query_vectors,
            target_vectors,
        )
        print(
            f"embedded {pool.dataset}/{pool.split}: "
            f"{len(query_ids)} queries, {len(target_ids)} targets, dim={query_vectors.shape[1]}",
            flush=True,
        )

    write_csv(args.out, submission_rows, ["query_id", "target_id_ranking"])
    print(f"wrote {len(submission_rows)} rows to {args.out}")
    print(f"saved vectors to {vectors_dir}")


def command_validate(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    rows = read_csv(args.submission)
    by_query = {row["query_id"]: row["target_id_ranking"].split() for row in rows}

    expected: dict[str, tuple[str, str, set[str]]] = {}
    for pool in all_prediction_pools(data_root):
        queries = load_query_manifest(data_root, pool.query_csv)
        targets = set(load_gallery_manifest(data_root, pool.gallery_csv))
        for query_id in queries:
            expected[query_id] = (pool.dataset, pool.split, targets)

    errors = []
    counts: dict[str, int] = {}
    for query_id, ranking in by_query.items():
        if query_id not in expected:
            errors.append(f"unexpected query_id {query_id}")
            continue
        dataset, split, targets = expected[query_id]
        key = f"{dataset}/{split}"
        counts[key] = counts.get(key, 0) + 1
        ranking_set = set(ranking)
        if len(ranking) != len(targets):
            errors.append(f"{query_id}: ranking length {len(ranking)} != {len(targets)}")
        if ranking_set != targets:
            missing = len(targets - ranking_set)
            extra = len(ranking_set - targets)
            errors.append(f"{query_id}: target set mismatch missing={missing} extra={extra}")
        if len(ranking_set) != len(ranking):
            errors.append(f"{query_id}: duplicate target IDs in ranking")

    if not args.allow_partial:
        missing_queries = sorted(set(expected) - set(by_query))
        if missing_queries:
            errors.append(f"missing {len(missing_queries)} expected queries")

    report = {
        "submission": str(args.submission),
        "rows": len(rows),
        "counts": counts,
        "allow_partial": bool(args.allow_partial),
        "errors": errors[:20],
        "num_errors": len(errors),
    }
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_p = sub.add_parser("inspect")
    inspect_p.add_argument("--sample", type=int, default=8)
    inspect_p.set_defaults(func=command_inspect)

    cache_p = sub.add_parser("cache")
    cache_p.add_argument("--scope", choices=["train", "predict", "all"], default="train")
    cache_p.add_argument("--datasets", nargs="+", choices=["dataset1", "dataset2", "dataset3"], default=["dataset1", "dataset2", "dataset3"])
    cache_p.add_argument("--splits", nargs="+", choices=["val", "test"], default=["val", "test"])
    cache_p.add_argument("--force", action="store_true")
    cache_p.set_defaults(func=command_cache)

    cv_p = sub.add_parser("cv")
    cv_p.add_argument("--splits", type=int, default=5)
    cv_p.add_argument("--pca-components", type=int, nargs="+", default=[32, 64, 128])
    cv_p.add_argument("--pca-alpha", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0])
    cv_p.add_argument("--out", type=Path)
    cv_p.set_defaults(func=command_cv)

    pred_p = sub.add_parser("predict")
    pred_p.add_argument(
        "--method",
        choices=list(FEATURES) + list(FUSIONS) + ["pca_ridge", "fusion_default_plus_pca"],
        default="fusion_default",
    )
    pred_p.add_argument("--datasets", nargs="+", choices=["dataset1", "dataset2", "dataset3"], default=["dataset1", "dataset2", "dataset3"])
    pred_p.add_argument("--splits", nargs="+", choices=["val", "test"], default=["val", "test"])
    pred_p.add_argument("--pca-components", type=int, default=64)
    pred_p.add_argument("--pca-alpha", type=float, default=10.0)
    pred_p.add_argument("--out", type=Path)
    pred_p.set_defaults(func=command_predict)

    mix_p = sub.add_parser("predict-mix")
    method_choices = list(FEATURES) + list(FUSIONS) + ["pca_ridge", "fusion_default_plus_pca"]
    mix_p.add_argument("--dataset1-method", choices=method_choices, default="pca_ridge")
    mix_p.add_argument("--dataset2-method", choices=method_choices, required=True)
    mix_p.add_argument("--dataset3-method", choices=method_choices, required=True)
    mix_p.add_argument("--datasets", nargs="+", choices=["dataset1", "dataset2", "dataset3"], default=["dataset1", "dataset2", "dataset3"])
    mix_p.add_argument("--splits", nargs="+", choices=["val", "test"], default=["val", "test"])
    mix_p.add_argument("--pca-components", type=int, default=128)
    mix_p.add_argument("--pca-alpha", type=float, default=100.0)
    mix_p.add_argument("--out", type=Path, required=True)
    mix_p.set_defaults(func=command_predict_mix)

    embed_p = sub.add_parser("embed-submit")
    embed_p.add_argument("--classical-fusion", choices=list(FUSIONS), default="fusion_default")
    embed_p.add_argument("--datasets", nargs="+", choices=["dataset1", "dataset2", "dataset3"], default=["dataset1", "dataset2", "dataset3"])
    embed_p.add_argument("--splits", nargs="+", choices=["val", "test"], default=["val", "test"])
    embed_p.add_argument("--classical-weight", type=float, default=0.65)
    embed_p.add_argument("--pca-weight", type=float, default=0.35)
    embed_p.add_argument("--pca-components", type=int, default=128)
    embed_p.add_argument("--pca-alpha", type=float, default=100.0)
    embed_p.add_argument("--vectors-dir", type=Path, default=Path("artifacts/vectors/fusion_default_plus_pca"))
    embed_p.add_argument("--out", type=Path, default=Path("submissions/cosine_vectors_fusion_default_plus_pca.csv"))
    embed_p.set_defaults(func=command_embed_submit)

    validate_p = sub.add_parser("validate")
    validate_p.add_argument("submission", type=Path)
    validate_p.add_argument("--allow-partial", action="store_true")
    validate_p.set_defaults(func=command_validate)
    return parser.parse_args()


def main() -> None:
    np.random.seed(SEED)
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    args.func(args)


if __name__ == "__main__":
    main()
