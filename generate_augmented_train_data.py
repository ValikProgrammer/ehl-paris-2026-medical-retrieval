from __future__ import annotations
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "nibabel>=5.3",
#   "numpy>=1.26",
#   "scipy>=1.11",
#   "tqdm>=4.67",
# ]
# ///

import argparse
import csv
import math
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from tqdm import tqdm


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(data_root: Path, image_path: str) -> Path:
    path = Path(image_path)
    resolved = path if path.is_absolute() else data_root / path
    if resolved.exists():
        return resolved
    if resolved.name.endswith(".nii.gz"):
        nii_path = resolved.with_suffix("")
        if nii_path.exists():
            return nii_path
    return resolved


def rotation_matrix(angles: np.ndarray) -> np.ndarray:
    ax, ay, az = angles
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return rz @ ry @ rx


def foreground_mask(volume: np.ndarray) -> np.ndarray:
    finite = np.isfinite(volume)
    if not finite.any():
        return np.zeros_like(volume, dtype=bool)
    nonzero = np.abs(volume) > 1e-6
    return finite & nonzero


def apply_affine(volume: np.ndarray, rng: np.random.Generator, preset: str) -> np.ndarray:
    if preset == "geom_contrast_stronger":
        max_rotate = math.radians(22.0)
        min_scale, max_scale = 0.85, 1.15
        max_shift = 16.0
    else:
        max_rotate = math.radians(18.0)
        min_scale, max_scale = 0.90, 1.10
        max_shift = 12.0
    angles = rng.uniform(-max_rotate, max_rotate, size=3)
    scales = rng.uniform(min_scale, max_scale, size=3).astype(np.float32)
    shifts = rng.uniform(-max_shift, max_shift, size=3).astype(np.float32)

    transform = rotation_matrix(angles) @ np.diag(scales)
    inverse = np.linalg.inv(transform)
    center = (np.asarray(volume.shape, dtype=np.float32) - 1.0) / 2.0
    offset = center - inverse @ (center + shifts)
    return ndimage.affine_transform(
        volume,
        matrix=inverse,
        offset=offset,
        order=1,
        mode="nearest",
        prefilter=False,
    ).astype(np.float32, copy=False)


def random_smooth_field(shape: tuple[int, int, int], rng: np.random.Generator) -> np.ndarray:
    coarse_shape = tuple(max(4, int(math.ceil(dim / 32))) for dim in shape)
    field = rng.normal(0.0, 1.0, size=coarse_shape).astype(np.float32)
    zoom = tuple(dim / coarse for dim, coarse in zip(shape, coarse_shape))
    field = ndimage.zoom(field, zoom=zoom, order=3)
    field = field[tuple(slice(0, dim) for dim in shape)]
    field = ndimage.gaussian_filter(field, sigma=2.0)
    std = float(field.std())
    if std > 1e-6:
        field = field / std
    return field.astype(np.float32, copy=False)


def apply_elastic(volume: np.ndarray, rng: np.random.Generator, preset: str) -> np.ndarray:
    if preset == "geom_contrast_stronger":
        max_disp = float(rng.uniform(4.0, 10.0))
    else:
        max_disp = float(rng.uniform(3.0, 8.0))
    displacements = [random_smooth_field(volume.shape, rng) * max_disp for _ in range(3)]
    coords = np.meshgrid(
        *[np.arange(dim, dtype=np.float32) for dim in volume.shape],
        indexing="ij",
    )
    warped_coords = [coord + disp for coord, disp in zip(coords, displacements)]
    return ndimage.map_coordinates(
        volume,
        warped_coords,
        order=1,
        mode="nearest",
        prefilter=False,
    ).astype(np.float32, copy=False)


def apply_bias_field(volume: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shape = volume.shape
    coarse_shape = tuple(max(4, int(math.ceil(dim / 64))) for dim in shape)
    field = rng.normal(0.0, 1.0, size=coarse_shape).astype(np.float32)
    zoom = tuple(dim / coarse for dim, coarse in zip(shape, coarse_shape))
    field = ndimage.zoom(field, zoom=zoom, order=3)
    field = field[tuple(slice(0, dim) for dim in shape)]
    field = ndimage.gaussian_filter(field, sigma=8.0)
    max_abs = float(np.max(np.abs(field)))
    if max_abs > 1e-6:
        field = field / max_abs
    multiplier = 1.0 + rng.uniform(0.10, 0.25) * field
    return (volume * multiplier).astype(np.float32, copy=False)


def apply_contrast(volume: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mask = foreground_mask(volume)
    if not mask.any():
        return volume
    values = volume[mask]
    lo, hi = np.percentile(values, [1.0, 99.0])
    if hi <= lo + 1e-6:
        return volume
    gamma = float(rng.uniform(0.75, 1.35))
    out = volume.copy()
    normalized = np.clip((out[mask] - lo) / (hi - lo), 0.0, 1.0)
    out[mask] = np.power(normalized, gamma) * (hi - lo) + lo
    return out.astype(np.float32, copy=False)


def apply_intensity_noise(volume: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mask = foreground_mask(volume)
    if not mask.any():
        return volume
    values = volume[mask]
    lo, hi = np.percentile(values, [1.0, 99.0])
    scale = max(float(hi - lo), 1e-6)
    out = volume.copy()
    out[mask] = out[mask] * float(rng.uniform(0.90, 1.10)) + float(rng.uniform(-0.05, 0.05) * scale)
    out += rng.normal(0.0, 0.025 * scale, size=volume.shape).astype(np.float32)
    return out.astype(np.float32, copy=False)


def augment_volume(volume: np.ndarray, rng: np.random.Generator, preset: str) -> np.ndarray:
    out = np.nan_to_num(volume.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    out = apply_affine(out, rng, preset)
    elastic_probability = 0.60 if preset == "geom_contrast_stronger" else 0.45
    if rng.random() < elastic_probability:
        out = apply_elastic(out, rng, preset)
    if preset in {"geom_contrast", "geom_contrast_stronger"}:
        intensity_probability = 0.40 if preset == "geom_contrast_stronger" else 0.30
        if rng.random() < intensity_probability:
            out = apply_bias_field(out, rng)
        if rng.random() < intensity_probability:
            out = apply_contrast(out, rng)
        if rng.random() < intensity_probability:
            out = apply_intensity_noise(out, rng)
    return out.astype(np.float32, copy=False)


def load_nifti(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.load(str(path))
    volume = np.asanyarray(image.dataobj).astype(np.float32)
    if volume.ndim > 3:
        volume = volume[..., 0]
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D image, got {volume.shape} for {path}")
    return volume, image


def save_nifti(path: Path, volume: np.ndarray, reference: nib.Nifti1Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    image = nib.Nifti1Image(volume.astype(np.float32, copy=False), reference.affine, header)
    nib.save(image, str(path))


def relative_to_data_root(path: Path, data_root: Path) -> str:
    try:
        return str(path.relative_to(data_root))
    except ValueError:
        return str(path)


def generated_row(
    row: dict[str, str],
    copy_index: int,
    query_rel_path: str,
    target_rel_path: str,
) -> dict[str, str]:
    suffix = f"aug{copy_index:02d}"
    out = dict(row)
    out["pair_id"] = f"{row.get('pair_id') or row['query_id']}_{suffix}"
    out["query_id"] = f"{row['query_id']}_{suffix}"
    out["target_id"] = f"{row['target_id']}_{suffix}"
    out["query_image"] = query_rel_path
    out["target_image"] = target_rel_path
    out["dataset"] = f"{row.get('dataset', 'dataset1')}_augmented"
    return out


def original_row(row: dict[str, str], data_root: Path) -> dict[str, str]:
    out = dict(row)
    out["query_image"] = relative_to_data_root(resolve_path(data_root, row["query_image"]), data_root)
    out["target_image"] = relative_to_data_root(resolve_path(data_root, row["target_image"]), data_root)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate augmented NIfTI train pairs from dataset1/train_pairs.csv.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-pair-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--copies", type=int, default=2)
    parser.add_argument("--copy-start", type=int, default=0)
    parser.add_argument("--preset", choices=["geom", "geom_contrast", "geom_contrast_stronger"], default="geom_contrast")
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--start-index", type=int, default=0, help="Debug only: start from this zero-based input row.")
    parser.add_argument("--limit", type=int, default=None, help="Debug only: generate from the first N pairs.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-original", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-shards", type=int, default=1, help="Split input rows across this many independent jobs.")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based shard index to generate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    all_rows = read_csv(args.train_pair_csv)
    indexed_rows = list(enumerate(all_rows))
    if args.start_index:
        indexed_rows = indexed_rows[args.start_index :]
    if args.limit is not None:
        indexed_rows = indexed_rows[: args.limit]
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    if args.num_shards > 1:
        indexed_rows = [(index, row) for index, row in indexed_rows if index % args.num_shards == args.shard_index]
    rows = [row for _, row in indexed_rows]
    if not rows:
        raise ValueError("No training rows selected.")

    fieldnames = list(all_rows[0].keys())
    output_rows: list[dict[str, str]] = []
    if args.include_original:
        output_rows.extend(original_row(row, data_root) for row in rows)

    total = len(rows) * args.copies
    with tqdm(total=total, desc="Generating augmented pairs") as progress:
        for row_index, row in indexed_rows:
            query_path = resolve_path(data_root, row["query_image"])
            target_path = resolve_path(data_root, row["target_image"])
            query_volume, query_reference = load_nifti(query_path)
            target_volume, target_reference = load_nifti(target_path)

            for copy_offset in range(args.copies):
                copy_index = args.copy_start + copy_offset
                rng = np.random.default_rng(args.seed + row_index * 1009 + copy_index * 9176)
                query_out = output_dir / "images" / "train" / "queries" / f"{row['query_id']}_aug{copy_index:02d}.nii.gz"
                target_out = output_dir / "images" / "train" / "gallery" / f"{row['target_id']}_aug{copy_index:02d}.nii.gz"

                if args.overwrite or not query_out.exists():
                    save_nifti(query_out, augment_volume(query_volume, rng, args.preset), query_reference)
                if args.overwrite or not target_out.exists():
                    save_nifti(target_out, augment_volume(target_volume, rng, args.preset), target_reference)

                output_rows.append(
                    generated_row(
                        row,
                        copy_index,
                        relative_to_data_root(query_out, data_root),
                        relative_to_data_root(target_out, data_root),
                    )
                )
                progress.update(1)

    write_csv(args.output_csv, output_rows, fieldnames)
    print(f"Wrote {len(output_rows)} rows to {args.output_csv}")
    print(f"Augmented images are under {output_dir}")


if __name__ == "__main__":
    main()
