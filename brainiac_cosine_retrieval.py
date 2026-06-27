from __future__ import annotations
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "monai>=1.3.0",
#   "torch>=2.7.0",
#   "numpy>=2.0",
#   "nibabel>=5.3",
#   "pandas>=2.0",
# ]
# ///

"""
BrainIAC foundation-feature retrieval for the Brain MRI Cross-Modal Retrieval Challenge.

This script:
- loads the pretrained BrainIAC backbone checkpoint
- embeds every query and gallery volume with the same encoder
- ranks targets by cosine similarity within each dataset/split pool
- writes per-pool vector artifacts and a combined Kaggle submission CSV
"""

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.networks.nets import ViT
from torch import nn
from torch.utils.data import DataLoader, Dataset


DEFAULT_DATA_ROOT = Path("ehl-paris-medical-image-retrieval")
DEFAULT_CHECKPOINT = Path("checkpoints/BrainIAC.ckpt")
DEFAULT_VECTORS_DIR = Path("artifacts/vectors/brainiac_cosine")
DEFAULT_SUBMISSION = Path("submissions/brainiac_cosine_submission.csv")
IMAGE_SIZE = (96, 96, 96)
SEED = 20260627


@dataclass(frozen=True)
class Pool:
    dataset: str
    split: str
    query_csv: Path
    gallery_csv: Path


class ViTBackboneNet(nn.Module):
    def __init__(self, checkpoint_path: Path) -> None:
        super().__init__()
        self.backbone = ViT(
            in_channels=1,
            img_size=IMAGE_SIZE,
            patch_size=(16, 16, 16),
            hidden_size=768,
            mlp_dim=3072,
            num_layers=12,
            num_heads=12,
            save_attn=True,
        )

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        backbone_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("backbone."):
                backbone_state_dict[key[9:]] = value
        self.backbone.load_state_dict(backbone_state_dict, strict=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        cls_token = features[0][:, 0]
        return F.normalize(cls_token, dim=1)


class ManifestDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], root: Path, id_key: str, image_key: str) -> None:
        self.rows = rows
        self.root = root
        self.id_key = id_key
        self.image_key = image_key

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        image_id = row[self.id_key]
        image_path = resolve_image_path(self.root, row[self.image_key])
        volume = load_and_preprocess_volume(image_path)
        return {"id": image_id, "image": volume}


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
    if path.name.endswith(".nii.gz"):
        fallback = path.with_name(path.name[:-3])
        if fallback.exists():
            return fallback
    raise FileNotFoundError(f"Image path not found: {path}")


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


def normalize_nonzero(volume: np.ndarray) -> np.ndarray:
    volume = np.nan_to_num(volume.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    mask = np.isfinite(volume) & (np.abs(volume) > 1e-6)
    foreground = volume[mask]
    if foreground.size == 0:
        return volume
    mean = float(foreground.mean())
    std = float(foreground.std())
    if std < 1e-6:
        return volume - mean
    out = volume.copy()
    out[mask] = (out[mask] - mean) / std
    out[~mask] = 0.0
    return out


def load_and_preprocess_volume(path: Path) -> torch.Tensor:
    image = nib.load(str(path))
    volume = np.asanyarray(image.dataobj).astype(np.float32)
    if volume.ndim > 3:
        volume = volume[..., 0]
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D image, got {volume.shape} for {path}")
    volume = normalize_nonzero(volume)
    tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=IMAGE_SIZE, mode="trilinear", align_corners=False)
    return tensor.squeeze(0).contiguous()


def embed_manifest(
    model: ViTBackboneNet,
    rows: list[dict[str, str]],
    root: Path,
    id_key: str,
    image_key: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[list[str], np.ndarray]:
    dataset = ManifestDataset(rows, root, id_key, image_key)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type != "cpu")

    ids: list[str] = []
    vectors: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            ids.extend(batch["id"])
            images = batch["image"].to(device)
            embeddings = model(images).cpu().numpy().astype(np.float32)
            vectors.append(embeddings)
    return ids, np.concatenate(vectors, axis=0)


def ranking_rows_from_scores(query_ids: list[str], target_ids: list[str], scores: np.ndarray) -> list[dict[str, str]]:
    rows = []
    target_ids_arr = np.asarray(target_ids)
    for qi, query_id in enumerate(query_ids):
        order = np.argsort(-scores[qi], kind="mergesort")
        rows.append({"query_id": query_id, "target_id_ranking": " ".join(target_ids_arr[order].tolist())})
    return rows


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


def validate_submission(data_root: Path, submission: Path) -> dict[str, object]:
    rows = read_csv(submission)
    by_query = {row["query_id"]: row["target_id_ranking"].split() for row in rows}

    expected: dict[str, tuple[str, str, set[str]]] = {}
    for pool in all_prediction_pools(data_root):
        queries = read_csv(pool.query_csv)
        targets = {row["target_id"] for row in read_csv(pool.gallery_csv)}
        for row in queries:
            expected[row["query_id"]] = (pool.dataset, pool.split, targets)

    counts: dict[str, int] = {}
    errors: list[str] = []
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
            errors.append(f"{query_id}: target set mismatch")
        if len(ranking) != len(ranking_set):
            errors.append(f"{query_id}: duplicate target IDs")

    missing_queries = sorted(set(expected) - set(by_query))
    if missing_queries:
        errors.append(f"missing {len(missing_queries)} expected queries")

    return {
        "submission": str(submission),
        "rows": len(rows),
        "counts": counts,
        "errors": errors[:20],
        "num_errors": len(errors),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--vectors-dir", type=Path, default=DEFAULT_VECTORS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_SUBMISSION)
    parser.add_argument("--datasets", nargs="+", choices=["dataset1", "dataset2", "dataset3"], default=["dataset1", "dataset2", "dataset3"])
    parser.add_argument("--splits", nargs="+", choices=["val", "test"], default=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    args = parse_args()

    data_root = args.data_root.resolve()
    checkpoint = args.checkpoint.resolve()
    vectors_dir = args.vectors_dir.resolve()
    out = args.out.resolve()

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = ViTBackboneNet(checkpoint).to(device)

    submission_rows: list[dict[str, str]] = []
    for pool in all_prediction_pools(data_root, tuple(args.datasets), tuple(args.splits)):
        query_rows = read_csv(pool.query_csv)
        gallery_rows = read_csv(pool.gallery_csv)

        query_ids, query_vectors = embed_manifest(
            model,
            query_rows,
            data_root,
            id_key="query_id",
            image_key="query_image",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        target_ids, target_vectors = embed_manifest(
            model,
            gallery_rows,
            data_root,
            id_key="target_id",
            image_key="target_image",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
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

    write_csv(out, submission_rows, ["query_id", "target_id_ranking"])
    report = validate_submission(data_root, out)
    print(pd.Series(report, dtype=object).to_json(indent=2), flush=True)


if __name__ == "__main__":
    main()
