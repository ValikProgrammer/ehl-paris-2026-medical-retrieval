from __future__ import annotations
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "monai==1.3.2",
#   "torch>=2.7.0",
#   "numpy>=2.0",
#   "nibabel>=5.3",
#   "pandas>=2.0",
# ]
# ///

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai.networks.nets import ViT
from torch import nn
from torch.utils.data import DataLoader, Dataset


DEFAULT_DATA_ROOT = Path("ehl-paris-medical-image-retrieval")
DEFAULT_CHECKPOINT = Path("checkpoints/BrainIAC.ckpt")
DEFAULT_RUNS_ROOT = Path("runs")
IMAGE_SIZE = (96, 96, 96)
SEED = 20260627
PAIR_AUG_SUFFIX_RE = re.compile(r"_aug\d+$")


@dataclass(frozen=True)
class Pool:
    dataset: str
    split: str
    query_csv: Path
    gallery_csv: Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    def json_default(value: object) -> object:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=json_default)
        f.write("\n")


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
            pools.append(Pool(dataset=dataset, split=split, query_csv=ds / f"{split}_queries.csv", gallery_csv=ds / f"{split}_gallery.csv"))
    return pools


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pair_key(row: dict[str, str]) -> str:
    return row.get("pair_id") or f"{row['query_id']}|{row['target_id']}"


def canonical_pair_key(row: dict[str, str]) -> str:
    """Group generated augmentation rows with their source pair."""
    key = pair_key(row)
    key = PAIR_AUG_SUFFIX_RE.sub("", key)
    if "|" in key:
        return "|".join(PAIR_AUG_SUFFIX_RE.sub("", part) for part in key.split("|"))
    return key


def stable_split(
    pairs: list[dict[str, str]],
    holdout_frac: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    scored = []
    for row in pairs:
        digest = hashlib.sha256(f"{seed}:{pair_key(row)}".encode()).hexdigest()
        scored.append((digest, row))
    scored.sort(key=lambda x: x[0])
    holdout_n = max(1, min(len(scored) - 1, round(len(scored) * holdout_frac)))
    holdout = [row for _, row in scored[:holdout_n]]
    train = [row for _, row in scored[holdout_n:]]
    return train, holdout


def split_augmented_training_rows(
    train_rows: list[dict[str, str]],
    holdout_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Keep augmented copies of holdout source pairs out of adapter training."""
    holdout_keys = {canonical_pair_key(row) for row in holdout_rows}
    return [row for row in train_rows if canonical_pair_key(row) not in holdout_keys]


def unique_manifest_rows(rows: list[dict[str, str]], id_key: str) -> list[dict[str, str]]:
    seen: set[str] = set()
    unique_rows: list[dict[str, str]] = []
    for row in rows:
        image_id = row[id_key]
        if image_id in seen:
            continue
        seen.add(image_id)
        unique_rows.append(row)
    return unique_rows


def normalize_nonzero(volume: np.ndarray) -> np.ndarray:
    volume = np.nan_to_num(volume.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    mask = np.isfinite(volume) & (np.abs(volume) > 1e-6)
    foreground = volume[mask]
    if foreground.size == 0:
        return volume
    mean = float(foreground.mean())
    std = float(foreground.std())
    out = volume.copy()
    if std < 1e-6:
        out[mask] = out[mask] - mean
    else:
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
        backbone_state_dict = {key[9:]: value for key, value in state_dict.items() if key.startswith("backbone.")}
        self.backbone.load_state_dict(backbone_state_dict, strict=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return F.normalize(features[0][:, 0], dim=1)


class ManifestImageDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], data_root: Path, id_key: str, image_key: str) -> None:
        self.rows = rows
        self.data_root = data_root
        self.id_key = id_key
        self.image_key = image_key

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        image_id = row[self.id_key]
        image = load_and_preprocess_volume(resolve_image_path(self.data_root, row[self.image_key]))
        return {"id": image_id, "image": image}


def embed_manifest(
    model: ViTBackboneNet,
    rows: list[dict[str, str]],
    data_root: Path,
    id_key: str,
    image_key: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[list[str], np.ndarray]:
    dataset = ManifestImageDataset(rows, data_root, id_key, image_key)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type != "cpu")
    ids: list[str] = []
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            ids.extend(batch["id"])
            embeddings = model(batch["image"].to(device)).detach().cpu().numpy().astype(np.float32)
            outputs.append(embeddings)
    return ids, np.concatenate(outputs, axis=0)


class PairEmbeddingDataset(Dataset):
    def __init__(self, query_vectors: np.ndarray, target_vectors: np.ndarray) -> None:
        self.query_vectors = torch.from_numpy(query_vectors.astype(np.float32))
        self.target_vectors = torch.from_numpy(target_vectors.astype(np.float32))

    def __len__(self) -> int:
        return len(self.query_vectors)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.query_vectors[index], self.target_vectors[index]


class AdapterModel(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dim: int = 768, output_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.query_adapter = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.target_adapter = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0), dtype=torch.float32))

    def encode_query(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query_adapter(x), dim=1)

    def encode_target(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.target_adapter(x), dim=1)

    def scale(self) -> torch.Tensor:
        return self.logit_scale.clamp(math.log(1.0), math.log(100.0)).exp()

    def forward(self, query: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.scale() * self.encode_query(query) @ self.encode_target(target).T


def metrics_from_scores(
    query_ids: list[str],
    target_ids: list[str],
    truth: dict[str, str],
    score_matrix: np.ndarray,
) -> dict[str, float]:
    target_ids_sorted = list(target_ids)
    target_index = {target_id: index for index, target_id in enumerate(target_ids_sorted)}
    ranks = []
    for query_index, query_id in enumerate(query_ids):
        true_index = target_index[truth[query_id]]
        true_score = score_matrix[query_index, true_index]
        rank = 1 + int(np.sum(score_matrix[query_index] > true_score))
        ranks.append(rank)
    ranks_np = np.asarray(ranks, dtype=np.float32)
    return {
        "mrr": float(np.mean(1.0 / ranks_np)),
        "recall_at_1": float(np.mean(ranks_np <= 1)),
        "recall_at_5": float(np.mean(ranks_np <= 5)),
        "median_rank": float(np.median(ranks_np)),
        "mean_rank": float(np.mean(ranks_np)),
        "pairs": float(len(query_ids)),
    }


def ranking_rows_from_scores(query_ids: list[str], target_ids: list[str], score_matrix: np.ndarray) -> list[dict[str, str]]:
    rows = []
    target_ids_arr = np.asarray(target_ids)
    for query_index, query_id in enumerate(query_ids):
        order = np.argsort(-score_matrix[query_index], kind="mergesort")
        rows.append({"query_id": query_id, "target_id_ranking": " ".join(target_ids_arr[order].tolist())})
    return rows


@torch.no_grad()
def adapted_vectors(model: AdapterModel, query_vectors: np.ndarray, target_vectors: np.ndarray, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    query_outputs: list[np.ndarray] = []
    target_outputs: list[np.ndarray] = []
    for start in range(0, len(query_vectors), batch_size):
        batch = torch.from_numpy(query_vectors[start : start + batch_size]).to(device)
        query_outputs.append(model.encode_query(batch).cpu().numpy().astype(np.float32))
    for start in range(0, len(target_vectors), batch_size):
        batch = torch.from_numpy(target_vectors[start : start + batch_size]).to(device)
        target_outputs.append(model.encode_target(batch).cpu().numpy().astype(np.float32))
    return np.concatenate(query_outputs, axis=0), np.concatenate(target_outputs, axis=0)


def evaluate_pair_metrics(
    model: AdapterModel | None,
    query_ids: list[str],
    query_vectors: np.ndarray,
    target_ids: list[str],
    target_vectors: np.ndarray,
    truth: dict[str, str],
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    if model is None:
        score_matrix = query_vectors @ target_vectors.T
    else:
        adapted_query, adapted_target = adapted_vectors(model, query_vectors, target_vectors, device, batch_size)
        score_matrix = adapted_query @ adapted_target.T
    return metrics_from_scores(query_ids, target_ids, truth, score_matrix.astype(np.float32))


def save_vectors(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def train_adapter(
    model: AdapterModel,
    train_query_vectors: np.ndarray,
    train_target_vectors: np.ndarray,
    holdout_query_ids: list[str],
    holdout_query_vectors: np.ndarray,
    holdout_target_ids: list[str],
    holdout_target_vectors: np.ndarray,
    holdout_truth: dict[str, str],
    all_target_ids: list[str],
    all_target_vectors: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[AdapterModel, list[dict[str, float]], dict[str, float], dict[str, float]]:
    dataset = PairEmbeddingDataset(train_query_vectors, train_target_vectors)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_state = None
    best_metrics = None
    best_all_gallery_metrics = None
    best_key = -1.0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for query_batch, target_batch in loader:
            query_batch = query_batch.to(device)
            target_batch = target_batch.to(device)
            labels = torch.arange(len(query_batch), device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(query_batch, target_batch)
            loss = (loss_fn(logits, labels) + loss_fn(logits.T, labels)) / 2.0
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(query_batch)
            total_seen += len(query_batch)

        holdout_metrics = evaluate_pair_metrics(
            model,
            holdout_query_ids,
            holdout_query_vectors,
            holdout_target_ids,
            holdout_target_vectors,
            holdout_truth,
            device,
            batch_size,
        )
        all_gallery_metrics = evaluate_pair_metrics(
            model,
            holdout_query_ids,
            holdout_query_vectors,
            all_target_ids,
            all_target_vectors,
            holdout_truth,
            device,
            batch_size,
        )
        row = {
            "epoch": float(epoch),
            "loss": total_loss / max(total_seen, 1),
            "temperature": float(model.scale().detach().cpu()),
            "holdout_mrr": holdout_metrics["mrr"],
            "holdout_r1": holdout_metrics["recall_at_1"],
            "all_gallery_mrr": all_gallery_metrics["mrr"],
            "all_gallery_r1": all_gallery_metrics["recall_at_1"],
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} loss={row['loss']:.5f} "
            f"holdout_mrr={row['holdout_mrr']:.4f} all_gallery_mrr={row['all_gallery_mrr']:.4f}",
            flush=True,
        )

        key = 1000.0 * all_gallery_metrics["mrr"] + holdout_metrics["mrr"]
        if key > best_key:
            best_key = key
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = holdout_metrics
            best_all_gallery_metrics = all_gallery_metrics

    if best_state is None or best_metrics is None or best_all_gallery_metrics is None:
        raise RuntimeError("Training finished without a best checkpoint")

    model.load_state_dict(best_state)
    return model, history, best_metrics, best_all_gallery_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--train-pair-csv", type=Path, default=None, help="Training pair CSV. Can be an augmented CSV.")
    parser.add_argument("--holdout-pair-csv", type=Path, default=None, help="Original labelled pairs used for leakage-free holdout.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUNS_ROOT / "brainiac_adapter")
    parser.add_argument("--holdout-frac", type=float, default=0.20)
    parser.add_argument("--split-seed", type=int, default=SEED)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embed-batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--output-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    data_root = args.data_root.resolve()
    checkpoint = args.checkpoint.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)

    train_pair_csv = args.train_pair_csv or data_root / "dataset1" / "train_pairs.csv"
    holdout_pair_csv = args.holdout_pair_csv or data_root / "dataset1" / "train_pairs.csv"
    train_pairs = read_csv(train_pair_csv)
    holdout_source_pairs = read_csv(holdout_pair_csv)
    source_train_split, holdout_split = stable_split(holdout_source_pairs, args.holdout_frac, args.split_seed)
    train_split = split_augmented_training_rows(train_pairs, holdout_split)
    if not train_split:
        raise ValueError("No training rows remain after excluding holdout source pairs")
    split_payload = {
        "seed": args.seed,
        "split_seed": args.split_seed,
        "holdout_frac": args.holdout_frac,
        "train_pair_csv": train_pair_csv,
        "holdout_pair_csv": holdout_pair_csv,
        "source_train_pairs": len(source_train_split),
        "source_holdout_pairs": len(holdout_split),
        "train_pairs": len(train_split),
        "holdout_pairs": len(holdout_split),
        "train_pair_ids": [pair_key(row) for row in train_split],
        "holdout_pair_ids": [pair_key(row) for row in holdout_split],
    }
    write_json(run_dir / "split.json", split_payload)

    backbone = ViTBackboneNet(checkpoint).to(device)

    rows_to_embed = train_split + holdout_split + holdout_source_pairs
    all_train_queries = unique_manifest_rows(
        [{"query_id": row["query_id"], "query_image": row["query_image"]} for row in rows_to_embed],
        "query_id",
    )
    all_train_targets = unique_manifest_rows(
        [{"target_id": row["target_id"], "target_image": row["target_image"]} for row in rows_to_embed],
        "target_id",
    )
    train_query_ids, train_query_vectors = embed_manifest(backbone, all_train_queries, data_root, "query_id", "query_image", args.embed_batch_size, args.num_workers, device)
    train_target_ids, train_target_vectors = embed_manifest(backbone, all_train_targets, data_root, "target_id", "target_image", args.embed_batch_size, args.num_workers, device)
    query_map = {image_id: vector for image_id, vector in zip(train_query_ids, train_query_vectors)}
    target_map = {image_id: vector for image_id, vector in zip(train_target_ids, train_target_vectors)}

    train_query_matrix = np.stack([query_map[row["query_id"]] for row in train_split]).astype(np.float32)
    train_target_matrix = np.stack([target_map[row["target_id"]] for row in train_split]).astype(np.float32)
    holdout_query_ids = [row["query_id"] for row in holdout_split]
    holdout_target_ids = [row["target_id"] for row in holdout_split]
    holdout_query_matrix = np.stack([query_map[query_id] for query_id in holdout_query_ids]).astype(np.float32)
    holdout_target_matrix = np.stack([target_map[target_id] for target_id in holdout_target_ids]).astype(np.float32)
    holdout_truth = {row["query_id"]: row["target_id"] for row in holdout_split}
    original_target_ids = [row["target_id"] for row in holdout_source_pairs]
    all_gallery_target_ids = [target_id for target_id in original_target_ids if target_id in target_map]
    all_gallery_target_matrix = np.stack([target_map[target_id] for target_id in all_gallery_target_ids]).astype(np.float32)

    raw_holdout_metrics = evaluate_pair_metrics(
        None,
        holdout_query_ids,
        holdout_query_matrix,
        holdout_target_ids,
        holdout_target_matrix,
        holdout_truth,
        device,
        args.batch_size,
    )
    raw_all_gallery_metrics = evaluate_pair_metrics(
        None,
        holdout_query_ids,
        holdout_query_matrix,
        all_gallery_target_ids,
        all_gallery_target_matrix,
        holdout_truth,
        device,
        args.batch_size,
    )

    adapter = AdapterModel(input_dim=train_query_matrix.shape[1], hidden_dim=args.hidden_dim, output_dim=args.output_dim, dropout=args.dropout)
    adapter, history, best_holdout_metrics, best_all_gallery_metrics = train_adapter(
        adapter,
        train_query_matrix,
        train_target_matrix,
        holdout_query_ids,
        holdout_query_matrix,
        holdout_target_ids,
        holdout_target_matrix,
        holdout_truth,
        all_gallery_target_ids,
        all_gallery_target_matrix,
        device,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.weight_decay,
    )

    torch.save(
        {
            "adapter_state_dict": adapter.state_dict(),
            "args": vars(args),
            "raw_holdout_metrics": raw_holdout_metrics,
            "raw_all_gallery_metrics": raw_all_gallery_metrics,
            "best_holdout_metrics": best_holdout_metrics,
            "best_all_gallery_metrics": best_all_gallery_metrics,
        },
        run_dir / "adapter.pt",
    )
    write_json(
        run_dir / "metrics.json",
        {
            "raw_holdout_metrics": raw_holdout_metrics,
            "raw_all_gallery_metrics": raw_all_gallery_metrics,
            "best_holdout_metrics": best_holdout_metrics,
            "best_all_gallery_metrics": best_all_gallery_metrics,
            "config": vars(args),
        },
    )
    write_csv(run_dir / "history.csv", history, list(history[0].keys()))

    submission_rows: list[dict[str, str]] = []
    for pool in all_prediction_pools(data_root):
        query_rows = read_csv(pool.query_csv)
        target_rows = read_csv(pool.gallery_csv)
        query_ids, query_vectors = embed_manifest(backbone, query_rows, data_root, "query_id", "query_image", args.embed_batch_size, args.num_workers, device)
        target_ids, target_vectors = embed_manifest(backbone, target_rows, data_root, "target_id", "target_image", args.embed_batch_size, args.num_workers, device)
        adapted_query, adapted_target = adapted_vectors(adapter, query_vectors, target_vectors, device, args.batch_size)
        score_matrix = adapted_query @ adapted_target.T
        submission_rows.extend(ranking_rows_from_scores(query_ids, target_ids, score_matrix.astype(np.float32)))
        save_vectors(
            run_dir / "vectors" / f"{pool.dataset}_{pool.split}_vectors.npz",
            {
                "dataset": np.asarray(pool.dataset),
                "split": np.asarray(pool.split),
                "query_ids": np.asarray(query_ids),
                "target_ids": np.asarray(target_ids),
                "query_vectors": adapted_query.astype(np.float32),
                "target_vectors": adapted_target.astype(np.float32),
                "raw_query_vectors": query_vectors.astype(np.float32),
                "raw_target_vectors": target_vectors.astype(np.float32),
            },
        )
        print(f"predicted {pool.dataset}/{pool.split}: {len(query_ids)}x{len(target_ids)}", flush=True)

    write_csv(run_dir / "submission.csv", submission_rows, ["query_id", "target_id_ranking"])
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "raw_holdout_metrics": raw_holdout_metrics,
                "raw_all_gallery_metrics": raw_all_gallery_metrics,
                "best_holdout_metrics": best_holdout_metrics,
                "best_all_gallery_metrics": best_all_gallery_metrics,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
