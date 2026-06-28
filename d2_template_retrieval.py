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

"""Dataset2 retrieval via template normalization.

Validated on the synthetic-d2 harness: template normalization scores ~0.55 MRR
vs ~0.26 for the canonical PCA-axis baseline at realistic d2 distortion.

Pipeline:
  1. Build T1 / T2 templates = mean of (downsampled) dataset1 train queries/targets.
     dataset1 train pairs are registered, so the means are sharp and the two
     templates share one grid.
  2. Fit PCA/Ridge cross-modal map on the clean dataset1 train pairs.
  3. For each dataset2 query/target: downsample, rigidly register to its
     same-modality template (intramodal intensity NCC -> robust), flatten.
  4. Score = cosine(ridge(query_pca), target_pca); optional Hungarian assignment.

Normalized features are cached in --cache-dir keyed by (image_id, grid).
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import normalize

from synthetic_d2_eval import _load_grid, read_pairs, resolve_image_path
from d2_methods import _register_to_template, flat_feature, rich_feature, sliceview_feature, _fit_pca_ridge  # noqa: F401

FEAT_FN = flat_feature  # swapped to rich_feature when --rich is passed


DEFAULT_DATA_ROOT = Path("ehl-paris-medical-image-retrieval")
DEFAULT_CACHE = Path(".d2cache")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query_id", "target_id_ranking"])
        w.writeheader()
        w.writerows(rows)


def normalized_feature(
    data_root: Path, image_path: str, image_id: str, grid: int, template: np.ndarray,
    cache_dir: Path, register: bool = True
) -> np.ndarray:
    fn = "_rich" if FEAT_FN is rich_feature else ("_slice" if FEAT_FN is sliceview_feature else "")
    tag = f"g{grid}" + fn + ("" if register else "_noreg")
    cache = cache_dir / f"{image_id}_{tag}.npy"
    if cache.exists():
        return np.load(cache)
    vol = _load_grid(data_root, image_path, grid)
    feat = FEAT_FN(_register_to_template(vol, template) if register else vol)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache, feat)
    return feat


def build_model(data_root: Path, grid: int, components: int, alpha: float,
                fit_pair_csv: Path | None = None, synth_aug_k: int = 0):
    # templates ALWAYS come from the clean registered dataset1 train pairs
    pairs = read_pairs(data_root)
    tq = np.stack([_load_grid(data_root, p["query_image"], grid) for p in pairs])
    tt = np.stack([_load_grid(data_root, p["target_image"], grid) for p in pairs])
    t1_tpl = tq.mean(axis=0)
    t2_tpl = tt.mean(axis=0)

    if synth_aug_k > 0:
        # fit the map on clean pairs + K geom+contrast augmented copies (registered)
        from synthetic_d2_eval import deform, _rng_for
        from d2_methods import _contrast_jitter
        dkw = dict(max_rot_deg=12, max_shift=6, elastic_sigma=8, elastic_alpha=3)
        qf = [FEAT_FN(v) for v in tq]
        tf = [FEAT_FN(v) for v in tt]
        for a in range(synth_aug_k):
            for i, v in enumerate(tq):
                w = _contrast_jitter(deform(v, _rng_for(1000 + a, f"q{i}"), **dkw), _rng_for(2000 + a, f"q{i}"))
                qf.append(FEAT_FN(_register_to_template(w, t1_tpl)))
            for i, v in enumerate(tt):
                w = _contrast_jitter(deform(v, _rng_for(3000 + a, f"t{i}"), **dkw), _rng_for(4000 + a, f"t{i}"))
                tf.append(FEAT_FN(_register_to_template(w, t2_tpl)))
            print(f"  synth-aug copy {a+1}/{synth_aug_k} built", flush=True)
        qf = np.stack(qf); tf = np.stack(tf)
        n_fit = len(qf)
    elif fit_pair_csv is None:
        # fit the map on the same clean pairs (already aligned -> no registration)
        qf = np.stack([FEAT_FN(v) for v in tq])
        tf = np.stack([FEAT_FN(v) for v in tt])
        n_fit = len(pairs)
    else:
        # fit the map on a (possibly augmented/deformed) manifest; register each
        # to its template first so it lands in the common frame
        fit_pairs = read_csv(Path(fit_pair_csv))
        qf, tf = [], []
        for i, p in enumerate(fit_pairs):
            qv = _load_grid(data_root, p["query_image"], grid)
            tv = _load_grid(data_root, p["target_image"], grid)
            qf.append(FEAT_FN(_register_to_template(qv, t1_tpl)))
            tf.append(FEAT_FN(_register_to_template(tv, t2_tpl)))
            if (i + 1) % 100 == 0:
                print(f"  fit-feature {i+1}/{len(fit_pairs)}", flush=True)
        qf = np.stack(qf); tf = np.stack(tf)
        n_fit = len(fit_pairs)
    q_pca, t_pca, ridge = _fit_pca_ridge(qf, tf, components, alpha)
    print(f"built templates from {len(pairs)} clean pairs; map fit on {n_fit} pairs", flush=True)
    return t1_tpl, t2_tpl, q_pca, t_pca, ridge


def sinkhorn(scores: np.ndarray, tau: float = 0.05, iters: int = 50) -> np.ndarray:
    """Doubly-stochastic (soft-bijection) normalization of a cosine score matrix.
    Parameter-free at inference (cannot overfit train) — a legitimate re-ranking,
    NOT a data leak. Sharpens the assignment by enforcing the bijection prior."""
    K = np.exp((scores - scores.max()) / tau)
    for _ in range(iters):
        K = K / (K.sum(axis=1, keepdims=True) + 1e-12)
        K = K / (K.sum(axis=0, keepdims=True) + 1e-12)
    return K


def score_pool(
    data_root, dataset, split, grid, model, cache_dir, assignment, register=True, rerank="none"
) -> list[dict[str, str]]:
    t1_tpl, t2_tpl, q_pca, t_pca, ridge = model
    root = data_root / dataset
    qrows = read_csv(root / f"{split}_queries.csv")
    trows = read_csv(root / f"{split}_gallery.csv")

    q = []
    for i, r in enumerate(qrows):
        feat = normalized_feature(data_root, r["query_image"], r["query_id"], grid, t1_tpl, cache_dir, register)
        q.append(normalize(ridge.predict(q_pca.transform(feat[None, :])))[0])
        if (i + 1) % 20 == 0:
            print(f"  {dataset}/{split} query {i+1}/{len(qrows)}", flush=True)
    t = []
    for i, r in enumerate(trows):
        feat = normalized_feature(data_root, r["target_image"], r["target_id"], grid, t2_tpl, cache_dir, register)
        t.append(normalize(t_pca.transform(feat[None, :]))[0])
        if (i + 1) % 20 == 0:
            print(f"  {dataset}/{split} target {i+1}/{len(trows)}", flush=True)
    q = np.stack(q)
    t = np.stack(t)
    scores = (q @ t.T).astype(np.float64)

    if rerank == "sinkhorn":
        scores = sinkhorn(scores)

    if assignment and scores.shape[0] == scores.shape[1]:
        row_ind, col_ind = linear_sum_assignment(-scores)
        assigned = np.empty(scores.shape[0], dtype=np.int64)
        assigned[row_ind] = col_ind
        scores[np.arange(scores.shape[0]), assigned] = scores.max() + 1e6

    target_ids = np.asarray([r["target_id"] for r in trows])
    rows = []
    for qi, r in enumerate(qrows):
        order = np.argsort(-scores[qi], kind="mergesort")
        rows.append({"query_id": r["query_id"], "target_id_ranking": " ".join(target_ids[order].tolist())})
    print(f"scored {dataset}/{split}: {len(qrows)}x{len(trows)}", flush=True)
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--datasets", nargs="+", default=["dataset2"])
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.add_argument("--grid", type=int, default=44)
    p.add_argument("--components", type=int, default=128)
    p.add_argument("--alpha", type=float, default=100.0)
    p.add_argument("--fit-pair-csv", type=Path, default=None,
                   help="Fit the PCA/Ridge map on this (e.g. augmented) manifest; "
                        "templates still come from clean dataset1 train pairs.")
    p.add_argument("--synth-aug-k", type=int, default=0,
                   help="Fit the map on clean + K synthetic geom+contrast augmented "
                        "copies of dataset1 train (local, no extra data needed).")
    p.add_argument("--assignment", action="store_true")
    p.add_argument("--rerank", choices=["none", "sinkhorn"], default="none")
    p.add_argument("--rich", action="store_true", help="Use multi-channel rich feature (intensity+edge+half-scale).")
    p.add_argument("--sliceview", action="store_true", help="Use multi-direction slice feature.")
    p.add_argument("--no-register", action="store_true",
                   help="Skip template registration (use raw downsampled grid feature). "
                        "For already-aligned sets like dataset3.")
    p.add_argument("--out", type=Path, default=Path("submissions/d2_template_g44_hungarian.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    global FEAT_FN
    if args.rich:
        FEAT_FN = rich_feature
    if args.sliceview:
        FEAT_FN = sliceview_feature
    model = build_model(args.data_root, args.grid, args.components, args.alpha,
                        args.fit_pair_csv, args.synth_aug_k)
    rows: list[dict[str, str]] = []
    for dataset in args.datasets:
        for split in args.splits:
            rows.extend(
                score_pool(args.data_root, dataset, split, args.grid, model, args.cache_dir,
                           args.assignment, register=not args.no_register, rerank=args.rerank)
            )
    write_csv(args.out, rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
