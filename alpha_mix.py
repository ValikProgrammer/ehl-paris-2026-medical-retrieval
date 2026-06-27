from __future__ import annotations
# /// script
# requires-python = ">=3.12"
# dependencies = ["nibabel>=5.3","numpy>=2.0","scipy>=1.14","scikit-learn>=1.5"]
# ///

"""Generate a full-mix submission for a given Ridge alpha (d1+d2 template), with
d3 held fixed to the grid-feature file. Uses cached g44 features so each alpha is
a fast Ridge refit + rescore. For the ternary alpha search."""

import argparse, csv
from pathlib import Path

from d2_template_retrieval import build_model, score_pool, write_csv, DEFAULT_DATA_ROOT, DEFAULT_CACHE


def loadmap(p):
    with open(p, newline="") as f:
        return {r["query_id"]: r["target_id_ranking"] for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--grid", type=int, default=44)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--d3-file", type=Path, default=Path("submissions/d3_gridfeat_g44_hungarian.csv"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    model = build_model(args.data_root, args.grid, 128, args.alpha)  # templates clean, map clean@alpha
    rows = []
    for ds in ("dataset1", "dataset2"):
        for split in ("val", "test"):
            rows.extend(score_pool(args.data_root, ds, split, args.grid, model, args.cache_dir,
                                   assignment=True, register=True))
    d3 = loadmap(args.d3_file)
    for qid, rk in d3.items():
        rows.append({"query_id": qid, "target_id_ranking": rk})
    write_csv(args.out, rows)
    print(f"alpha={args.alpha} wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
