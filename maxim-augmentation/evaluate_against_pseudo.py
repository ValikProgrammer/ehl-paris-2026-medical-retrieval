from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def build_truth(pseudo_matches_csv: Path, min_margin: float, min_score: float) -> dict[str, str]:
    truth: dict[str, str] = {}
    for row in read_csv(pseudo_matches_csv):
        if float(row["margin"]) < min_margin:
            continue
        if float(row["score"]) < min_score:
            continue
        truth[row["query_id"]] = row["target_id"]
    return truth


def load_rankings(submission_csv: Path) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = {}
    for row in read_csv(submission_csv):
        rankings[row["query_id"]] = row["target_id_ranking"].split()
    return rankings


def evaluate(truth: dict[str, str], rankings: dict[str, list[str]]) -> dict[str, float]:
    ranks = []
    missing = 0
    for query_id, target_id in truth.items():
        ranking = rankings.get(query_id)
        if not ranking:
            missing += 1
            ranks.append(np.inf)
            continue
        try:
            ranks.append(ranking.index(target_id) + 1)
        except ValueError:
            missing += 1
            ranks.append(np.inf)
    if not ranks:
        raise ValueError("No pseudo labels matched this filter.")
    ranks_np = np.asarray(ranks, dtype=np.float64)
    finite = np.isfinite(ranks_np)
    reciprocal = np.where(finite, 1.0 / ranks_np, 0.0)
    return {
        "n": float(len(ranks)),
        "missing": float(missing),
        "mrr": float(reciprocal.mean()),
        "top1": float(np.mean(ranks_np == 1)),
        "top3": float(np.mean(ranks_np <= 3)),
        "top5": float(np.mean(ranks_np <= 5)),
        "median_rank": float(np.median(ranks_np[finite])) if finite.any() else float("inf"),
        "mean_rank": float(np.mean(ranks_np[finite])) if finite.any() else float("inf"),
    }


def compare_to_baseline(
    truth: dict[str, str],
    candidate: dict[str, list[str]],
    baseline: dict[str, list[str]] | None,
) -> dict[str, float]:
    if baseline is None:
        return {}
    better = worse = same = 0
    candidate_top1_diff = 0
    for query_id, target_id in truth.items():
        cand_rank = candidate.get(query_id, [])
        base_rank = baseline.get(query_id, [])
        try:
            c = cand_rank.index(target_id) + 1
        except ValueError:
            c = 10**9
        try:
            b = base_rank.index(target_id) + 1
        except ValueError:
            b = 10**9
        if c < b:
            better += 1
        elif c > b:
            worse += 1
        else:
            same += 1
        if cand_rank and base_rank and cand_rank[0] != base_rank[0]:
            candidate_top1_diff += 1
    n = max(1, len(truth))
    return {
        "better_than_baseline": float(better),
        "worse_than_baseline": float(worse),
        "same_as_baseline": float(same),
        "top1_changed_vs_baseline": float(candidate_top1_diff),
        "better_frac": better / n,
        "worse_frac": worse / n,
        "top1_changed_frac": candidate_top1_diff / n,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pseudo-matches", type=Path, default=Path("output/pseudo_matches.csv"))
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--min-margin", type=float, default=0.02)
    parser.add_argument("--min-score", type=float, default=-1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    truth = build_truth(args.pseudo_matches, args.min_margin, args.min_score)
    candidate = load_rankings(args.submission)
    baseline = load_rankings(args.baseline) if args.baseline else None
    metrics = evaluate(truth, candidate)
    metrics.update(compare_to_baseline(truth, candidate, baseline))
    print(f"pseudo_labels={int(metrics['n'])} min_margin={args.min_margin:g} min_score={args.min_score:g}")
    for key in sorted(k for k in metrics if k != "n"):
        value = metrics[key]
        if float(value).is_integer():
            print(f"{key}: {int(value)}")
        else:
            print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
