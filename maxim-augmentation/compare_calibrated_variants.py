from __future__ import annotations

import argparse
import csv
import itertools
import re
import subprocess
import sys
from pathlib import Path


SCORE_RE = re.compile(r"raw_mrr=(?P<raw>[0-9.]+)\s+hungarian_acc=(?P<hung>[0-9.]+)")


def run_variant(args: argparse.Namespace, opt_size: int, angle: float, shift: float, maxiter: int, jitter: float) -> dict[str, str]:
    cmd = [
        sys.executable,
        "evaluate_calibrated_synthetic.py",
        "--data-root", str(args.data_root),
        "--transform-estimates", str(args.transform_estimates),
        "--n-train", str(args.n_train),
        "--n-eval", str(args.n_eval),
        "--grid", str(args.grid),
        "--components", str(args.components),
        "--alpha", str(args.alpha),
        "--opt-size", str(opt_size),
        "--start-angle-deg", str(angle),
        "--start-shift-vox", str(shift),
        "--reg-maxiter", str(maxiter),
        "--min-transform-margin", str(args.min_transform_margin),
        "--jitter", str(jitter),
        "--elastic-probability", str(args.elastic_probability),
        "--elastic-sigma", str(args.elastic_sigma[0]), str(args.elastic_sigma[1]),
        "--elastic-magnitude", str(args.elastic_magnitude[0]), str(args.elastic_magnitude[1]),
    ]
    print("RUN", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, text=True, capture_output=True)
    output = (proc.stdout + "\n" + proc.stderr).strip()
    match = SCORE_RE.search(output)
    row = {
        "opt_size": str(opt_size),
        "start_angle_deg": str(angle),
        "start_shift_vox": str(shift),
        "reg_maxiter": str(maxiter),
        "jitter": str(jitter),
        "returncode": str(proc.returncode),
        "raw_mrr": "",
        "hungarian_acc": "",
        "output_tail": output[-1000:].replace("\n", " | "),
    }
    if match:
        row["raw_mrr"] = match.group("raw")
        row["hungarian_acc"] = match.group("hung")
    print(
        f"DONE opt={opt_size} angle={angle} shift={shift} maxiter={maxiter} jitter={jitter} "
        f"raw_mrr={row['raw_mrr'] or 'NA'} hungarian_acc={row['hungarian_acc'] or 'NA'}",
        flush=True,
    )
    return row


def parse_csv_numbers(value: str, typ):
    return [typ(x) for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/root/.cache/kagglehub/competitions/ehl-paris-medical-image-retrieval"))
    parser.add_argument("--transform-estimates", type=Path, default=Path("output/transform_estimates.csv"))
    parser.add_argument("--out", type=Path, default=Path("output/calibrated_variant_scores.csv"))
    parser.add_argument("--grid", type=int, default=44)
    parser.add_argument("--n-train", type=int, default=250)
    parser.add_argument("--n-eval", type=int, default=60)
    parser.add_argument("--components", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--opt-sizes", default="20,24")
    parser.add_argument("--angles", default="14,18")
    parser.add_argument("--shifts", default="0,2")
    parser.add_argument("--maxiters", default="100,120")
    parser.add_argument("--jitters", default="1.0")
    parser.add_argument("--min-transform-margin", type=float, default=0.01)
    parser.add_argument("--elastic-probability", type=float, default=0.45)
    parser.add_argument("--elastic-sigma", type=float, nargs=2, default=(5.0, 8.0))
    parser.add_argument("--elastic-magnitude", type=float, nargs=2, default=(3.0, 8.0))
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = list(
        itertools.product(
            parse_csv_numbers(args.opt_sizes, int),
            parse_csv_numbers(args.angles, float),
            parse_csv_numbers(args.shifts, float),
            parse_csv_numbers(args.maxiters, int),
            parse_csv_numbers(args.jitters, float),
        )
    )
    if args.limit is not None:
        variants = variants[: args.limit]
    rows = []
    for opt_size, angle, shift, maxiter, jitter in variants:
        rows.append(run_variant(args, opt_size, angle, shift, maxiter, jitter))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    ranked = sorted(
        [row for row in rows if row["raw_mrr"]],
        key=lambda row: (float(row["raw_mrr"]), float(row["hungarian_acc"])),
        reverse=True,
    )
    print("\nRANKED")
    for row in ranked:
        print(
            f"raw_mrr={float(row['raw_mrr']):.6f} hungarian_acc={float(row['hungarian_acc']):.6f} "
            f"opt={row['opt_size']} angle={row['start_angle_deg']} shift={row['start_shift_vox']} "
            f"maxiter={row['reg_maxiter']} jitter={row['jitter']}"
        )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
