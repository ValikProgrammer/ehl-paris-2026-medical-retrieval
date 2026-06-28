#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data}"
DEVICE="${DEVICE:-auto}"
EPOCHS="${EPOCHS:-500}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PRESETS="${PRESETS:-geom geom_contrast}"

mkdir -p submissions
mkdir -p runs/slice_clip_aug

run_experiment() {
  local preset="$1"
  local out="submissions/slice_clip_${preset}.csv"
  local run_dir="runs/slice_clip_aug/${preset}_e${EPOCHS}"

  mkdir -p "$run_dir"

  echo "Running SliceCLIP preset=${preset}"
  "$PYTHON_BIN" slice_clip_baseline.py \
    --data-root "$DATA_ROOT" \
    --train-pair-csv "$DATA_ROOT/dataset1/train_pairs.csv" \
    --query-csv "$DATA_ROOT/dataset1/val_queries.csv" \
    --gallery-csv "$DATA_ROOT/dataset1/val_gallery.csv" \
    --query-csv "$DATA_ROOT/dataset1/test_queries.csv" \
    --gallery-csv "$DATA_ROOT/dataset1/test_gallery.csv" \
    --query-csv "$DATA_ROOT/dataset2/val_queries.csv" \
    --gallery-csv "$DATA_ROOT/dataset2/val_gallery.csv" \
    --query-csv "$DATA_ROOT/dataset2/test_queries.csv" \
    --gallery-csv "$DATA_ROOT/dataset2/test_gallery.csv" \
    --query-csv "$DATA_ROOT/dataset3/val_queries.csv" \
    --gallery-csv "$DATA_ROOT/dataset3/val_gallery.csv" \
    --query-csv "$DATA_ROOT/dataset3/test_queries.csv" \
    --gallery-csv "$DATA_ROOT/dataset3/test_gallery.csv" \
    --augmentation-preset "$preset" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --num-workers "$NUM_WORKERS" \
    --device "$DEVICE" \
    --loss-csv "$run_dir/history.csv" \
    --loss-plot "$run_dir/loss.png" \
    --out "$out"
}

for preset in $PRESETS; do
  run_experiment "$preset"
done
