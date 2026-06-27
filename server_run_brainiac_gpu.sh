#!/usr/bin/env bash
set -euo pipefail

HOST_DATA_ROOT="${HOST_DATA_ROOT:-/root/.cache/kagglehub/competitions/ehl-paris-medical-image-retrieval}"
HOST_CHECKPOINT="${HOST_CHECKPOINT:-/root/BrainIAC/checkpoints/BrainIAC.ckpt}"
HOST_CODE_ROOT="${HOST_CODE_ROOT:-/root/paris}"
CONTAINER_NAME="${CONTAINER_NAME:-rocm}"
CONTAINER_APP_ROOT="${CONTAINER_APP_ROOT:-/app}"
CONTAINER_DATA_ROOT="${CONTAINER_DATA_ROOT:-$CONTAINER_APP_ROOT/ehl-paris-medical-image-retrieval}"
CONTAINER_CHECKPOINT="${CONTAINER_CHECKPOINT:-$CONTAINER_APP_ROOT/BrainIAC.ckpt}"
CONTAINER_VECTORS_DIR="${CONTAINER_VECTORS_DIR:-$CONTAINER_APP_ROOT/brainiac_artifacts/vectors/brainiac_cosine}"
CONTAINER_SUBMISSION="${CONTAINER_SUBMISSION:-$CONTAINER_APP_ROOT/brainiac_submissions/brainiac_cosine_submission.csv}"

echo "[1/6] Preparing container directories"
docker exec "$CONTAINER_NAME" bash -lc "mkdir -p '$CONTAINER_APP_ROOT' '$CONTAINER_APP_ROOT/brainiac_artifacts/vectors' '$CONTAINER_APP_ROOT/brainiac_submissions'"

echo "[2/6] Copying BrainIAC retrieval script"
docker cp "$HOST_CODE_ROOT/brainiac_cosine_retrieval.py" "$CONTAINER_NAME:$CONTAINER_APP_ROOT/brainiac_cosine_retrieval.py"

echo "[3/6] Copying BrainIAC checkpoint"
docker cp "$HOST_CHECKPOINT" "$CONTAINER_NAME:$CONTAINER_CHECKPOINT"

echo "[4/6] Copying challenge dataset into container"
docker exec "$CONTAINER_NAME" bash -lc "rm -rf '$CONTAINER_DATA_ROOT' && mkdir -p '$CONTAINER_DATA_ROOT'"
docker cp "$HOST_DATA_ROOT/." "$CONTAINER_NAME:$CONTAINER_DATA_ROOT/"

echo "[5/6] Installing runtime dependencies in container"
docker exec "$CONTAINER_NAME" bash -lc "python -m pip install --quiet 'monai==1.3.2' 'nibabel>=5.3,<6'"

echo "[6/6] Running BrainIAC cosine retrieval on GPU"
docker exec "$CONTAINER_NAME" bash -lc "cd '$CONTAINER_APP_ROOT' && python brainiac_cosine_retrieval.py --data-root '$CONTAINER_DATA_ROOT' --checkpoint '$CONTAINER_CHECKPOINT' --vectors-dir '$CONTAINER_VECTORS_DIR' --out '$CONTAINER_SUBMISSION' --device cuda --batch-size 4"

echo "[done] Copying artifacts back to host"
mkdir -p "$HOST_CODE_ROOT/artifacts/vectors" "$HOST_CODE_ROOT/submissions"
rm -rf "$HOST_CODE_ROOT/artifacts/vectors/brainiac_cosine"
docker cp "$CONTAINER_NAME:$CONTAINER_VECTORS_DIR" "$HOST_CODE_ROOT/artifacts/vectors/"
docker cp "$CONTAINER_NAME:$CONTAINER_SUBMISSION" "$HOST_CODE_ROOT/submissions/brainiac_cosine_submission.csv"

echo "Artifacts copied to:"
echo "  $HOST_CODE_ROOT/artifacts/vectors/brainiac_cosine"
echo "  $HOST_CODE_ROOT/submissions/brainiac_cosine_submission.csv"
