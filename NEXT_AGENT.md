# NEXT_AGENT.md

## Current user intent

User changed direction: **do not train the model right now** and **do not submit to Kaggle without explicit approval**.

Next task is to generate more training data by augmenting `dataset1/train_pairs.csv`, then leave augmented files/CSV ready for later training.

## Server access data

Sensitive. Do not publish.

- IP: `134.199.198.104`
- Jupyter URL: `http://134.199.198.104/lab?token=tP1Kw7bI4y0kM0qNhesV3OezEd1Ii1YDHTCyFfuUhgyRsKXzd`
- Jupyter token: `tP1Kw7bI4y0kM0qNhesV3OezEd1Ii1YDHTCyFfuUhgyRsKXzd`
- root password given by user: `Drake2550`
- User said repo path on server: `~/ehl-paris-2026-medical-retrieval`
- In practice, Jupyter terminal opens inside container at `/app`.
- `ssh root@134.199.198.104` failed with `Permission denied (publickey,password)`.
- `docker` is not available inside the Jupyter terminal because that terminal is already inside the container.

## Server observations already made

Jupyter API works.

Terminal `/app` contains:

- `/app/ehl-paris-medical-image-retrieval`
- `/app/BrainIAC.ckpt`
- `/app/brainiac_adapter_train.py`
- `/app/brainiac_cosine_retrieval.py`
- `/app/contrastive_3d_train.py`
- `/app/runs`

Disk:

```bash
df -h .
# overlay 697G total, 155G used, 543G available

du -sh /app/ehl-paris-medical-image-retrieval
# 26G

wc -l /app/ehl-paris-medical-image-retrieval/dataset1/train_pairs.csv
# 351 lines = 350 train pairs + header
```

Dependencies in `/app`:

```text
torch 2.10.0+git8514f05
torch.cuda.is_available() = True
monai 1.3.2
nibabel ok
matplotlib missing
```

## Files changed locally in this repo

- `slice_clip_baseline.py`
  - Added train-time augmentation presets: `geom`, `geom_contrast`, `light`, `brief`.
  - Added CLI overrides for `--epochs`, `--batch-size`, `--learning-rate`, `--num-workers`, `--device`.
  - Added loss CSV/PNG outputs.

- `server_run_slice_clip_aug.sh`
  - Runs SliceCLIP experiments for `PRESETS`.
  - Default `PRESETS="geom geom_contrast"`.
  - Uses `python3`, not `uv`.

- `generate_augmented_train_data.py`
  - New script to generate augmented `.nii.gz` training pairs and a new CSV.
  - It has not yet been uploaded to the server.
  - It was only syntax-checked locally with `python -m py_compile`.
  - It requires `scipy`; local machine does not have scipy. Need check/install on server.

## Files already uploaded to server

These were uploaded through Jupyter contents API to `/shared-docker` and then copied into `/app`:

- `/app/slice_clip_baseline.py`
- `/app/server_run_slice_clip_aug.sh`

`generate_augmented_train_data.py` still needs upload/copy.

## Important: do not run these yet unless user approves

Do not run:

```bash
bash /app/server_run_slice_clip_aug.sh
python3 /app/brainiac_adapter_train.py ...
kaggle competitions submit ...
```

No Kaggle submit without explicit user approval. User requested at most 3 submissions at a time.

## Upload `generate_augmented_train_data.py` to server

Run locally from repo root:

```bash
python3 - <<'PY'
import base64, json, urllib.request
from pathlib import Path

TOKEN = "tP1Kw7bI4y0kM0qNhesV3OezEd1Ii1YDHTCyFfuUhgyRsKXzd"
BASE = "http://134.199.198.104/api/contents"
name = "generate_augmented_train_data.py"
data = Path(name).read_bytes()
payload = json.dumps({
    "type": "file",
    "format": "base64",
    "content": base64.b64encode(data).decode(),
}).encode()
req = urllib.request.Request(
    f"{BASE}/{name}?token={TOKEN}",
    data=payload,
    method="PUT",
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=30) as r:
    print(name, r.status)
PY
```

Then in Jupyter terminal at `http://134.199.198.104/lab?...`, run:

```bash
cd /app
cp /shared-docker/generate_augmented_train_data.py /app/generate_augmented_train_data.py
python3 -m py_compile /app/generate_augmented_train_data.py
python3 - <<'PY'
import scipy, nibabel, numpy
print("scipy", scipy.__version__)
print("nibabel ok")
print("numpy", numpy.__version__)
PY
```

If `scipy` is missing in `/app`, install it:

```bash
pip install scipy tqdm
```

## Smoke-test data generation

Run this first. It creates only 2 source pairs x 1 augmented copy.

```bash
cd /app
python3 /app/generate_augmented_train_data.py \
  --data-root /app/ehl-paris-medical-image-retrieval \
  --train-pair-csv /app/ehl-paris-medical-image-retrieval/dataset1/train_pairs.csv \
  --output-dir /app/ehl-paris-medical-image-retrieval/dataset1_aug_geom_contrast_smoke \
  --output-csv /app/ehl-paris-medical-image-retrieval/dataset1/train_pairs_aug_geom_contrast_smoke.csv \
  --copies 1 \
  --preset geom_contrast \
  --limit 2 \
  --overwrite
```

Verify:

```bash
wc -l /app/ehl-paris-medical-image-retrieval/dataset1/train_pairs_aug_geom_contrast_smoke.csv
find /app/ehl-paris-medical-image-retrieval/dataset1_aug_geom_contrast_smoke -type f -name '*.nii.gz' | wc -l
head -3 /app/ehl-paris-medical-image-retrieval/dataset1/train_pairs_aug_geom_contrast_smoke.csv
```

Expected:

- CSV should have 5 lines: header + 2 original rows + 2 augmented rows.
- NIfTI count should be 4: 2 augmented queries + 2 augmented targets.

## Full augmented dataset generation

If smoke-test is fine, run:

```bash
cd /app
nohup python3 /app/generate_augmented_train_data.py \
  --data-root /app/ehl-paris-medical-image-retrieval \
  --train-pair-csv /app/ehl-paris-medical-image-retrieval/dataset1/train_pairs.csv \
  --output-dir /app/ehl-paris-medical-image-retrieval/dataset1_aug_geom_contrast_c2 \
  --output-csv /app/ehl-paris-medical-image-retrieval/dataset1/train_pairs_aug_geom_contrast_c2.csv \
  --copies 2 \
  --preset geom_contrast \
  --overwrite \
  > /app/runs/generate_aug_geom_contrast_c2.log 2>&1 &
```

Monitor:

```bash
tail -f /app/runs/generate_aug_geom_contrast_c2.log
```

Verify when complete:

```bash
wc -l /app/ehl-paris-medical-image-retrieval/dataset1/train_pairs_aug_geom_contrast_c2.csv
find /app/ehl-paris-medical-image-retrieval/dataset1_aug_geom_contrast_c2 -type f -name '*.nii.gz' | wc -l
du -sh /app/ehl-paris-medical-image-retrieval/dataset1_aug_geom_contrast_c2
```

Expected:

- CSV lines: `1051` = header + 350 original + 700 augmented rows.
- NIfTI files: `1400` = 350 pairs x 2 copies x 2 modalities.

## What augmentations are generated

Preset `geom_contrast`:

- affine rotation up to about `18 deg`
- translation up to about `12 voxels`
- scale `0.90-1.10`
- elastic deformation with random smooth displacement, applied with probability `0.45`
- optional bias field, contrast gamma, intensity scale/shift, Gaussian noise

Query and target are augmented independently, so each generated positive pair simulates geometric mismatch between T1 and T2.

## Later training command, only after user asks

Train SliceCLIP using generated CSV:

```bash
cd /app
DATA_ROOT=/app/ehl-paris-medical-image-retrieval \
DEVICE=cuda \
EPOCHS=200 \
BATCH_SIZE=128 \
NUM_WORKERS=4 \
PRESETS="none" \
python3 /app/slice_clip_baseline.py \
  --data-root /app/ehl-paris-medical-image-retrieval \
  --train-pair-csv /app/ehl-paris-medical-image-retrieval/dataset1/train_pairs_aug_geom_contrast_c2.csv \
  --query-csv /app/ehl-paris-medical-image-retrieval/dataset1/val_queries.csv \
  --gallery-csv /app/ehl-paris-medical-image-retrieval/dataset1/val_gallery.csv \
  --query-csv /app/ehl-paris-medical-image-retrieval/dataset1/test_queries.csv \
  --gallery-csv /app/ehl-paris-medical-image-retrieval/dataset1/test_gallery.csv \
  --query-csv /app/ehl-paris-medical-image-retrieval/dataset2/val_queries.csv \
  --gallery-csv /app/ehl-paris-medical-image-retrieval/dataset2/val_gallery.csv \
  --query-csv /app/ehl-paris-medical-image-retrieval/dataset2/test_queries.csv \
  --gallery-csv /app/ehl-paris-medical-image-retrieval/dataset2/test_gallery.csv \
  --query-csv /app/ehl-paris-medical-image-retrieval/dataset3/val_queries.csv \
  --gallery-csv /app/ehl-paris-medical-image-retrieval/dataset3/val_gallery.csv \
  --query-csv /app/ehl-paris-medical-image-retrieval/dataset3/test_queries.csv \
  --gallery-csv /app/ehl-paris-medical-image-retrieval/dataset3/test_gallery.csv \
  --augmentation-preset none \
  --epochs 200 \
  --batch-size 128 \
  --num-workers 4 \
  --device cuda \
  --loss-csv /app/runs/slice_clip_aug/offline_aug_c2_e200/history.csv \
  --loss-plot /app/runs/slice_clip_aug/offline_aug_c2_e200/loss.png \
  --out /app/submissions/slice_clip_offline_aug_c2.csv
```

Do not submit this CSV to Kaggle until user approves.
