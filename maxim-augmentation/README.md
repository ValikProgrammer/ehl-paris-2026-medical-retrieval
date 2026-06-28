# Pseudo-Calibrated Dataset2 Augmentation

This folder implements a data-calibrated replacement for the hand-picked
augmentation ranges described in `augmentation.tex`.

The script:

1. Builds the current template-normalization retrieval model from labelled
   `dataset1/train_pairs.csv`.
2. Scores `dataset2` validation and test pools.
3. Keeps only high-confidence pseudo-matches where:
   - the top-1 target is also the Hungarian assignment,
   - the top-1 match is mutual nearest neighbor,
   - the top-1 vs top-2 score margin passes `--min-margin`.
4. Registers accepted dataset2 queries to the T1 template and accepted dataset2
   targets to the T2 template.
5. Estimates empirical query/target rigid transform distributions.
6. Generates independently transformed dataset1 T1/T2 augmented pairs sampled
   from those empirical distributions.

Run on the server:

```bash
cd /root/maxim-augmentation
/root/miniforge3/envs/brainiac/bin/python pseudo_calibrated_augmentation.py \
  --data-root /root/.cache/kagglehub/competitions/ehl-paris-medical-image-retrieval \
  --out-dir output \
  --grid 44 \
  --copies 5
```

Fast calibration-only run:

```bash
/root/miniforge3/envs/brainiac/bin/python pseudo_calibrated_augmentation.py --skip-generate
```

Main outputs:

- `output/pseudo_matches.csv`
- `output/transform_estimates.csv`
- `output/train_pairs_pseudo_calibrated.csv`
- `output/images/*.nii.gz`
- `output/summary.json`

The output CSV includes the original 350 dataset1 rows plus generated calibrated
copies. Paths to generated NIfTI files are absolute, so existing training scripts
can consume the CSV with `--train-pair-csv`.

Classical template test with the calibrated CSV:

```bash
cd /root/maxim-augmentation
/root/miniforge3/envs/brainiac/bin/python classical_template_aug_test.py \
  --data-root /root/.cache/kagglehub/competitions/ehl-paris-medical-image-retrieval \
  --train-pair-csv output/train_pairs_pseudo_calibrated.csv \
  --datasets dataset2 \
  --grid 44 \
  --assignment \
  --out output/d2_template_pseudo_calibrated_g44_hungarian.csv
```

Submit the resulting partial CSV to Kaggle and multiply the displayed score by
3 to get dataset2 MRR.

Local pseudo-d2 evaluation:

```bash
cd /root/maxim-augmentation
/root/miniforge3/envs/brainiac/bin/python evaluate_against_pseudo.py \
  --pseudo-matches output/pseudo_matches.csv \
  --submission output/d2_candidate.csv \
  --baseline output/d2_baseline.csv \
  --min-margin 0.02
```

This evaluates candidate rankings against high-confidence dataset2 pseudo-labels.
Use it only for relative screening before Kaggle submissions.

Calibrated synthetic holdout with true dataset1 labels:

```bash
cd /root/maxim-augmentation
/root/miniforge3/envs/brainiac/bin/python evaluate_calibrated_synthetic.py \
  --n-train 250 \
  --n-eval 60 \
  --grid 44 \
  --opt-size 20
```

This deforms labelled dataset1 eval pairs using the empirical dataset2 transform
distribution and evaluates with the true diagonal labels.

Compare several calibrated-synthetic variants:

```bash
cd /root/maxim-augmentation
/root/miniforge3/envs/brainiac/bin/python compare_calibrated_variants.py \
  --n-train 250 \
  --n-eval 60 \
  --grid 44 \
  --opt-sizes 20,24 \
  --angles 14,18 \
  --shifts 0,2 \
  --maxiters 100,120 \
  --out output/calibrated_variant_scores.csv
```
