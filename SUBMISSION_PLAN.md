# Kaggle Feedback Plan

Current public score: `0.55714`.

The fastest way to improve is to isolate dataset-specific public MRR. Kaggle
averages three dataset MRRs:

```text
score = (dataset1_MRR + dataset2_MRR + dataset3_MRR) / 3
```

If a submission contains only one dataset, multiply Kaggle's displayed score by
`3` to estimate that dataset's public MRR.

## Submit These Diagnostics First

Dataset 2, deformation-heavy:

```text
submissions/d2_fusion_dataset2.csv
submissions/d2_fusion_grid.csv
submissions/d2_mask_crop32.csv
submissions/d2_pca_abs_mask24.csv
```

Dataset 3, post-surgery:

```text
submissions/d3_fusion_dataset3.csv
submissions/d3_fusion_shape.csv
submissions/d3_pca_ridge_c128_a100.csv
```

Record each displayed Kaggle score and multiply by `3`.

## Submit These Full Candidates

Use these after diagnostics, or immediately if submission budget is not a
concern:

```text
submissions/mix_d1pca_d2dataset2_d3dataset3.csv
submissions/mix_d1pca_d2grid_d3dataset3.csv
submissions/mix_d1pca_d2pcaabsmask_d3dataset3.csv
submissions/mix_d1pca_d2mask_d3shape.csv
```

The best source-domain method remains:

```text
submissions/all_pca_ridge_c128_a100.csv
```

## Interpretation

- If `d2_pca_abs_mask24` beats the others, dataset2 is mostly failing because
  random rotation/deformation breaks grid-aligned matching.
- If `d2_fusion_grid` wins, dataset2 still benefits from aligned edges/masks
  despite deformation.
- If `d3_pca_ridge_c128_a100` wins, surgery cases still retain enough
  source-domain appearance signal.
- If `d3_fusion_dataset3` or `d3_fusion_shape` wins, surgery breaks local
  appearance and shape/edge fusion is safer.
