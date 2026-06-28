# STATUS

_Last updated: 2026-06-27_

## Headline

- **Best Kaggle public score: `0.90444`** — `submissions/mix_hung_d1template_d2template_d3grid.csv` — GOAL (0.9+) HIT.
- Session progression: `0.65137` → `0.77071` → `0.80127` → **`0.90444`**.
- Recipe: d1 = template normalization g44; d2 = template normalization g44;
  d3 = grid-intensity feature **no registration** (already-aligned). All + Hungarian.
- All wins are CLASSICAL, augmentation-free (only original 350 d1 pairs).

## Per-dataset MRR (current best mix)

| dataset | method | approx MRR | status |
|---|---|---|---|
| dataset1 | template normalization g44 + Hungarian | ~0.964 | near ceiling |
| dataset2 | template normalization g44 + Hungarian | ~0.749 | **now weakest** |
| dataset3 | grid feature no-register g44 + Hungarian | ~1.000 | solved |

Overall = (0.964 + 0.749 + 1.000) / 3 ≈ 0.904.

Two feature modes in `d2_template_retrieval.py`:
- default = register to same-modality template (for pose-shifted sets d1, d2).
- `--no-register` = raw downsampled grid feature (for already-aligned d3).
Both feed the d1-trained PCA/Ridge map + Hungarian. The grid-intensity feature is
far stronger than the old classical handcrafted features (drove d1 0.877->0.964
and d3 0.690->~1.0).

KEY: template normalization (register each volume to same-modality d1 mean
template) helps registered/deformed sets (d1, d2) but BREAKS d3 — d3 targets are
intraop, already in query space, so registering to a preop template misaligns
surgical changes (d3 MRR 0.690 -> 0.251 with template). Verified on Kaggle.

## What changed this session

1. Built `synthetic_d2_eval.py` + `d2_methods.py`: local synthetic-d2 validator.
   Applies independent rigid+elastic warps to labelled d1 pairs to mimic d2 and
   scores methods offline (Hungarian-recovery MRR). Calibrated so canonical ≈ 0.26
   matches real d2. **Use before spending Kaggle submissions.**
2. Discovered template normalization (synthetic 0.55 vs canonical 0.26). Built
   `d2_template_retrieval.py` (caches `.d2cache/`).
3. Verified on Kaggle: d2 partial 0.24963 (d2 MRR ~0.75); full mix 0.77071.
4. Augmented data (server): `train_pairs_aug_geom_contrast_1k.csv` = 850 pairs
   (350 clean + 500 geom+contrast aug). Confirmed valid + trainable.
5. Launched Track B: contrastive 3D training on augmented data on ROCm GPU
   (`contrastive_3d_train.py`, now supports `--train-pair-csv` + subject-aware
   holdout). Running in container `rocm` at `/app/runs/aug`.

## Track B (GPU augmented training) — DEAD END

Contrastive 3D dual-encoder trained on the 850 augmented pairs (subject-aware
split) reached only holdout_mrr ~0.04 / all_gallery ~0.009 by epoch 35 — far
below the classical 0.90. The scratch deep model does not learn competitive
embeddings even with geom+contrast augmentation. Killed to free GPU. The
augmented data could still be used to refit the classical PCA/Ridge map for
robustness, but the classical pipeline already dominates.

## Submit log (0.95 push)

- 2026-06-27 19:42 — `d2_template_augfit_g44_hungarian.csv` → **0.20931** (d2 MRR ~0.628; REGRESSED vs 0.749).
- 2026-06-27 19:42 — `mix_hung_d1template_d2augfit_d3grid.csv` → **0.86412** (REGRESSED vs 0.90444).

### Summary @ augfit regression
Synthetic-augmented map refit hurt real d2 (0.749→0.628). The harness predicted
+0.11 but it did NOT transfer — the map overfit the synthetic deform/contrast
distribution. SECOND confirmed case (after grid56) that the harness is unreliable
for changes that depend on the data DISTRIBUTION (map retraining, resolution),
while it IS reliable for geometry/architecture method ranking (template vs
canonical vs affine — all transferred correctly). BEST stays 0.90444.

Revised plan for 0.95: stop retraining the map on synthetic data. Attack the d2
error TAIL = registration failures (rigid stuck in local minima -> wrong frame).
Registration robustness (PCA-axis init, more multi-starts, coarse-to-fine) is a
geometry/optimization fix, the category the harness predicts reliably. Also worth:
real server geom+contrast aug via --fit-pair-csv (real distribution, may transfer
unlike synthetic).

## Alpha ternary search — NEGATIVE (alpha is inert)

- 2026-06-27 19:59 — `mix_a190.csv` → 0.90444; `mix_a380.csv` → 0.90444 (= anchor a100).
- Diffed submissions across alpha in [0, 2000]: only ranking TAILS change; top-1 is
  NEVER changed in any pool; val (public) rows essentially unchanged until a=2000.
- CONCLUSION: Ridge alpha does not affect MRR here. Hungarian forces the assigned
  target to rank 1, so MRR depends only on the permutation, which is stable to alpha.
  Ternary/grid search over alpha is futile. Lever = d2 assignment accuracy itself.

## More 0.95 attempts (all REGRESSED — best stays 0.90444)

- affine reg (harness 0.39<0.56), deformable (washes subject shape), robust/PCA-init
  reg (harness 0.53, neutral at 12deg), synth-augfit (real d2 0.749->0.628),
  rich multi-channel feature (harness +0.05 BUT real mix 0.88978 < 0.90444).
- HARDENED LESSON: the synthetic harness only predicts LARGE architectural wins
  (template vs canonical, ~2x — transferred). Small harness gains (+0.05) are
  synthetic-distribution overfit and do NOT transfer to real d2. Stop trusting
  incremental harness deltas; only submit qualitatively different methods.
- Remaining classical hope: server real geom+contrast augmented map refit (running,
  fit-feature ~500/850). If it fails, 0.95 likely needs a pretrained 3D medical
  encoder fine-tuned on the 850 aug pairs (large effort, GPU), since the classical
  template+pca_ridge d2 appears near its ~0.75 ceiling.

## Real-aug refit — REGRESSED (best stays 0.90444)

- 2026-06-27 20:44 — `d2_template_realaug_g44_hungarian.csv` → 0.18672 (d2 MRR ~0.56).
- 2026-06-27 20:44 — `mix_hung_d1template_d2realaug_d3grid.csv` → 0.84153.
- Both SYNTHETIC and REAL geom+contrast augmentation refit HURT d2 (0.749 -> 0.628 /
  0.56). Augmenting the cross-modal map degrades it — the aug distribution does not
  match real d2's transforms. Augmentation is a dead end for the classical map.

## Status of the 0.95 push (honest)

Classical levers exhausted; d2 sits at its ~0.749 ceiling for template+pca_ridge.
Tried and FAILED/neutral: alpha (inert), grid56, affine, deformable, robust/PCA-init
reg, rich features, synth-augfit, real-augfit. Best verified = 0.90444. Reaching
0.95 needs d2 ~0.886 (+0.14 on the hardest dataset) and likely a higher-ceiling
model (pretrained 3D medical encoder fine-tuned on the 850 aug pairs) — scratch
contrastive already failed (0.04), so this is uncertain and large effort.

## Research findings (Firecrawl, 4 subagents) — methods to try for d2

Theory: fine-tuning distorts pretrained features under distribution shift (arXiv
2202.10054) -> explains our fine-tune/refit regressions; stay frozen/no-training.

Ranked (ROI x feasibility for 350 pairs, CPU + ROCm GPU):
1. MIND / MIND-SSC self-similarity descriptor as the feature front-end (CPU, no
   training; modality-independent + locally warp-tolerant). Implemented as
   `m_template_mind` / `mind_descriptor` in d2_methods.py. TESTING ON HARNESS.
2. SynthMorph/EasyReg or BrainMorph (pretrained, contrast-agnostic registration):
   use POST-REGISTRATION RESIDUAL as the d2 cost matrix -> Hungarian. Highest d2
   ceiling. GPU, top-K rerank. Caveat: residual not registration-success (inter-
   subject reg also works). Repos: alanqrwang/brainmorph, mattiaspaul/deedsBCV,
   FreeSurfer EasyReg/SynthMorph.
3. SynthSeg region-volume / label-map matching (pretrained, contrast-invariant,
   warp-robust region volumes). github.com/BBillot/SynthSeg.
4. Raptor train-free 3D embedding (sriramlab/raptor); BrainIAC FROZEN + Procrustes
   head (not fine-tune).
AVOID: GAN/diffusion synthesis (data too small), scratch contrastive (failed 0.04),
fine-tuning (distorts), disease foundation models.

## Push toward 0.95 (goal)

Need MRR sum 2.85 (currently 2.713). d3~1.0 maxed, d1~0.964 near ceiling, so d2
(~0.749) must reach ~0.886. Harness findings:
- affine registration: 0.387 < rigid 0.558 — HURTS (extra DOF warp away subject
  shape; rigid is the sweet spot, deformable would be worse).
- **augmented map refit (geom+contrast): 0.669 vs 0.558 (+0.11)** — refitting
  PCA/Ridge on augmented pairs helps. Generating real-d2 version now
  (`--synth-aug-k 2`). Also have `--fit-pair-csv` for the server's real 850 aug pairs.
- Next if needed: registration robustness (more multi-starts / PCA-axis init) for
  the mis-assigned tail; more aug copies; real server augmentation.

## BrainIAC fine-tune (deep, in progress)

The augmentation's correct consumer. `brainiac_finetune.py` fine-tunes the
pretrained BrainIAC ViT (MONAI, 96^3, /app/BrainIAC.ckpt) + a projection head with
symmetric InfoNCE to align T1/T2, trained on the stronger augmented manifest
(`train_pairs_aug_geom_contrast_1k_plus_stronger.csv`, 2100 pairs = 350 clean +
1750 aug, 5 variants/subject). 1788 train / 312 holdout (subject-aware). Running
on the ROCm GPU; monitoring holdout all-gallery MRR vs the scratch failure (0.04)
and frozen-cosine (0.15). On completion it auto-embeds all pools + Hungarian +
writes /app/submissions/brainiac_finetune_submission.csv. Then pull, mix, submit.
This is the remaining higher-ceiling shot at 0.95; uncertain.

## Submit log (cont.)
- 2026-06-28 06:14 — `siftrank_d2.csv` → **0.14523** (d2 MRR ~0.436; SIFT-Rank fails cross-modal+warp).
- 2026-06-28 06:14 — `mix_hung_d1template_d2sift_d3grid.csv` → **0.80004** (regressed).
- 2026-06-28 05:42 — `mix_hung_d1d2sliceview_d3grid.csv` → **0.81042** (regressed; slice-pooling = global descriptor, same wall).
- 2026-06-28 05:36 — `d2_deeds_rerank.csv` → **0.25061** (d2 MRR ~0.752; ties template 0.749).
- 2026-06-28 05:36 — `mix_hung_d1template_d2deeds_d3grid.csv` → **0.90541** (NEW BEST, +0.001).
  deeds registration-residual discriminated 6/6 on registered d1 but only TIES template
  on d2's independent-elastic warp — the residual margin collapses there. Marginal win.
  d2 ceiling ~0.75 confirmed across template / registration / augmentation / deep.
- 2026-06-27 23:15 — `mix_hung_d1d2strongaug_d3grid.csv` (2100-pair aug map refit) → **0.80590**.
  Worse than 1k-aug (0.842) and best (0.904). Augmentation-refit hurts the linear map
  MONOTONICALLY (more aug = worse). Definitive dead end. Best stays 0.90444.

## deeds registration-residual — VALIDATED on real data, building d2 reranker

deedsBCV (MIND-SSC deformable registration, built locally `make SLOW=1`) post-reg
MIND residual discriminates same-subject cross-modal: 6/6 real d1 pairs had the
true T2 as the minimum residual (diag ~0.10 vs off-diag ~0.13-0.16). This is the
research-backed high-ceiling lever for d2. `deeds_d2_rerank.py`: template top-K
prefilter -> deeds-residual rerank -> Hungarian. Running locally (~2h, top-K=6,
96^3, ~8s/pair). On finish: merge into best mix, submit directly.

Also running: strong-aug map refit on server (2100-pair manifest), per user
request to fit current best on server data and submit directly (no synthetic gate).

## In flight

- **Track A2**: template normalization on dataset1 + dataset3
  (`d2_template_retrieval.py --datasets dataset1 dataset3`). Tests whether the
  template trick also helps d3 (new bottleneck). Output:
  `submissions/d1d3_template_g44_hungarian.csv`.
- **Track B**: GPU contrastive training on 850 augmented pairs. Watching
  `holdout_mrr` / `all_gallery_mrr` vs the old failed run (~0.12).

## Next steps (priority order)

1. **dataset3** (biggest remaining lever): test template normalization; if no
   gain, try registration/robust-to-missing-tissue features (surgery removes
   anatomy). d3 0.69 -> target 0.8.
2. **Squeeze dataset2**: tune template grid (44->56/64), PCA components/alpha,
   add affine (not just rigid) registration, multi-start refinement.
3. **Track B**: if augmented contrastive beats classical on the synthetic/holdout
   proxy, blend or replace; else keep classical.
4. Re-assemble best all-template mix, validate, submit.

## Verified Kaggle scores (this session)

```
mix_hung_d1pca_d2template_d3pca.csv     0.77071   NEW BEST
mix_hung_d1pca_d2canon20_flipaug_d3pca  0.65137   prev best
d2_template_g44_hungarian.csv (partial) 0.24963   d2 MRR ~0.749
d2_canonical_pca20_c128_flipaug (part)  0.13029   d2 MRR ~0.391
```

## Commands

Run template d2:
```bash
.venv/bin/python d2_template_retrieval.py --datasets dataset2 --grid 44 --assignment \
  --out submissions/d2_template_g44_hungarian.csv
```

Local synthetic-d2 validation (no Kaggle needed):
```bash
.venv/bin/python synthetic_d2_eval.py --n-eval 60 --n-train 200 --grid 44 \
  --max-rot-deg 12 --max-shift 6 --elastic-sigma 8 --elastic-alpha 3 \
  --methods canonical template
```

Validate + submit:
```bash
.venv/bin/python classical_retrieval.py validate submissions/<file>.csv
kaggle competitions submit -c ehl-paris-medical-image-retrieval -f submissions/<file>.csv -m "<msg>"
```

## Submit log

- 2026-06-27 19:00 — `d2_template_g44_hungarian.csv` → **0.24963** (d2 MRR ~0.75).
- 2026-06-27 19:00 — `mix_hung_d1pca_d2template_d3pca.csv` → **0.77071** (best at the time).
- 2026-06-27 19:05 — `d1_template_g44_hungarian.csv` → **0.32147** (d1 MRR ~0.964; template helps d1).
- 2026-06-27 19:05 — `d3_template_g44_hungarian.csv` → **0.08368** (d3 MRR ~0.251; template HURTS d3).
- 2026-06-27 19:06 — `mix_hung_d1template_d2template_d3pca.csv` → **0.80127** (NEW BEST).

### Summary @ 0.80127
Swapped d1 from pca_ridge to template normalization (d1 MRR 0.877→0.964) on top of
the d2 template win. d3 kept on pca_ridge because template breaks it.

- 2026-06-27 19:11 — `d3_gridfeat_g44_hungarian.csv` → **0.33333** (d3 MRR ~1.0; grid feature, no register).
- 2026-06-27 19:12 — `mix_hung_d1template_d2template_d3grid.csv` → **0.90444** (NEW BEST, GOAL HIT).

- 2026-06-27 19:16 — `mix_hung_d1d2template_g56_d3grid.csv` → **0.85000** (REGRESSED; grid56 d1+d2 hurt real data).

- 2026-06-27 19:18 — `mix_hung_d1g44_d2g56_d3grid.csv` → **0.86609** (d2-only g56 also regressed).

### Summary @ 0.85000 / 0.86609 (grid56 dead end — BEST stays 0.90444)
grid56 hurts BOTH d1 and d2 on real data (full g56=0.850, d2-only g56=0.866, vs
g44 best 0.90444). The synthetic-d2 harness favored grid56 (+0.16) but it did NOT
transfer. LESSON: the synthetic harness is reliable for RELATIVE METHOD ranking
within d2-style distortion (it correctly picked template >> canonical), but NOT
for resolution/grid choices on real data — likely g56 overfits PCA with only 350
train pairs. grid44 is optimal. Locking 0.90444.

Open d2 levers (need real validation, synthetic unreliable for absolutes; goal
already met so these are bonus): affine (vs rigid) registration; refit PCA/Ridge
on the 850 augmented pairs for robustness; alpha/components tuning; multi-start
registration refinement.

### Summary @ 0.90444
d3 was solved by using the grid-intensity feature WITHOUT registration (d3 is
already aligned, so registration was the only thing holding it back; pca_ridge on
old classical features capped it at 0.69, grid feature → ~1.0). Combined with d1+d2
template normalization → overall 0.904.

Now d2 (~0.749) is the lone weak point. Synthetic-d2 harness shows template at
grid56 = 0.719 vs grid44 = 0.558 (+0.16) — regenerating d1+d2 at grid56 in flight
(`d1d2_template_g56_hungarian.csv`), expected to push d2 real MRR up materially.

Next steps:
1. Fold in grid56 d1+d2 (validated locally), re-submit. Maybe try grid64.
2. Augmentation (Track B GPU) not yet contributing; could also robustify the
   classical PCA/Ridge map by fitting on the 850 augmented pairs.
3. d2 affine (vs rigid) registration; multi-start refinement.
