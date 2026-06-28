# Architecture — Cross-modal Brain MRI Retrieval

A fully **classical** (no deep learning, no augmentation) registration-based
retrieval pipeline for the *EHL Paris: Medical Image Retrieval* competition.

- **Best leak-free public score: 0.80462** (d1+d2 template, **d3 = brain-bbox + MIND**).
- **0.91981** (d2 deformable-Demons + slab) and **0.90444** (d2 template) both use the
  **d3 raw-grid** path, which rides the dataset3 geometry shortcut (§5, §11). The
  external **BraTS** identity-matching exploit was investigated and **removed** (§11).

Up from a 0.651 baseline.

---

## 1. Problem

For each **query** volume (3D T1 post-contrast MRI) we rank every **target** (3D
T2 MRI) in the gallery so the same-subject target ranks as high as possible. All
images are NIfTI volumes in RAS orientation, 1 mm isotropic; shapes vary (e.g.
240×240×155).

Three independent pools:

| Pool | Geometry | Labels | Notes |
|---|---|---|---|
| **dataset1** | registered preop pairs; query & target share a grid | **350 labeled pairs** (only training data) | the source domain |
| **dataset2** | each query & target **independently** warped by random rigid (rotation+shift) **and** elastic deformation | none | the hard set |
| **dataset3** | preop→intraop; intraop target resampled into query space (already shares geometry) but surgery moves/removes tissue | none | already aligned |

Sizes: d1 40/100 val/test, d2 40/100, d3 20/77 → full submission = **377 rows**.

## 2. Metric & the bijection insight

Score is mean reciprocal rank, averaged over datasets:

```
score = (MRR1 + MRR2 + MRR3) / 3
```

A single-dataset partial submission displays `MRR_d / 3`, so multiplying by 3
recovers that dataset's MRR — the basis of all diagnostics.

**Bijection.** In each pool #queries == #targets and the match is one-to-one. So
retrieval is exactly a **linear assignment problem**: find the permutation pairing
queries to targets. This motivates the Hungarian re-rank (§6).

## 3. Preprocessing — intensity normalization

Each volume `V` is mapped to a normalized field on a fixed `G×G×G` grid (`G=44`):

1. Load NIfTI; if 4D keep first channel; NaN/Inf → 0.
2. Foreground mask `M = {|V| > 1e-6}`.
3. Robust window: `lo, hi` = 1st / 99th percentile of `V` over `M`.
4. Clip-and-scale: `V ← clip((V-lo)/(hi-lo), 0, 1)`, background → 0.
5. Resample to `G³` via trilinear interpolation (`map_coordinates`, order 1).

This makes intensities comparable across scanners and turns every volume into a
fixed-length `G³` vector.

## 4. Cross-modal embedding — PCA + Ridge

Query (T1) and target (T2) are different modalities, so raw intensities aren't
directly comparable. Learn the cross-modal link on the 350 labeled d1 pairs.

Let `φ(·) ∈ ℝ^(G³)` be the flattened, mean-centered, L2-normalized grid feature.
From pairs `{(φ_q_i, φ_t_i)}`:

1. Whitening PCA on queries `P_q: ℝ^(G³)→ℝ^c` and a **separate** whitening PCA on
   targets `P_t`, with `c = min(128, n-1)`.
2. Ridge regression `W` mapping query codes → target codes:

   ```
   W = argmin_W  Σ_i ‖ W·P_q(φ_q_i) − P_t(φ_t_i) ‖²  +  α‖W‖²_F ,   α = 100
   ```

At inference: `u = norm(W·P_q(φ_q))`, `v = norm(P_t(φ_t))`, similarity = `uᵀv`.
PCA denoises/decorrelates each modality; Ridge learns the linear T1→T2 code map so
matched subjects land close in the shared target-code space.

> Note: Ridge `α` does not change MRR rank-1 — the Hungarian step fixes assignment
> regardless of `α`.

## 5. Geometry — template normalization (the d2 fix)

The embedding assumes query and target share a coordinate frame. True for d1
(registered), false for d2 (every volume has a different pose). We restore a shared
frame with **template normalization**.

**Templates.** Because d1 pairs are registered, the voxelwise means

```
T1 = mean_i(V_q_i)   T2 = mean_i(V_t_i)
```

are two sharp templates that **share one grid** (mutually registered mean T1 / mean
T2 brains).

**Registration.** Rigidly align each query to `T1` and each target to `T2` —
**intramodal** registration (same modality on both sides), far more robust than
cross-modal. Optimize the 6-DOF rigid transform `θ` to maximize foreground
normalized cross-correlation:

```
θ* = argmax_θ  NCC( T_m , R_θ[V] )
```

Powell from 4 starts (identity + 3 seeded rotations) on a coarse 20³ copy for
speed; the recovered `θ*` is applied at full grid resolution, then fed to `φ(·)`.

**Why it works.** After normalization, same-subject query/target occupy the same
template frame, so the d1-trained map transfers to d2. Cost is **O(N)** (one
registration per volume), not O(N²). This single step:
- d2 MRR ~0.39 → ~0.75
- d1 MRR ~0.877 → ~0.964

**dataset3 (aligned) — and its geometry leak.** d3 targets are resampled into the
query's physical space, so query & target already share geometry. Registering an
intraop volume to a *preop* template misaligns the surgical changes (template norm
**hurts** d3: 0.690 → 0.251). The naive `--no-register` raw-grid feature gives d3
MRR ≈ **1.0**, but that **rides the geometry leak** (§11): the unique per-subject
affine/FOV is shared by a true pair and, when the native box is resized into the
44³ cube, it is baked in as a fingerprint + voxel-wise alignment.

The **honest d3 path** removes that shortcut:

| d3 method | MRR | what it removes |
|---|---|---|
| raw-grid (`--no-register`) | **1.000** | nothing — full geometry leak |
| brain-bbox grid (`--bbox-crop`) | 0.442 | FOV/box-size fingerprint |
| **brain-bbox + MIND, grid 56** (`--bbox-crop --mind --grid 56`) | **0.700** | FOV; MIND adds modality-invariance on the aligned brain |
| template-register | 0.251 | also the pair co-registration (fully leak-free) |

We ship **bbox + MIND g56** (0.700) as the honest dataset3 method.

## 6. Deformable registration + slab fusion (the 0.91981 lever)

Rigid template normalization removes pose but not the **elastic** warp in d2. The
real winner is **deformable** registration to the template:

- `deform_lib.py::_deformable_to_template` — regularized symmetric **Demons**
  deformable registration to the same-modality template (smooth = 3, ~40 iters),
  run *after* rigid alignment. It attacks the actual elastic deformation and
  **transfers** to the real leaderboard. **+0.0143 over rigid** (0.90444 → 0.91874).
- **Per-slab features** (`_slab_feats`) + fusion adds a further small bump
  → **0.91981**.

Honest caveat (verified on Kaggle): only *registration* improvements transfer.
Reranking / fusion / Sinkhorn tricks looked great on the synthetic harness but
added ≈0 on real data (Sinkhorn actually regressed to 0.890).

## 7. One-to-one assignment (Hungarian)

Given a cosine similarity matrix `S ∈ ℝ^(N×N)` for a pool, exploit the bijection.
Solve the optimal assignment

```
π* = argmax_π  Σ_i S_{i, π(i)}
```

with `scipy.optimize.linear_sum_assignment` on `−S`. For each query, place its
assigned target first, then rank the rest by raw similarity. Large, reproducible
gain (e.g. 0.557 → 0.594 early on).

## 8. Synthetic dataset2 validation harness

Kaggle allows only ~100 submissions/day and d2/d3 are unlabeled, so blind
iteration is expensive. Build a **local** d2 proxy from labeled d1.

- **Construction.** Split the 350 d1 pairs into a clean *train* set (templates +
  PCA/Ridge map) and an *eval* set. Independently warp each eval query & target by
  a random rigid transform (±r°, ±s vox) and an elastic deformation (Gaussian
  displacement field σ, scaled by α via `map_coordinates`). Report
  Hungarian-recovery MRR (eval diagonal = ground truth).
- **Calibration.** `(r, s, σ, α) = (12, 6, 8, 3)` so the canonical baseline scores
  ≈0.26, matching its real d2 MRR. At that operating point template normalization
  scored ≈0.55 — correctly predicting the real win *before* any submission.
- **Scope & caveat.** Reliable for **relative ranking** of methods under
  d2-style distortion. **Not** reliable for absolute choices on real d1/d3 or for
  resolution: it preferred grid 56 (0.72 vs 0.56) but grid 56 **regressed** on real
  d1 and d2 (PCA overfit on only 350 train pairs). **Grid 44 is optimal.**

## 9. Full pipeline by dataset

| Dataset | Geometry handling | Feature | Map + ranking |
|---|---|---|---|
| dataset1 | template normalization (T1/T2), rigid | φ on 44³ | PCA+Ridge, cosine, Hungarian |
| dataset2 | template normalization + **deformable Demons** + slab | φ on 44³ | PCA+Ridge, cosine, Hungarian |
| dataset3 | **none** (already aligned) | φ on 44³ | PCA+Ridge, cosine, Hungarian |

End-to-end per pool:
1. Build `T1, T2` and fit `P_q, P_t, W` on d1 (once).
2. Per query/target: normalize intensities → resample to 44³ → register to its
   template (except d3) → φ.
3. Embed: `u = norm(W·P_q(φ_q))`, `v = norm(P_t(φ_t))`.
4. `S = UVᵀ`; Hungarian boost; write space-separated per-query ranking.

## 10. Results

| Submission | Public | Note |
|---|---|---|
| canon20 flipaug (prior best) | 0.65137 | session start |
| d1 pca + d2 **template** + d3 pca | 0.77071 | d2 fix |
| d1 **template** + d2 template + d3 pca | 0.80127 | template helps d1 too |
| d1 template + d2 template + d3 **grid** | 0.90444 | goal hit |
| d1 template + d2 **deformable** + d3 grid | 0.91874 | elastic reg transfers |
| d1 template + d2 **deform + slab** + d3 grid | 0.91981 | d3 grid uses geometry leak |
| **leak-free:** d1+d2 template + **d3 bbox+MIND g56** | **0.80462** | **best honest** |
| leak-free, d3 bbox+MIND g44 | 0.78971 | |
| leak-free, d3 bbox-grid, Hungarian on/off | 0.71836 / 0.65057 | Hungarian = +0.068 |
| external BraTS matching (removed) | (~1.0 reported) | data leak, not in repo |
| grid56 variants (d2) | 0.850 / 0.866 | regress (see §8) |

Per-dataset val MRR: `MRR1 ≈ 0.964`, `MRR2 ≈ 0.749` (the wall), `MRR3` = 1.0
(geometry leak) / 0.700 (leak-free bbox+MIND) / 0.251 (fully registered).

## 11. Negative results

The harness + partial submissions ruled out:
- **Pose-invariant descriptors** (intensity/gradient histograms, radial profiles,
  shape eigenvalues): too lossy; discard the spatial structure that distinguishes
  subjects (~0.11 on harness).
- **Pairwise rigid edge-NCC re-rank**: elastic warp beats rigid; optimizer stuck in
  local minima (0.20 < 0.26).
- **grid 56**: better on harness, worse on real data.
- **Deep contrastive 3D model** on 850 geom+contrast augmented pairs (ROCm GPU):
  holdout MRR ≈0.04, far below classical 0.90. Scratch model never learns
  competitive embeddings even with augmentation.
- **BrainIAC foundation-model embeddings** (cosine / adapter / patch): all below
  the classical pipeline.
- **Sinkhorn / fusion / reranking** beyond Hungarian: no transfer to real data.

## 12. Data-leak audit

Several leaderboard teams scored a perfect 1.0. Findings:

- **dataset2 — no accessible leak.** IDs are random hashes (no hash/sort/row-order
  relation), file sizes constant, NIfTI headers byte-identical constants, affines a
  single constant value, write-timestamps batched, no byte-duplicate files. d2's
  ~0.749 is the honest ceiling (≈13 methods tried).
- **dataset3 — real geometry leak.** True pairs share an exact affine/FOV (target
  resampled into query space). Matching by affine equality is a pure metadata leak;
  the raw-grid path rides it implicitly (§5). The honest fix is brain-bbox + MIND.
- **External BraTS-GLI matching — the 1.0 exploit (removed).** d1/d2 are public
  BraTS-GLI; intramodally registering each warped d2 image to the clean BraTS
  library recovers subject identity → its known T2 → the pairing (~1.0, works on
  private). The exploit code (`tmp/srv_*`, probe data) has been removed from the repo.
- **Public-LB probing.** A single-query partial submission scores `1/(120·rank)`,
  exposing each validation answer. Inflates public only (val = ~27%), not private.
  Not used.

See [`augmentation.md`](augmentation.md) for the augmentation experiments.
