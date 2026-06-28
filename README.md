# Cross-Modal Brain MRI Retrieval — EHL Paris

Classical, registration-based retrieval for the *EHL Paris: Medical Image Retrieval*
competition. For each query (3D **T1 post-contrast** MRI) rank the gallery of
**T2** MRIs so the same-subject target ranks highest. No deep learning, no
augmentation — pretrained-free, content-only.

> **Headline (leak-free):** public **0.80462**
> (`submissions/mix_leakfree_d1d2template_d3mind_g56.csv`).
> With the dataset3 geometry shortcut counted as legitimate content it is **0.904**.
> The external-data-leak route (BraTS matching) is intentionally **removed** — see §6.

---

## 1. Task & data

Three independent retrieval pools (rank a query only against its own pool):

| Pool | Geometry | Labels | Difficulty |
|---|---|---|---|
| **dataset1** | registered preop pairs; query & target share one grid | **350 labeled pairs** (the only training data) | easy (source domain) |
| **dataset2** | query & target each **independently** warped — random rigid (rot+shift) **+** nonlinear elastic | none | **hard** (the wall) |
| **dataset3** | preop→intraop; target resampled into the query's physical space (already aligned), but surgery moves/removes tissue | none | aligned, but cross-modal+surgical |

Counts: d1 40/100, d2 40/100, d3 20/77 (val/test) → full submission = **377 rows**.
All volumes are NIfTI, RAS, 1 mm isotropic; shapes vary (d1/d2 are 240×240×155).

## 2. Metric & the bijection

```
score = (MRR_d1 + MRR_d2 + MRR_d3) / 3        # mean reciprocal rank
```
A single-dataset partial submission shows `MRR_d / 3` → ×3 recovers that dataset's
MRR (basis of all diagnostics). **Public LB = the 100 validation queries (~27%);
private = the 277 test queries (73%).**

**Bijection:** in every pool #queries == #targets, one-to-one. Retrieval is a
**linear assignment problem** → solved with the Hungarian algorithm (worth **+0.068**
overall, see §7). It is the given task structure, not a leak.

## 3. Pipeline

```
NIfTI ─► intensity-normalize ─► resample to 44³ ─► [geometry step] ─► feature
      ─► PCA + Ridge cross-modal map (fit on d1) ─► cosine ─► Hungarian ─► ranking
```

**Preprocessing.** Load → foreground mask → robust 1/99-percentile window →
clip to [0,1] → trilinear resample to a fixed `44³` grid.

**Cross-modal map (PCA + Ridge).** T1 and T2 differ in appearance, so we learn the
relationship on the 350 labeled d1 pairs: whitening PCA on each modality, then Ridge
maps query-codes → target-codes. Score = cosine in the shared code space.

**Geometry step (per pool):**
- **d1, d2 — template normalization.** d1 train pairs are registered, so the mean
  T1 and mean T2 form two templates that share one grid. Each volume is rigidly
  registered to its **same-modality** template (intramodal NCC — robust). This
  removes pose so the d1-trained map transfers to d2's independent warps.
- **d3 — MIND on the aligned volumes (no template registration).** d3 is already
  co-registered, so instead of intensity we use the **MIND** descriptor
  (modality-independent self-similarity, §5) cropped to the brain bounding box.

**Hungarian.** The cosine matrix is turned into a one-to-one assignment; the
assigned target is forced to rank 1, the rest by similarity.

## 4. Results (verified on Kaggle)

### Per-dataset validation MRR (leak-free pipeline)

| dataset | method | MRR |
|---|---|---|
| dataset1 | template + Hungarian | **0.964** |
| dataset2 | template + Hungarian | **0.749** (the wall) |
| dataset3 | bbox + **MIND** g56 + Hungarian | **0.700** |

### dataset3 method ladder (shows the geometry leak)

| d3 method | MRR | note |
|---|---|---|
| raw-box grid | **1.000** | rides the FOV/affine geometry leak (see §6) |
| bbox-crop grid | 0.442 | FOV leak removed; co-registration kept |
| bbox-crop MIND g44 | 0.656 | modality-invariant feature |
| **bbox-crop MIND g56** | **0.700** | finer grid (d3 is aligned → resolution helps) |
| template-register | 0.251 | co-registration also removed (fully leak-free) |

### Full-submission scores

| submission | overall | notes |
|---|---|---|
| `mix_leakfree_d1d2template_d3mind_g56.csv` | **0.80462** | **best leak-free** |
| leak-free, d3 MIND g44 | 0.78971 | |
| leak-free, d3 bbox-grid, **with** Hungarian | 0.71836 | |
| leak-free, d3 bbox-grid, **no** Hungarian | 0.65057 | Hungarian = +0.068 |
| d1+d2 template + **d3 raw-grid** | 0.90444 | d3 uses the geometry shortcut |
| d2 Sinkhorn rerank | 0.89133 | regressed → off by default |
| external BraTS matching (removed) | 0.91981 | data leak, not in this repo |

## 5. MIND (Modality-Independent Neighbourhood Descriptor)

Replaces intensity with **local self-similarity**: for each voxel, how its local
patch relates to neighboring patches *within the same image*. That pattern is the
same across modalities (an edge is an edge in T1 and T2), turning a cross-modal
match into a quasi-mono-modal one. We compute a 6-channel field
`MIND(x,r)=exp(-Dp(x,r)/V(x))` (`Dp`=patch SSD to neighbor `r`, `V`=local variance).
It helps **d3** (aligned: fields overlap voxel-wise, 0.442→0.700) but **not d2**
(independent warp: fields don't line up — MIND fixes modality, not geometry).

## 6. Data-leak analysis (and why it's removed)

Multiple leaderboard teams hit 1.0. Investigated exhaustively (`docs/architecture.md`,
git history); findings:

- **No leak in d2** — IDs are random hashes (no hash/sort/row relation), file sizes
  constant, NIfTI headers byte-identical constants, affines a single constant,
  timestamps batched, no byte-duplicate files. d2's ~0.749 is the honest ceiling
  (≈13 methods tried, §7).
- **dataset3 geometry leak (real).** d3 targets are resampled into the query's
  physical space, so a true pair shares an **exact affine/FOV**. Matching by
  affine-equality is a pure metadata leak (trivial 1.0). Even the content path
  rides it: resizing the unique per-subject box into the 44³ cube bakes in the FOV
  fingerprint + alignment. Cropping to the brain box removes the FOV part
  (1.0→0.442); full template-registration removes the alignment too (→0.251). We
  ship the **bbox + MIND** variant (0.700) as the honest middle.
- **External BraTS matching (the 1.0 exploit).** d1/d2 are public **BraTS-GLI**
  volumes. Registering each warped d2 image intramodally to the clean public BraTS
  library recovers the subject identity → its known T2 → the pairing → ~1.0 (works
  on private). This is a genuine data leak; the exploit code has been **removed**.
- **Public-LB probing.** A single-query partial submission scores `1/(120·rank)`,
  revealing each validation answer — inflates public only, not private. Not used.

## 7. What did not work (negative results)

dataset2 is the wall; everything below tied or regressed vs template's 0.749:
deeds registration-residual (0.752, tie), 3D SIFT-Rank keypoints (0.436), SynthSeg
region-volumes (0.432), canonical PCA-axis, MIND-after-registration (neutral),
affine/deformable registration, grid56, augmentation refit (synthetic & real, both
worse), rich/sliceview features, Ridge-alpha (inert under Hungarian), Sinkhorn
(0.891), scratch 3D contrastive (0.04), BrainIAC frozen/fine-tuned (0.15/0.045).
Root cause: cross-modal + **independent** rigid+elastic warp + only 350 train pairs
destroys the per-subject signal honest methods need.

## 8. Reproduce

```bash
# environment
python3 -m venv .venv && .venv/bin/pip install numpy scipy scikit-learn nibabel

DR=ehl-paris-medical-image-retrieval

# d1 + d2 (template normalization, Hungarian)
.venv/bin/python d2_template_retrieval.py --data-root $DR --datasets dataset1 dataset2 \
  --splits val test --grid 44 --assignment --out submissions/_d1d2.csv

# d3 (leak-free: brain bbox + MIND, finer grid)
.venv/bin/python d2_template_retrieval.py --data-root $DR --datasets dataset3 \
  --splits val test --grid 56 --assignment --no-register --bbox-crop --mind \
  --out submissions/_d3.csv

# assemble the 377-row mix (concatenate both CSVs' rows under one header), validate, submit
.venv/bin/python classical_retrieval.py --data-root $DR validate submissions/<mix>.csv
kaggle competitions submit -c ehl-paris-medical-image-retrieval -f submissions/<mix>.csv -m "msg"
```

For the d3 raw-grid (0.904) variant, drop `--bbox-crop --mind` and use `--grid 44`.

## 9. Repo layout

- `d2_template_retrieval.py` — main pipeline (template norm, `--mind`, `--bbox-crop`,
  `--rerank sinkhorn`, `--no-register`, Hungarian). Caches features in `.d2cache/`.
- `d2_methods.py` — feature extractors & the synthetic-d2 method zoo.
- `synthetic_d2_eval.py` — synthetic-d2 validator (independent rigid+elastic warps on
  labeled d1 pairs → offline method ranking without spending submissions).
- `classical_retrieval.py` — feature cache + submission validator.
- `assignment_rerank.py` — Hungarian re-ranking of classical score matrices.
- `docs/architecture.md` — detailed architecture; `docs/augmentation.md` — aug notes.
- Research / negative-evidence scripts (deeds, SIFT, SynthSeg, BrainIAC, contrastive)
  kept for reference.

Data, caches, venvs, and generated submissions are git-ignored.
