
## Leak audit + honest leak-free results (2026-06-28)

Удалён код лика BraTS (`tmp/srv_*`) и зондов (`d1_zip_leak.py`, `d1_leak_hunt.py`).
Лик 0.91981 у Valik = внешний матч к публичному BraTS-GLI; убран.

### Лик в d3 (объяснение)
d3: intraop-таргет переложен в координаты preop-запроса → пара со-регистрирована
и имеет уникальную геометрию (affine/размеры FOV) на пациента. Это даёт d3=1.0
«даром». Два уровня:
- affine-заголовок (точное совпадение, метаданные) — чистый лик.
- ресайз нативной коробки в куб 44³ — переносит FOV-отпечаток + выравнивание в куб
  (мягкий лик). d1 этого лишён (общая сетка у всех) → d1=0.964, не 1.0.

### d3 leak-ladder (партиалы, displayed×3)
- raw-box grid (FOV-лик + со-регистрация): **1.000**
- bbox-crop (убран FOV-отпечаток, со-регистрация осталась): **0.442**
- template-register (убрана и со-регистрация): **0.251**

### Per-dataset val MRR (leak-free, template d1/d2)
- d1 = 0.964, d2 = 0.749 (стена), d3(bbox) = 0.442

### Full leak-free (d3 = bbox-crop)
- WITH Hungarian (бийекция): **0.71836**
- NO Hungarian (сырой косинус): **0.65057**  → эффект венгерского +0.068

### Best honest (если считать d3-grid легитимным контентом, не метаданными)
- d1 template + d2 template + d3 grid + Hungarian = **0.90444**
- Sinkhorn rerank d2 (без обучения): регресс 0.904→0.891, выключен по умолчанию.

Стена d2 ≈0.749 подтверждена ~13 методами; честный потолок без лика BraTS = 0.904
(если d3-grid ок) или 0.718 (строго, d3-bbox).

## d2/d3 improvement attempts (2026-06-28, leak-free regime)

- **d3 MIND (modality-invariant) — WIN.** On aligned d3, MIND beats raw grid:
  d3-bbox grid 0.442 -> d3-bbox MIND **0.656** (partial 0.21859x3).
  Leak-free full (d1 template + d2 template + d3 bbox MIND) = **0.78971** (vs 0.718 grid).
- **d2 MIND-after-register: neutral** (synthetic 0.554 vs template 0.558) -> won't move
  real d2. d2 wall (0.749) is geometry (independent warp), MIND fixes modality not geometry.
- Next d3 levers (room at 0.656): finer grid MIND (d3 aligned, grid56/64 may help unlike
  d2), MIND+grid rank-fusion. d2: exhausted honestly.

Leak-free best so far: **0.78971** (d3 MIND). With d3-grid (FOV-leak) it's 0.904.
