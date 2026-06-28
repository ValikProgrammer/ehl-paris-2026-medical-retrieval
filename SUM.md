# Сводка

## Что было сделано

1. Я сначала проверил структуру проекта и данных.
   Выяснил, что код лежит в `/Users/mikhail.fadin/paris`, а данные в `/Users/mikhail.fadin/paris/ehl-paris-medical-image-retrieval`.
   Подтвердил, что:
   - `dataset1` содержит `train_pairs.csv` с размеченными парами;
   - `dataset2` и `dataset3` не имеют train-разметки;
   - локально изображения лежат как `.nii`, хотя в CSV указаны `.nii.gz`.

2. Я прочитал baseline и условия задачи.
   Подтвердил, что это retrieval целых 3D volume, а не matching отдельных slice.
   Query = `T1 post-contrast`, target = `T2`.
   Метрика Kaggle: средний `MRR` по `dataset1`, `dataset2`, `dataset3`.

3. Я поднял локальное окружение через `uv`.
   Использовал локальные каталоги:
   - `.uv-cache`
   - `.uv-python`

   Поставил:
   - `numpy`
   - `scipy`
   - `scikit-learn`
   - `nibabel`

4. Я написал отдельный экспериментальный пайплайн в `classical_retrieval.py`.
   Он делает:
   - загрузку `.nii` и `.nii.gz`;
   - fallback с `.nii.gz` на `.nii`;
   - нормализацию интенсивностей;
   - построение foreground mask;
   - crop по области мозга;
   - извлечение hand-crafted признаков;
   - кэширование признаков;
   - CV на `dataset1/train_pairs.csv`;
   - генерацию submission CSV;
   - валидацию submission.

5. Я реализовал несколько типов признаков:
   - `raw_grid32`
   - `raw_crop32`
   - `edge_grid32`
   - `edge_crop32`
   - `mask_crop32`
   - `proj64`
   - `mask_proj64`
   - `meta12`
   - `shape_moments`
   - `pca_abs_mask24`
   - `pca_abs_raw24`

6. Я реализовал несколько стратегий скоринга:
   - cosine similarity по отдельным признакам;
   - fusion нескольких признаков;
   - `pca_ridge`, где:
     - query и target признаки сжимаются через PCA;
     - Ridge учится переводить query-признаки в пространство target;
     - потом идет ranking по cosine similarity.

7. Я прогнал CV на `dataset1`.
   Лучший источник-сигнал локально:
   - `pca_ridge_c128_a100`

   Его локальный результат:
   - `MRR ≈ 0.865`
   - `top1 ≈ 0.843`

   Сильные classical варианты:
   - `mask_crop32`
   - `fusion_shape`
   - `fusion_grid`

8. Я закэшировал признаки для всех 1454 volume.
   Кэш лежит в `.classical_cache/`.

9. Я сгенерировал и провалидировал несколько submission-файлов.

   Полные:
   - `submissions/all_pca_ridge_c128_a100.csv`
   - `submissions/all_fusion_default_plus_pca_c128_a100.csv`

   Для `dataset2 + dataset3`:
   - `submissions/ds23_pca_ridge_c128_a100.csv`
   - `submissions/ds23_mask_crop32.csv`
   - `submissions/ds23_fusion_shape.csv`
   - `submissions/ds23_fusion_grid.csv`
   - `submissions/ds23_fusion_robust_shape.csv`
   - `submissions/ds23_fusion_default_plus_pca_c128_a100.csv`

10. После первого Kaggle-результата `0.55714` я перестроил стратегию.
    Вывод был такой:
    - нельзя оптимизировать один общий метод на все 3 датасета;
    - `dataset2` требует большей инвариантности к деформациям;
    - `dataset3` требует устойчивости к послеоперационным изменениям.

11. Я добавил dataset-specific методы.

   Для `dataset2`:
   - `fusion_dataset2`
   - `pca_abs_mask24`
   - `fusion_grid`
   - `mask_crop32`

   Для `dataset3`:
   - `fusion_dataset3`
   - `fusion_shape`
   - `pca_ridge`

12. Я сделал смешанные full submissions:
   - `submissions/mix_d1pca_d2dataset2_d3dataset3.csv`
   - `submissions/mix_d1pca_d2grid_d3dataset3.csv`
   - `submissions/mix_d1pca_d2pcaabsmask_d3dataset3.csv`
   - `submissions/mix_d1pca_d2mask_d3shape.csv`

13. Я сделал диагностические partial submissions для Kaggle:

   Dataset 2:
   - `submissions/d2_fusion_dataset2.csv`
   - `submissions/d2_fusion_grid.csv`
   - `submissions/d2_mask_crop32.csv`
   - `submissions/d2_pca_abs_mask24.csv`

   Dataset 3:
   - `submissions/d3_fusion_dataset3.csv`
   - `submissions/d3_fusion_shape.csv`
   - `submissions/d3_pca_ridge_c128_a100.csv`

14. Я добавил валидацию submission-файлов.
   Скрипт проверяет:
   - число строк;
   - длину ranking;
   - совпадение target set с нужной gallery;
   - отсутствие дублей.

15. Я добавил план отправок в `SUBMISSION_PLAN.md`.
   Там описано:
   - какие partial submissions отправлять первыми;
   - как интерпретировать score;
   - какие full mixed submissions пробовать после диагностики.

16. Я обновил `.gitignore` для runtime-артефактов:
   - `.classical_cache/`
   - `.uv-cache/`
   - `.uv-python/`
   - `__pycache__/`
   - `.ssh_known_hosts`

## Итог

В результате получился не один baseline, а полноценный локальный retrieval framework:
- обучение на `dataset1/train_pairs.csv`;
- генерация нескольких retrieval-стратегий;
- отдельные методы для `dataset2` и `dataset3`;
- mixed submission-файлы;
- валидация submission;
- диагностические partial submissions для Kaggle.

## Что осталось

Текущий проверенный best на Kaggle: `0.62522`.

Лучший файл:

```text
submissions/mix_hung_d1pca_d2rankfusion_canon20_grid_d3pca.csv
```

Он состоит из:
- `dataset1`: `pca_ridge_c128_a100` + Hungarian assignment;
- `dataset2`: rank-fusion canonical PCA-axis `size=20` + старый `pca/grid` blend, затем Hungarian assignment;
- `dataset3`: `pca_ridge_c128_a100` + Hungarian assignment.

Важные подтвержденные результаты:
- `all_pca_ridge_c128_a100.csv`: `0.55714`;
- `all_pca_ridge_c128_a100_hungarian.csv`: `0.59436`;
- `mix_hung_d1pca_d2canonpca20_d3pca.csv`: `0.62491`;
- `mix_hung_d1pca_d2rankfusion_canon20_grid_d3pca.csv`: `0.62522`.

Самый важный вывод: главный bottleneck остается `dataset2`.

Partial scores:
- `d1_pca_ridge_c128_a100.csv`: displayed `0.29244`, то есть d1 MRR примерно `0.87732`;
- `d3_pca_ridge_c128_a100_hungarian.csv`: displayed `0.23016`, то есть d3 MRR примерно `0.69048`;
- `d2_rankfusion_canon20_080_pca075grid025_020_hungarian.csv`: displayed `0.10414`, то есть d2 MRR примерно `0.31242`.

Что дополнительно пробовалось и не стало best:
- BrainIAC cosine: `0.15121`;
- BrainIAC adapter training: слабые holdout/all-gallery metrics;
- BrainIAC patch-token matching для d2: хуже classical/canonical;
- scratch 3D contrastive model на GPU: не стал конкурентным;
- leakage diagnostics по order/header/file-size/sample submission: не помогли;
- MIND-like descriptors и lesion-focused descriptors: ниже canonical baseline;
- `d2_canonical_pca24_c128_hungarian.csv`: displayed `0.08255`, хуже `size=20`.

Следующие разумные шаги:
- добавить кэширование в `canonical_pca_retrieval.py`, чтобы быстро перебирать canonical features;
- развивать `dataset2` canonical/rank-fusion направление вокруг `size=20`;
- не тратить время на общий BrainIAC cosine/scratch 3D без новой сильной гипотезы;
- держать `ONBOARD.md` как основной handoff для нового ML developer/agent.
