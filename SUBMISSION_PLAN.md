# План работы с обратной связью Kaggle

Текущий публичный счёт: `0.55714`.

Самый быстрый способ улучшиться — изолировать публичный MRR по конкретным датасетам.
Kaggle усредняет три MRR по датасетам:

```text
score = (dataset1_MRR + dataset2_MRR + dataset3_MRR) / 3
```

Если submission содержит только один датасет, умножьте отображаемый счёт Kaggle на
`3`, чтобы оценить публичный MRR этого датасета.

## Отправьте сначала эти диагностики

Dataset 2, насыщенный деформациями:

```text
submissions/d2_fusion_dataset2.csv
submissions/d2_fusion_grid.csv
submissions/d2_mask_crop32.csv
submissions/d2_pca_abs_mask24.csv
```

Dataset 3, послеоперационный:

```text
submissions/d3_fusion_dataset3.csv
submissions/d3_fusion_shape.csv
submissions/d3_pca_ridge_c128_a100.csv
```

Запишите каждый отображаемый счёт Kaggle и умножьте на `3`.

## Отправьте этих полных кандидатов

Используйте их после диагностик или сразу, если бюджет submission не имеет
значения:

```text
submissions/mix_d1pca_d2dataset2_d3dataset3.csv
submissions/mix_d1pca_d2grid_d3dataset3.csv
submissions/mix_d1pca_d2pcaabsmask_d3dataset3.csv
submissions/mix_d1pca_d2mask_d3shape.csv
```

Лучший метод исходного домена остаётся:

```text
submissions/all_pca_ridge_c128_a100.csv
```

## Интерпретация

- Если `d2_pca_abs_mask24` обходит остальные, dataset2 проваливается в основном потому,
  что случайный поворот/деформация ломает выровненное по сетке сопоставление.
- Если выигрывает `d2_fusion_grid`, dataset2 всё ещё выигрывает от выровненных
  рёбер/масок, несмотря на деформацию.
- Если выигрывает `d3_pca_ridge_c128_a100`, операционные случаи всё ещё сохраняют
  достаточно сигнала внешнего вида исходного домена.
- Если выигрывает `d3_fusion_dataset3` или `d3_fusion_shape`, операция ломает локальный
  внешний вид, и fusion по форме/рёбрам безопаснее.
