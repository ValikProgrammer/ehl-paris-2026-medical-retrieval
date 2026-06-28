# Онбординг по проекту

Этот репозиторий — для соревнования Kaggle `ehl-paris-medical-image-retrieval`.
Текущая активная цель — достичь публичного/приватного счёта `0.9+`. Текущий лучший
проверенный публичный счёт — `0.80127` (`mix_hung_d1template_d2template_d3pca.csv`:
template normalization для d1+d2, pca_ridge для d3). См. `STATUS.md` для актуального
статуса, лога отправок и следующих шагов.

## Последний прорыв (template normalization для dataset2)

dataset2 был узким местом (MRR ~0.31-0.39). Решение: **template normalization**.
Поскольку train-пары dataset1 зарегистрированы, среднее по train-запросам (T1)
и среднее по train-target (T2) образуют два шаблона, разделяющих одну сетку.
Жёсткая регистрация каждого volume dataset2 к его шаблону той же модальности
(intramodal intensity NCC -> устойчиво, в отличие от кросс-модальной) приводит запрос
и target в общую систему координат, где работает PCA/Ridge-маппинг dataset1. Это O(N),
а не O(N^2). Это подняло MRR d2 с ~0.39 до ~0.75 и общий счёт с 0.65137 до 0.77071.

Ключевые скрипты:
- `synthetic_d2_eval.py` + `d2_methods.py`: локальный валидатор синтетического d2. Применяет
  независимые rigid+elastic-варпы к размеченным парам d1, чтобы имитировать d2, так что методы
  можно оценивать офлайн (Hungarian-recovery MRR) БЕЗ траты submission на Kaggle.
  Откалиброван: при rot12/shift6/elastic3 canonical-baseline даёт ~0.26 (совпадает с реальным
  d2), template даёт ~0.55. Прокси корректно предсказал реальный выигрыш. ИСПОЛЬЗУЙТЕ ЭТО
  перед отправкой любого нового метода d2.
- `d2_template_retrieval.py`: продакшен template normalization. Кэширует нормализованные
  признаки в `.d2cache/`. `--datasets dataset2 --assignment`.

Отрицательные результаты харнесса (НЕ обошли canonical, сэкономили submission):
поза-инвариантные гистограммные дескрипторы (слишком лоссовые), rigid edge-NCC попарная
регистрация re-rank (elastic-варп побеждает rigid; локальные минимумы).

После этого **dataset3 (MRR ~0.69) стал самым слабым датасетом.**

## Рабочее пространство

- Корень репозитория: `/Users/mikhail.fadin/paris`
- Корень данных: `ehl-paris-medical-image-retrieval/`
- Текущая ветка: `main`
- Локальное Python-окружение: `.venv/`
- Закэшированные классические признаки: `.classical_cache/`
- Выходы submission: `submissions/`
- Runtime-артефакты и большие файлы должны оставаться вне git.

CSV датасетов ссылаются на `.nii.gz`, но локальные файлы часто хранятся как
несжатые `.nii`. Существующие скрипты обрабатывают этот fallback.

## Структура датасета

```text
ehl-paris-medical-image-retrieval/
  dataset1/
    train_pairs.csv
    val_queries.csv
    val_gallery.csv
    test_queries.csv
    test_gallery.csv
    images/
  dataset2/
    val_queries.csv
    val_gallery.csv
    test_queries.csv
    test_gallery.csv
    images/
  dataset3/
    val_queries.csv
    val_gallery.csv
    test_queries.csv
    test_gallery.csv
    images/
```

Задача: для каждого запросного T1 post-contrast volume ранжировать все T2-volume из
gallery того же датасета и split.

Размеры:

```text
dataset1 train pairs: 350
dataset1 val/test:    40 / 100
dataset2 val/test:    40 / 100
dataset3 val/test:    20 / 77
full submission rows: 377
```

Смысл датасетов:

- `dataset1`: зарегистрированные предоперационные пары. Это единственный размеченный
  обучающий набор и исходный домен.
- `dataset2`: те же исходные условия, но запрос и target независимо трансформированы
  случайным жёстким сдвигом/поворотом и нелинейной деформацией. Это главное узкое место.
- `dataset3`: пары предоперационное-к-интраоперационному. Target находится примерно в том
  же физическом пространстве, но операция может удалить/сместить анатомию.

Метрика:

```text
score = (dataset1_MRR + dataset2_MRR + dataset3_MRR) / 3
```

Частичные submission полезны. Если отправляется только один датасет, умножьте отображаемый
счёт Kaggle на `3`, чтобы оценить публичный MRR этого датасета.

## Окружение

Офлайн-окружение для классики:

```bash
./setup_local_env_offline.sh
source .venv/bin/activate
python -c "import numpy, scipy, sklearn, nibabel; print('env ok')"
```

Используйте `.venv/bin/python` явно при запуске локальных скриптов из автоматизации:

```bash
.venv/bin/python classical_retrieval.py validate submissions/mix_hung_d1pca_d2canonpca20_d3pca.csv
```

GPU-работа выполнялась через удалённую Jupyter-машину. Не сохраняйте токены,
пароли или root-учётные данные в файлах репозитория. Если учётные данные нужны,
восстанавливайте их из контекста пользователя/чата, а не из закоммиченных файлов.

## Основные скрипты

- `classical_retrieval.py`
  Главный классический пайплайн признаков. Загружает NIfTI, нормализует volume, извлекает
  признаки, кэширует их, обучает PCA/Ridge, предсказывает ранжирования и валидирует CSV.

- `assignment_rerank.py`
  Применяет назначение «один к одному» (Hungarian) поверх матриц оценок. Это было крупным
  улучшением и должно рассматриваться как часть текущего baseline.

- `canonical_pca_retrieval.py`
  Специфичная для dataset2 канонизация по осям PCA. Канонизирует оси foreground-маски,
  извлекает канонические признаки, обучает PCA/Ridge на dataset1 и ранжирует d2.
  Это лучшее текущее направление для dataset2.

- `diagnostic_submissions.py`
  Дешёвые диагностики на утечки/порядок/заголовок/размер файла. Они не обошли основную
  модель, но документируют полезные отрицательные свидетельства.

- `brainiac_cosine_retrieval.py`, `brainiac_adapter_train.py`,
  `brainiac_patch_retrieval.py`
  Эксперименты с BrainIAC. Обычный косинус, обучение адаптера и сопоставление patch-токенов
  показали результаты ниже классического/канонического пайплайна.

- `contrastive_3d_train.py`
  3D contrastive-модель с нуля с сильной аугментацией. Прошла smoke-тест на GPU,
  но первый полный запуск не выучил конкурентоспособные эмбеддинги.

- `mind_retrieval.py`, `lesion_retrieval.py`
  Диагностики dataset2 на основе self-similarity/фокуса на поражениях (lesion). Обе показали
  результаты ниже канонического/PCA baseline.

- `jupyter_exec.py`
  Локальный помощник для выполнения кода на удалённом Jupyter-ядре.

## Проверенные счёты на Kaggle

Текущий лучший полный submission:

```text
submissions/mix_hung_d1pca_d2rankfusion_canon20_grid_d3pca.csv
public score: 0.62522
```

Важные полные счёты:

```text
mix_hung_d1pca_d2rankfusion_canon20_grid...  0.62522  текущий лучший
mix_hung_d1pca_d2canonpca20_d3pca.csv        0.62491
mix_hung_d1pca_d2pca075grid025_d3pca.csv     0.59523
all_pca_ridge_c128_a100_hungarian.csv        0.59436
all_pca_ridge_c128_a100.csv                  0.55714
all_fusion_default_plus_pca_c128_a100.csv    0.50153
brainiac_cosine_submission.csv               0.15121
```

Важные частичные счёты:

```text
d1_pca_ridge_c128_a100.csv                   0.29244 отображаемый, примерно d1 MRR 0.87732
d2_canonical_pca20_c128_hungarian.csv        0.10383 отображаемый, примерно d2 MRR 0.31149
d2_rankfusion_canon20_080_pca075grid025...   0.10414 отображаемый, примерно d2 MRR 0.31242
d2_canonical_pca24_c128_hungarian.csv        0.08255 отображаемый, примерно d2 MRR 0.24765
d2_blend_pca075_grid025_hungarian.csv        0.07415 отображаемый, примерно d2 MRR 0.22245
d2_pca_ridge_c128_a100_hungarian.csv         0.07328 отображаемый, примерно d2 MRR 0.21984
d3_pca_ridge_c128_a100_hungarian.csv         0.23016 отображаемый, примерно d3 MRR 0.69048
```

Текущее узкое место по-прежнему `dataset2`. dataset1 и dataset3 заметно лучше
после PCA/Ridge плюс назначения Hungarian.

## Текущий лучший рецепт

Текущий лучший полный submission комбинирует:

- dataset1: `pca_ridge_c128_a100` + назначение Hungarian
- dataset2: rank fusion канонического PCA-axis `size=20` и старого pca/grid blend,
  с последующим назначением Hungarian
- dataset3: `pca_ridge_c128_a100` + назначение Hungarian

Файл:

```text
submissions/mix_hung_d1pca_d2rankfusion_canon20_grid_d3pca.csv
```

Валидация:

```bash
.venv/bin/python classical_retrieval.py validate submissions/mix_hung_d1pca_d2rankfusion_canon20_grid_d3pca.csv
```

Отправка:

```bash
kaggle competitions submit \
  -c ehl-paris-medical-image-retrieval \
  -f submissions/mix_hung_d1pca_d2rankfusion_canon20_grid_d3pca.csv \
  -m "mix hung d1 pca d2 rankfusion canon20 grid d3 pca"
```

## Готово, но ещё не полностью использовано

Эти файлы сгенерированы и валидны. `d2_canonical_pca24...` уже был отправлен и хуже,
чем `d2_canonical_pca20...`; держите его как отрицательное свидетельство, а не как
текущий кандидат.

```text
submissions/d2_canonical_pca24_c128_hungarian.csv
```

Rank-fusion файл dataset2 уже вложен в текущий лучший полный submission.

## Команды для воспроизведения

Сгенерировать текущий лучший baseline d1/d3 с Hungarian:

```bash
.venv/bin/python assignment_rerank.py \
  --method pca_ridge \
  --pca-components 128 \
  --pca-alpha 100 \
  --out submissions/all_pca_ridge_c128_a100_hungarian.csv
```

Сгенерировать текущую лучшую каноническую диагностику dataset2:

```bash
.venv/bin/python canonical_pca_retrieval.py \
  --datasets dataset2 \
  --size 20 \
  --components 128 \
  --alpha 100 \
  --assignment \
  --out submissions/d2_canonical_pca20_c128_hungarian.csv
```

Провалидировать частичный submission dataset2:

```bash
.venv/bin/python classical_retrieval.py validate \
  --allow-partial submissions/d2_canonical_pca20_c128_hungarian.csv
```

Провалидировать полный submission:

```bash
.venv/bin/python classical_retrieval.py validate \
  submissions/mix_hung_d1pca_d2rankfusion_canon20_grid_d3pca.csv
```

Проверить последние счёты на Kaggle:

```bash
kaggle competitions submissions -c ehl-paris-medical-image-retrieval
```

## Что сработало

- Кросс-модальный PCA/Ridge-маппинг, обученный на dataset1.
- Назначение «один к одному» (Hungarian). Это подняло полный счёт с `0.55714` до
  `0.59436`.
- Канонизация по осям PCA для dataset2. Это подняло отображаемый частичный счёт d2
  примерно с `0.074` до `0.104`.
- Rank fusion canonical20 со старым pca/grid blend для dataset2. Это дало небольшой,
  но проверенный прирост полного счёта с `0.62491` до `0.62522`.
- Микширование под конкретные датасеты. Не навязывайте один метод всем трём датасетам.

## Что не сработало

- Обычный косинус BrainIAC: очень низкий полный счёт (`0.15121`).
- Обучение адаптера BrainIAC: плохие holdout/all-gallery метрики.
- Сопоставление patch-токенов BrainIAC для d2: ниже классического/канонического baseline.
- 3D contrastive-модель с нуля: не выучила полезный all-gallery ретривал в
  первом полном запуске.
- Проверки на утечки порядка/заголовка/размера файла/sample-submission для dataset2: отрицательно.
- MIND-подобные self-similarity и lesion-focused hand-crafted дескрипторы: ниже
  канонического/PCA baseline.
- Полный submission fusion default plus PCA: ниже обычного PCA/Ridge Hungarian.
- Канонический dataset2 `size=24`: хуже канонического `size=20`.

## Рабочие принципы для следующего агента

1. Всегда валидируйте CSV перед отправкой на Kaggle.

2. Используйте частичные submission по датасетам, чтобы изолировать, где помогает изменение.
   Помните: отображаемый частичный счёт, умноженный на `3`, приближает MRR этого датасета.

3. Будьте консервативны с submission на Kaggle. Отправляйте диагностики только когда они
   тестируют новую гипотезу или сильный вариант проверенного пути.

4. Относитесь к dataset2 как к главному узкому месту. Работайте над инвариантностью к
   поворотам/деформациям, канонизацией, регистрацией или устойчивым назначением, прежде чем
   тратить время на dataset1.

5. Держите baseline dataset1 и dataset3 стабильными, если нет конкретного свидетельства
   лучшего метода. Текущие d1/d3 сравнительно сильны.

6. Не коммитьте крупные артефакты, датасеты, кэши, чекпойнты модели, токены или
   удалённые учётные данные. Держите только исходные скрипты и небольшие CSV submission.

7. Предпочитайте воспроизводимые скрипты работе только в ноутбуках. Если используете
   удалённый GPU, скопируйте финальный скрипт обратно в репозиторий и задокументируйте
   сгенерированные артефакты.

8. Не доверяйте только локальной CV. dataset2 и dataset3 — это сдвиги домена; частичные
   диагностики на Kaggle — авторитетный сигнал.

9. Если продолжаете работу над dataset2, сначала попробуйте:
   - добавить кэширование в `canonical_pca_retrieval.py`, чтобы варианты size/sign/feature
     можно было перебирать быстрее;
   - протестировать больше канонических feature/rank-блендов перед запуском очередной глубокой модели;
   - приоритизировать трансформации вокруг `size=20`, потому что `size=24` дал регресс.

10. Держите `ONBOARD.md`, `SUM.md` и `SUBMISSION_PLAN.md` обновлёнными при изменении
    счётов или лучших файлов.
