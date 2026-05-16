[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — Fraud Detection for Vehicle Insurance Claims

**Студент:** Алексеев Владислав Алексеевич

**Группа:** БИВ232


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуск](#запуск)
4. [Данные](#данные)
5. [Результаты](#результаты)
6. [Отчёт](#отчёт)


## Описание задачи

**Задача:** классификация (fraud detection для страховых заявок).

**Датасет:** `fraud_oracle.csv` (файл проекта, используется из `data/raw/`).

**Целевая метрика:** `PR-AUC` (основная метрика для несбалансированного класса fraud), дополнительные: `Recall`, `F1`, `Balanced Accuracy`.


## Структура репозитория

```
.
├── data
│   ├── processed               # Очищенные и обработанные данные
│   └── raw                     # Исходные файлы
├── models                      # Сохранённые модели 
├── notebooks
│   ├── 01_eda.ipynb            # EDA
│   ├── 02_baseline.ipynb       # Baseline-модель
│   └── 03_experiments.ipynb    # Эксперименты и ablation study
├── presentation                # Презентация для защиты
├── report
│   ├── images                  # Изображения для отчёта
│   └── report.md               # Финальный отчёт
├── src
│   ├── preprocessing.py        # Предобработка данных
│   └── modeling.py             # Обучение и оценка моделей
├── tests
│   └── test.py                 # Тесты пайплайна
├── requirements.txt
└── README.md
```

## Запуск

Локальный запуск проекта:
```bash
# 1. Клонировать репозиторий
git clone https://github.com/hsemlcourse/hseml-group-project-c0ldesty.git
cd hseml-group-project-c0ldesty

# 2. Создать виртуальное окружение
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # Windows
# source .venv/bin/activate   # Linux/macOS

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Проверить качество кода
make lint

# 5. Запустить Jupyter ноутбуки
```

Проверку качества кода можно запустить командой:
```bash
make lint
```
Она выполняет `flake8 src tests --max-line-length=120`.
Если `make` недоступен, можно запустить ту же проверку напрямую:
```bash
flake8 src tests --max-line-length=120
```

Порядок выполнения ноутбуков:

1. `notebooks/01_eda.ipynb` — загрузка, очистка, EDA, feature engineering и сохранение файлов в `data/processed/`.
2. `notebooks/02_baseline.ipynb` — baseline-модели на raw и FE-признаках.
3. `notebooks/03_experiments.ipynb` — расширенные эксперименты, ансамбли, подбор гиперпараметров, уменьшение размерности, настройка threshold и финальная проверка на test.

## Данные
- `data/raw/` — исходные файлы (`fraud_oracle.csv`).
- `data/processed/` — предобработанные файлы, которые создаются после выполнения `01_eda.ipynb`:
  - `fraud_oracle_clean.csv`
  - `fraud_oracle_fe.csv`


## Результаты
Краткие итоговые выводы:

- Baseline-модели показали, что для несбалансированной fraud-задачи accuracy недостаточно: важнее смотреть на `PR-AUC`, `Recall`, `F1` и `Balanced Accuracy`.
- В `03_experiments.ipynb` были проверены ансамбли, tuned-модели после `GridSearchCV`, варианты с `TruncatedSVD` и подбор threshold на validation.
- Лучшим финальным вариантом стал tuned `CatBoost` на FE-признаках с threshold `0.33`. Модель хорошо находит fraud-случаи, но делает это ценой большого числа ложных срабатываний.

| Модель | Accuracy | Balanced Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Примечание |
|--------|----------|-------------------|-----------|--------|----|---------|--------|------------|
| Baseline | `0.6641 / 0.6508` | `0.7353 / 0.7510` | `0.1310 / 0.1320` | `0.8162 / 0.8649` | `0.2257 / 0.2291` | `0.7974 / 0.7884` | `0.1566 / 0.1560` | `LogisticRegression / LinearSVC` |
| Лучшая модель | `0.7101` | `0.8129` | `0.1633` | `0.9297` | `0.2779` | `0.8680` | `0.2647` | tuned `CatBoost`, threshold `0.33`, test |

Финальная confusion matrix на test:
- `TN = 2018`
- `FP = 881`
- `FN = 13`
- `TP = 172`


## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
