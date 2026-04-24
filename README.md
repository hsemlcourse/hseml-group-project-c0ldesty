[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — Предсказание факта мошенничества по страховым заявкам

**Студент:** Алексеев Владислав Алексеевич

**Группа:** БИВ232


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуски](#быстрый-старт)
4. [Данные](#данные)
5. [Результаты](#результаты)
7. [Отчёт](#отчёт)


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
│   └── 03_experiments.ipynb    # Эксперименты cp2
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

# 4. Запустить Jupyter ноутбуки
```

Порядок выполнения ноутбуков:

1. `notebooks/01_eda.ipynb` — загрузка, очистка, EDA, feature engineering и сохранение файлов в `data/processed/`.
2. `notebooks/02_baseline.ipynb` — baseline-модели на raw и FE-признаках.

## Данные
- `data/raw/` — исходные файлы (`fraud_oracle.csv`).
- `data/processed/` — предобработанные файлы, которые создаются после выполнения `01_eda.ipynb`:
  - `fraud_oracle_clean.csv`
  - `fraud_oracle_fe.csv`


## Результаты
Краткие выводы по текущему этапу (cp1):

- Базовые линейные модели с `class_weight='balanced'` показывают высокий `recall` по классу fraud.
- `DecisionTree` и `KNN` в baseline-режиме почти не помечают случаи как fraud и дают более низкий `recall`.
- Feature engineering из `01_eda.ipynb` позволяет сравнить качество на raw и FE признаках в одинаковом baseline-наборе.

| Модель | Accuracy | Balanced Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Примечание |
|--------|----------|-------------------|-----------|--------|----|---------|--------|------------|
| Baseline | ~0.67 | ~0.75 | ~0.14 | ~0.84 | ~0.24 | ~0.80 | ~0.16 | Лучшими baseline-моделями оказались `LogisticRegression` и `LinearSVC` |
| Лучшая модель | — | — | — | — | — | — | — | |


## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md) (промежуточные результаты пока в README.md)
