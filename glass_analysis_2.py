# -*- coding: utf-8 -*-
"""# Классификация типов стекла по результатам анализа данных

**Цель:** выбрать признаки и числовые условия для каждого типа стекла и собрать из них понятный классификатор.

**Порядок:** качество данных → распределения → ручные пороги → сравнение моделей.

Источник данных — локальный файл `31_glass.csv`.

## 1. Библиотеки и настройки
"""

from pathlib import Path

import matplotlib

if "__file__" in globals():
    matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


FEATURES = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]
GLASS_TYPES = [1, 2, 3, 5, 6, 7]

TYPE_NAMES = {
    1: "Строительные окна, float",
    2: "Строительные окна, не float",
    3: "Автомобильные окна, float",
    5: "Тара",
    6: "Посуда",
    7: "Фары",
}

TYPE_COLORS = {
    1: "#147D92",
    2: "#3454A1",
    3: "#AE7F20",
    5: "#C95845",
    6: "#8154A2",
    7: "#2D926B",
}

sns.set_theme(
    style="whitegrid",
    font="DejaVu Sans",
    rc={"grid.alpha": 0.2, "axes.spines.top": False},
)

pd.set_option("display.max_columns", 12)
pd.set_option("display.width", 140)

BASE_DIRECTORY = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUTPUT_DIRECTORY = BASE_DIRECTORY / "glass_rules_outputs"
FIGURE_DIRECTORY = OUTPUT_DIRECTORY / "figures"
FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)


def show_figure(figure, filename):
    '''Показать график и сохранить его для отдельного использования.'''
    figure.tight_layout()
    figure.savefig(
        FIGURE_DIRECTORY / filename,
        dpi=160,
        bbox_inches="tight",
    )
    if "__file__" not in globals():
        plt.show()
    plt.close(figure)


def show_table(table):
    '''В Jupyter показать таблицу, в обычном Python вывести текст.'''
    try:
        from IPython.display import display
    except ImportError:
        print(table.to_string())
    else:
        display(table)

"""## 2. Загрузка данных"""

source_path = BASE_DIRECTORY / "31_glass.csv"
data = pd.read_csv(source_path)
print("Источник:", source_path.name)
print("Размер таблицы:", data.shape)
show_table(data.head())

"""**Обозначения:** RI — показатель преломления, остальные признаки — массовые доли соответствующих оксидов в процентах, type — категория, id — номер образца.

## 3. Качество данных
"""

duplicate_columns = FEATURES + ["type"]
duplicate_mask = data.duplicated(
    subset=duplicate_columns,
    keep=False,
)

quality_summary = pd.DataFrame({
    "Проверка": ["Наблюдения", "Пропуски", "Дополнительные повторы"],
    "Количество": [
        len(data),
        int(data.isna().sum().sum()),
        int(data.duplicated(subset=duplicate_columns).sum()),
    ],
})

show_table(quality_summary)
show_table(data.loc[duplicate_mask])

# Тип и идентификатор не включаем в числовые признаки.
feature_data = data[FEATURES]
target = data["type"]

"""**Вывод:** пропусков нет, строки 39 и 40 совпадают по измерениям, поэтому сохраняем исходную выборку и исключаем id из правил.

## 4. Баланс типов
"""

class_counts = target.value_counts().reindex(GLASS_TYPES)

class_summary = pd.DataFrame({
    "Тип": GLASS_TYPES,
    "Название": [TYPE_NAMES[glass_type] for glass_type in GLASS_TYPES],
    "Количество": class_counts.to_numpy(),
    "Доля, %": (100 * class_counts / len(data)).round(1).to_numpy(),
})

show_table(class_summary)

figure, axis = plt.subplots(figsize=(10, 4))
axis.bar(
    class_counts.index.astype(str),
    class_counts,
    color=[TYPE_COLORS[glass_type] for glass_type in GLASS_TYPES],
)
axis.set_title("Количество образцов каждого типа")
axis.set_xlabel("Тип стекла")
axis.set_ylabel("Число образцов")

show_figure(figure, "01_class_balance.png")

"""**Вывод:** типы 1 и 2 занимают 68,2% выборки, поэтому качество классификации нужно смотреть отдельно для каждого типа.

## 5. Общие распределения признаков
"""

figure, axes = plt.subplots(3, 3, figsize=(12, 9))

for axis, feature in zip(axes.flat, FEATURES):
    values = data[feature]

    axis.hist(values, bins="fd", color="#147D92", edgecolor="white")
    axis.axvline(values.median(), color="#C95845", linestyle="--")
    axis.set_title(feature)
    axis.set_ylabel("Число образцов")
    axis.set_xlabel("RI" if feature == "RI" else "Массовая доля оксида, %")

show_figure(figure, "02_overall_distributions.png")

"""**Вывод:** распределения Mg, K, Ba и Fe содержат нули и асимметричные хвосты, поэтому далее сравниваем типы по отдельности.

## 6. Гистограммы каждого признака по типам
"""

def plot_distributions_by_type(selected_features, filename):
    '''Сравнить типы при одинаковых интервалах и нормировке внутри типа.'''
    figure, axes = plt.subplots(6, 3, figsize=(12, 13))

    for column_number, feature in enumerate(selected_features):
        bin_edges = np.histogram_bin_edges(data[feature], bins=16)

        for row_number, glass_type in enumerate(GLASS_TYPES):
            axis = axes[row_number, column_number]
            values = data.loc[target == glass_type, feature]
            weights = np.full(len(values), 100 / len(values))

            axis.hist(
                values,
                bins=bin_edges,
                weights=weights,
                color=TYPE_COLORS[glass_type],
                edgecolor="white",
            )
            axis.axvline(values.median(), color="black", linestyle="--")
            axis.set_xlim(bin_edges[0], bin_edges[-1])
            axis.set_ylim(0, 100)

            if row_number == 0:
                axis.set_title(feature)

            if column_number == 0:
                axis.set_ylabel(f"Тип {glass_type}, n={len(values)}\nДоля, %")

    show_figure(figure, filename)

plot_distributions_by_type(
    selected_features=["RI", "Na", "Mg"],
    filename="03_distributions_ri_na_mg.png",
)

"""**Вывод:** Mg отделяет многие образцы типов 5–7 от оконных типов, а высокий Na характерен для типов 6 и 7."""

plot_distributions_by_type(
    selected_features=["Al", "Si", "K"],
    filename="04_distributions_al_si_k.png",
)

"""**Вывод:** Al помогает разделять оконные типы, повышен у типов 5 и 7, а K равен нулю у всех девяти образцов типа 6."""

plot_distributions_by_type(
    selected_features=["Ca", "Ba", "Fe"],
    filename="05_distributions_ca_ba_fe.png",
)

"""**Вывод:** Ba выделяет тип 7, повышенный Ca характерен для многих образцов типа 5, а Fe слабо разделяет классы.

## 7. Перекрытие диапазонов
"""

figure, axes = plt.subplots(3, 3, figsize=(12, 10))

for axis, feature in zip(axes.flat, FEATURES):
    sns.boxplot(
        data=data,
        x="type",
        y=feature,
        order=GLASS_TYPES,
        hue="type",
        palette=TYPE_COLORS,
        legend=False,
        showfliers=True,
        ax=axis,
    )
    axis.set_title(feature)
    axis.set_xlabel("Тип стекла")
    axis.set_ylabel(feature)

show_figure(figure, "06_boxplots.png")

"""**Вывод:** типы 1–3 сильно перекрываются, поэтому для них понадобятся сочетания признаков и возможность отказа от классификации.

## 8. Какие признаки выбрать
"""

def calculate_between_class_share(feature):
    '''Доля вариации признака, связанная с разницей средних типов.'''
    overall_mean = data[feature].mean()
    total_variation = ((data[feature] - overall_mean) ** 2).sum()

    between_class_variation = 0.0

    for glass_type in GLASS_TYPES:
        class_values = data.loc[target == glass_type, feature]
        mean_difference = class_values.mean() - overall_mean
        between_class_variation += len(class_values) * mean_difference ** 2

    return between_class_variation / total_variation


feature_scores = pd.DataFrame({
    "Признак": FEATURES,
    "Доля межтиповой вариации": [
        calculate_between_class_share(feature)
        for feature in FEATURES
    ],
})
feature_scores = feature_scores.sort_values(
    "Доля межтиповой вариации",
    ascending=False,
)

show_table(feature_scores.round(3))

figure, axis = plt.subplots(figsize=(9, 4))
axis.barh(
    feature_scores["Признак"][::-1],
    feature_scores["Доля межтиповой вариации"][::-1],
    color="#147D92",
)
axis.set_xlim(0, 1)
axis.set_title("Различия между средними значениями типов")
axis.set_xlabel("Доля общей вариации")

show_figure(figure, "07_feature_selection.png")

"""**Вывод:** основные признаки для правил — Mg, Ba, Al и Na, а K, Ca и RI используем для уточнения отдельных классов.

## 9. Корреляции признаков
"""

pearson_correlations = feature_data.corr(method="pearson")
spearman_correlations = feature_data.corr(method="spearman")

def plot_correlation_matrix(correlation_matrix, title, filename):
    figure, axis = plt.subplots(figsize=(9, 7))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        vmin=-1,
        vmax=1,
        center=0,
        cmap="vlag",
        square=True,
        ax=axis,
    )
    axis.set_title(title)

    show_figure(figure, filename)

plot_correlation_matrix(
    correlation_matrix=pearson_correlations,
    title="Линейные корреляции Pearson",
    filename="08_pearson.png",
)

"""**Вывод:** Ca и RI сильно связаны (r=0,81), но их связь сама по себе не означает хорошего разделения типов."""

plot_correlation_matrix(
    correlation_matrix=spearman_correlations,
    title="Ранговые корреляции Spearman",
    filename="09_spearman.png",
)

"""**Вывод:** ранговая связь Ca–RI также положительна (ρ=0,70), однако для классификации важнее различия между типами.

## 10. Числовые ориентиры для порогов
"""

selected_features = ["Mg", "Al", "Na", "Ba", "K", "Ca", "RI"]

median_profiles = (
    data.groupby("type")[selected_features]
    .median()
    .reindex(GLASS_TYPES)
)

quartile_profiles = (
    data.groupby("type")[selected_features]
    .quantile([0.25, 0.75])
)

show_table(median_profiles.round(5).reset_index())
show_table(quartile_profiles.round(5).reset_index())

"""## 11. Визуальное обоснование правил"""

def plot_rule_projection(
    x_feature,
    y_feature,
    highlighted_type,
    vertical_thresholds,
    horizontal_thresholds,
    title,
    filename,
):
    '''Показать два признака правила без скрытой фильтрации наблюдений.'''
    figure, axis = plt.subplots(figsize=(9, 5))

    for glass_type in GLASS_TYPES:
        class_data = data.loc[target == glass_type]
        is_highlighted = glass_type == highlighted_type

        axis.scatter(
            class_data[x_feature],
            class_data[y_feature],
            label=f"Тип {glass_type}",
            color=TYPE_COLORS[glass_type],
            alpha=0.9 if is_highlighted else 0.35,
            s=65 if is_highlighted else 25,
            edgecolors="black" if is_highlighted else "none",
            linewidths=0.5,
        )

    for threshold in vertical_thresholds:
        axis.axvline(threshold, color="black", linestyle="--")

    for threshold in horizontal_thresholds:
        axis.axhline(threshold, color="black", linestyle="--")

    axis.set_title(title)
    axis.set_xlabel(x_feature)
    axis.set_ylabel(y_feature)
    axis.legend(ncol=3, fontsize=9)

    show_figure(figure, filename)

plot_rule_projection(
    x_feature="Ba",
    y_feature="Al",
    highlighted_type=7,
    vertical_thresholds=[0.25],
    horizontal_thresholds=[1.65],
    title="Тип 7: совместно повышенные Ba и Al",
    filename="10_type_7.png",
)

"""**Вывод:** для типа 7 используем сочетание Ba > 0,25 и Al > 1,65."""

plot_rule_projection(
    x_feature="Na",
    y_feature="K",
    highlighted_type=6,
    vertical_thresholds=[13.8],
    horizontal_thresholds=[0.05],
    title="Тип 6: высокий Na и почти нулевой K",
    filename="11_type_6.png",
)

"""**Вывод:** для типа 6 проверяем Na > 13,8 и K ≤ 0,05, дополнительно ограничивая Mg < 2,8 и Ba ≤ 0,25."""

plot_rule_projection(
    x_feature="Mg",
    y_feature="Al",
    highlighted_type=5,
    vertical_thresholds=[2.0],
    horizontal_thresholds=[1.5],
    title="Тип 5: низкий Mg и повышенный Al",
    filename="12_type_5.png",
)

"""**Вывод:** для типа 5 используем Mg < 2 и Al > 1,5, а условия K > 0,1 и Na < 13,7 уменьшают смешение с типами 6–7."""

plot_rule_projection(
    x_feature="Mg",
    y_feature="Al",
    highlighted_type=2,
    vertical_thresholds=[2.8],
    horizontal_thresholds=[1.4],
    title="Тип 2: высокий Mg и Al выше 1,4",
    filename="13_type_2.png",
)

"""**Вывод:** для части типа 2 проверяем Mg ≥ 2,8, Al > 1,4 и Ca < 8,4, сохраняя возможность пропуска других составов этого типа."""

plot_rule_projection(
    x_feature="Mg",
    y_feature="Al",
    highlighted_type=1,
    vertical_thresholds=[3.0],
    horizontal_thresholds=[1.4],
    title="Тип 1: высокий Mg и Al не выше 1,4",
    filename="14_type_1.png",
)

"""**Вывод:** для части типа 1 используем Mg ≥ 3, Al ≤ 1,4 и Ca ≥ 8,4, но эти условия также захватывают некоторые образцы типов 2–3."""

plot_rule_projection(
    x_feature="Na",
    y_feature="RI",
    highlighted_type=3,
    vertical_thresholds=[13.4],
    horizontal_thresholds=[1.518],
    title="Тип 3: кандидатная область по Na и RI",
    filename="15_type_3.png",
)

"""**Вывод:** для узкой подгруппы типа 3 проверяем Na > 13,4, RI < 1,518, Ca > 8,6 и Mg > 3, поскольку широкие интервалы пересекаются с типами 1–2.

## 12. Условия для каждого типа
"""

def build_rule_masks(samples):
    '''Каждое правило возвращает логическую маску своих кандидатов.'''
    return {
        7: (
            (samples["Ba"] > 0.25)
            & (samples["Al"] > 1.65)
        ),
        6: (
            (samples["K"] <= 0.05)
            & (samples["Na"] > 13.8)
            & (samples["Mg"] < 2.8)
            & (samples["Ba"] <= 0.25)
        ),
        5: (
            (samples["Mg"] < 2.0)
            & (samples["Al"] > 1.5)
            & (samples["K"] > 0.1)
            & (samples["Na"] < 13.7)
        ),
        3: (
            (samples["Na"] > 13.4)
            & (samples["RI"] < 1.518)
            & (samples["Ca"] > 8.6)
            & (samples["Mg"] > 3.0)
        ),
        2: (
            (samples["Al"] > 1.4)
            & (samples["Mg"] >= 2.8)
            & (samples["Ca"] < 8.4)
        ),
        1: (
            (samples["Mg"] >= 3.0)
            & (samples["Al"] <= 1.4)
            & (samples["Ca"] >= 8.4)
        ),
    }


RULE_DESCRIPTIONS = {
    1: "Mg ≥ 3; Al ≤ 1.4; Ca ≥ 8.4",
    2: "Al > 1.4; Mg ≥ 2.8; Ca < 8.4",
    3: "Na > 13.4; RI < 1.518; Ca > 8.6; Mg > 3",
    5: "Mg < 2; Al > 1.5; K > 0.1; Na < 13.7",
    6: "K ≤ 0.05; Na > 13.8; Mg < 2.8; Ba ≤ 0.25",
    7: "Ba > 0.25; Al > 1.65",
}

rule_masks = build_rule_masks(data)

"""## 13. Проверка отдельных правил"""

def evaluate_individual_rules(samples, masks):
    rows = []

    for glass_type in GLASS_TYPES:
        matches = masks[glass_type]
        actual_class = samples["type"] == glass_type

        true_positives = int((matches & actual_class).sum())
        false_positives = int((matches & ~actual_class).sum())
        matched_count = int(matches.sum())
        class_size = int(actual_class.sum())

        precision = true_positives / matched_count if matched_count else np.nan
        recall = true_positives / class_size if class_size else np.nan

        rows.append({
            "Тип": glass_type,
            "Условия одновременно": RULE_DESCRIPTIONS[glass_type],
            "Срабатывания": matched_count,
            "Верно": true_positives,
            "Ошибки": false_positives,
            "Точность, %": round(100 * precision, 1),
            "Полнота, %": round(100 * recall, 1),
        })

    return pd.DataFrame(rows)


individual_rule_scores = evaluate_individual_rules(data, rule_masks)
show_table(individual_rule_scores)

"""**Вывод:** результаты описывают ту же выборку, на которой выбраны пороги, поэтому даже 100% точности при четырёх или восьми срабатываниях не подтверждают качество на новых данных.

## 14. Как объединить правила в классификатор

Правила применяются в порядке 7 → 6 → 5 → 3 → 2 → 1, а при отсутствии совпадений классификатор возвращает «не определён».
"""

RULE_PRIORITY = [7, 6, 5, 3, 2, 1]


def classify_glass(samples):
    '''Применить первое подходящее правило, сохранив отказы и конфликты.'''
    masks = build_rule_masks(samples)

    predictions = pd.Series(
        0,
        index=samples.index,
        name="predicted_type",
        dtype=int,
    )

    for glass_type in RULE_PRIORITY:
        not_classified = predictions == 0
        matching_rows = not_classified & masks[glass_type]
        predictions.loc[matching_rows] = glass_type

    match_counts = pd.DataFrame(masks).sum(axis=1)

    return pd.DataFrame({
        "predicted_type": predictions,
        "matching_rules": match_counts,
    })


classification = classify_glass(data)

classification_details = data[["id", "type"]].join(classification)
classification_details["correct"] = (
    classification_details["type"]
    == classification_details["predicted_type"]
)

show_table(classification_details.head(12))
print("Образцов с несколькими правилами:", int((classification.matching_rules > 1).sum()))

"""**Вывод:** приоритет позволяет узкому правилу типа 3 сработать раньше широких оконных правил, а образцы вне всех областей остаются неопределёнными.

## 15. Описательная оценка ручных правил на исходной выборке
"""

predictions = classification["predicted_type"]
classified_mask = predictions != 0
correct_mask = predictions == target

classified_count = int(classified_mask.sum())
correct_count = int(correct_mask.sum())
unknown_count = int((~classified_mask).sum())
error_count = classified_count - correct_count

overall_scores = pd.DataFrame({
    "Показатель": [
        "Всего образцов",
        "Выдан тип",
        "Верно определён тип",
        "Ошибочно определён тип",
        "Тип не определён",
        "Покрытие правилами, %",
        "Точность среди классифицированных, %",
        "Доля верных ответов среди всех, %",
    ],
    "Значение": [
        len(data),
        classified_count,
        correct_count,
        error_count,
        unknown_count,
        round(100 * classified_count / len(data), 1),
        round(100 * correct_count / classified_count, 1),
        round(100 * correct_count / len(data), 1),
    ],
})

show_table(overall_scores)

final_class_scores = []

for glass_type in GLASS_TYPES:
    predicted_class = predictions == glass_type
    actual_class = target == glass_type

    true_positives = int((predicted_class & actual_class).sum())
    predicted_count = int(predicted_class.sum())
    class_size = int(actual_class.sum())

    precision = true_positives / predicted_count if predicted_count else np.nan
    recall = true_positives / class_size if class_size else np.nan

    final_class_scores.append({
        "Тип": glass_type,
        "Всего в типе": class_size,
        "Предсказан тип": predicted_count,
        "Верно": true_positives,
        "Точность, %": round(100 * precision, 1),
        "Полнота, %": round(100 * recall, 1),
    })

final_class_scores = pd.DataFrame(final_class_scores)
show_table(final_class_scores)

confusion_table = pd.crosstab(
    target,
    predictions,
    rownames=["Истинный тип"],
    colnames=["Предсказанный тип"],
)
confusion_table = confusion_table.reindex(
    index=GLASS_TYPES,
    columns=GLASS_TYPES + [0],
    fill_value=0,
)
confusion_table = confusion_table.rename(columns={0: "Не определён"})

figure, axis = plt.subplots(figsize=(10, 5))
sns.heatmap(
    confusion_table,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    ax=axis,
)
axis.set_title("Результат последовательного применения правил")

show_figure(figure, "16_rule_confusion.png")

"""**Вывод:** на исходной выборке правила дают 130 верных ответов, 30 ошибок и 54 отказа. Доля верных ответов по всем образцам — 60,7%, среди покрытых правилами — 81,2%. Эти оценки описывают данные, по которым выбраны пороги.

## 16. Какие образцы правила не разделяют
"""

problem_rows = classification_details.loc[
    ~classification_details["correct"]
]

problem_details = problem_rows.merge(
    data[["id", "Na", "Mg", "Al", "Ca", "Ba", "K", "RI"]],
    on="id",
    how="left",
)

show_table(problem_details.head(15))

"""**Вывод:** главная трудность — разделение оконных типов 1–3, для которых приведённые пороги описывают только часть класса.

**Вывод:** образец с Ba=0,8 и Al=2,1 попадает под правило типа 7.

## 17. Итог: признаки и значения для каждого типа
"""

final_rule_table = individual_rule_scores[[
    "Тип",
    "Условия одновременно",
    "Верно",
    "Ошибки",
    "Точность, %",
    "Полнота, %",
]].copy()

show_table(final_rule_table)

individual_rule_scores.to_csv(
    OUTPUT_DIRECTORY / "individual_rules.csv",
    index=False,
)
final_class_scores.to_csv(
    OUTPUT_DIRECTORY / "classifier_by_type.csv",
    index=False,
)
classification_details.to_csv(
    OUTPUT_DIRECTORY / "classification_results.csv",
    index=False,
)


"""## 18. Независимая оценка деревьев и случайных лесов

Пороги ручных правил выбраны после просмотра всего набора. Их оценки выше
характеризуют покрытие и ошибки на исходных данных. Для независимой проверки
этих правил понадобится новая выборка.

Для моделей делаем пять стратифицированных внешних фолдов. Каждый уникальный
образец проверяется один раз. Повтор с тем же составом и типом удаляем только
для оценки моделей. Одинаковые составы группируем, чтобы они не попали в обе
части разбиения. Подбор глубины, размера листа и весов классов выполняется
на четырёх внутренних фолдах по Macro F1.

Семь признаков предложены после общего анализа данных, поэтому их сравнение
с девятью носит исследовательский характер. Основная оценка использует все
девять измерений, заданные структурой CSV.
"""

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree


model_data = data.drop_duplicates(subset=FEATURES + ["type"]).reset_index(drop=True)
features = model_data[FEATURES]
model_target = model_data["type"]
sample_groups = pd.util.hash_pandas_object(features, index=False)
selected_features = ["RI", "Na", "Mg", "Al", "K", "Ca", "Ba"]
# После удаления дубля в этом CSV все составы уникальны. Обычная стратификация
# распределяет редкие типы равномернее. Группировка нужна при совпадении составов.
if sample_groups.is_unique:
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    cv_groups = None
else:
    outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    cv_groups = sample_groups
outer_splits = list(outer_cv.split(features, model_target, groups=cv_groups))

tree = DecisionTreeClassifier(random_state=42)
forest = RandomForestClassifier(n_estimators=120, random_state=42)
fixed_tree = DecisionTreeClassifier(
    max_depth=4, min_samples_leaf=3, class_weight="balanced", random_state=42
)
tree_grid = {
    "max_depth": [3, 5, None],
    "min_samples_leaf": [1, 3, 5],
    "class_weight": [None, "balanced"],
}
forest_grid = {
    "max_depth": [5, None],
    "min_samples_leaf": [1, 3],
    "class_weight": [None, "balanced"],
}
experiments = {
    "Частый класс": (DummyClassifier(strategy="most_frequent"), {}, FEATURES),
    "Дерево, фикс. параметры": (fixed_tree, {}, FEATURES),
    "Дерево, 9 признаков": (tree, tree_grid, FEATURES),
    "Лес, 9 признаков": (forest, forest_grid, FEATURES),
    "Дерево, 7 признаков": (tree, tree_grid, selected_features),
    "Лес, 7 признаков": (forest, forest_grid, selected_features),
}
# Выбор между деревом и лесом тоже можно сделать внутри каждого внешнего фолда.
selection_name = "Выбор по внутренней CV, 9 признаков"
oof_predictions = pd.DataFrame(
    index=model_data.index, columns=[*experiments, selection_name], dtype=int
)
split_counts = []
fold_results = []
selections = []


def model_metrics(actual, predicted):
    return {
        "Accuracy": accuracy_score(actual, predicted),
        "Balanced accuracy": recall_score(
            actual, predicted, labels=GLASS_TYPES, average="macro", zero_division=0
        ),
        "Macro F1": f1_score(
            actual, predicted, labels=GLASS_TYPES, average="macro", zero_division=0
        ),
    }


for fold, (train_indices, test_indices) in enumerate(outer_splits, start=1):
    train_target = model_target.iloc[train_indices]
    test_target = model_target.iloc[test_indices]
    train_groups = sample_groups.iloc[train_indices]
    assert set(train_groups).isdisjoint(sample_groups.iloc[test_indices])
    assert set(train_target) == set(test_target) == set(GLASS_TYPES)

    split_counts.append({
        "Фолд": fold,
        "Обучение": len(train_indices),
        "Проверка": len(test_indices),
        **{f"Тип {label}": int((test_target == label).sum()) for label in GLASS_TYPES},
    })
    inner_scores = {}
    for name, (estimator, grid, columns) in experiments.items():
        train_features = features.iloc[train_indices][columns]
        test_features = features.iloc[test_indices][columns]
        if grid:
            search = GridSearchCV(
                estimator, grid, scoring="f1_macro", cv=inner_cv, error_score="raise"
            )
            fit_groups = {"groups": train_groups} if cv_groups is not None else {}
            search.fit(train_features, train_target, **fit_groups)
            fitted = search.best_estimator_
            if columns == FEATURES:
                inner_scores[name] = (search.best_score_, search.best_params_)
        else:
            fitted = estimator.fit(train_features, train_target)
        oof_predictions.loc[test_indices, name] = fitted.predict(test_features)

    chosen = max(inner_scores, key=lambda name: inner_scores[name][0])
    oof_predictions.loc[test_indices, selection_name] = oof_predictions.loc[
        test_indices, chosen
    ]
    selections.append({
        "Фолд": fold, "Выбранная модель": chosen, "Параметры": inner_scores[chosen][1]
    })
    for name in oof_predictions:
        predicted = oof_predictions.loc[test_indices, name].astype(int)
        fold_results.append({"Фолд": fold, "Модель": name, **model_metrics(test_target, predicted)})

assert not oof_predictions.isna().any().any()
oof_predictions = oof_predictions.astype(int)
show_table(pd.DataFrame(split_counts))
show_table(pd.DataFrame(selections))

fold_results = pd.DataFrame(fold_results)
comparison = pd.DataFrame([
    {
        "Модель": name,
        **model_metrics(model_target, oof_predictions[name]),
        "Macro F1, SD по фолдам": fold_results.loc[
            fold_results["Модель"] == name, "Macro F1"
        ].std(),
    }
    for name in oof_predictions
]).set_index("Модель")
show_table(comparison.round(3))
comparison.to_csv(OUTPUT_DIRECTORY / "model_comparison.csv", encoding="utf-8-sig")
fold_results.to_csv(OUTPUT_DIRECTORY / "model_folds.csv", index=False, encoding="utf-8-sig")
manual_metrics = pd.DataFrame([model_metrics(target, predictions)])
print("Те же метрики для ручных порогов на исходных данных, с отказами как ошибками")
show_table(manual_metrics.round(3))

"""Macro F1 даёт каждому типу равный вес. Balanced accuracy здесь означает
среднюю полноту по шести типам. SD показывает разброс Macro F1 по фолдам.
Главная таблица считает метрики по объединённым внешним предсказаниям.
"""

figure, axis = plt.subplots(figsize=(10, 5))
comparison[["Accuracy", "Balanced accuracy", "Macro F1"]].plot.barh(ax=axis)
axis.set(xlim=(0, 1), xlabel="Значение метрики", title="Стекло. Оценка на пяти внешних фолдах")
axis.set_ylabel("")
show_figure(figure, "17_model_comparison.png")

selected_oof = oof_predictions[selection_name]
model_confusion = confusion_matrix(model_target, selected_oof, labels=GLASS_TYPES)
figure, axis = plt.subplots(figsize=(8, 5))
sns.heatmap(
    model_confusion, annot=True, fmt="d", cmap="Blues", cbar=False,
    xticklabels=GLASS_TYPES, yticklabels=GLASS_TYPES, ax=axis,
)
axis.set(
    title="Ошибки модели, выбранной внутри каждого фолда",
    xlabel="Предсказанный тип",
    ylabel="Фактический тип",
)
show_figure(figure, "18_model_confusion.png")
print("Полнота по типам для процедуры выбора модели")
show_table(pd.DataFrame({
    "Тип": GLASS_TYPES,
    "Образцов": model_confusion.sum(axis=1),
    "Верно": model_confusion.diagonal(),
    "Полнота": model_confusion.diagonal() / model_confusion.sum(axis=1),
}).round(3))
class_recall = model_confusion.diagonal() / model_confusion.sum(axis=1)
hardest_type = GLASS_TYPES[int(np.argmin(class_recall))]
best_all_features = comparison.loc[["Дерево, 9 признаков", "Лес, 9 признаков"], "Macro F1"]
print(f"Среди двух настроенных моделей на всех признаках выше Macro F1 у варианта "
      f"«{best_all_features.idxmax()}» ({best_all_features.max():.3f}). "
      f"Процедура выбора модели хуже всего распознаёт тип {hardest_type} "
      f"(полнота {class_recall.min():.1%}).")

"""## 19. Пороги дерева и пример классификации

Для визуализации обучаем одно неглубокое дерево на всех доступных уникальных
образцах. Внешние оценки выше рассчитаны отдельно. Пороги на рисунке показывают,
какие условия дерево строит автоматически.
"""

illustration_tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=42)
illustration_tree.fit(features, model_target)
figure, axis = plt.subplots(figsize=(16, 7))
plot_tree(
    illustration_tree, feature_names=FEATURES, class_names=[str(label) for label in GLASS_TYPES],
    filled=True, rounded=True, impurity=False, precision=5, fontsize=9, ax=axis,
)
axis.set_title("Неглубокое дерево. Пороги состава и прогноз типа")
show_figure(figure, "19_tree.png")

manual_sample = {
    "RI": 1.517, "Na": 14.0, "Mg": 0.2, "Al": 2.1, "Si": 72.5,
    "K": 0.05, "Ca": 9.0, "Ba": 0.8, "Fe": 0.0,
}
example = pd.DataFrame([manual_sample], columns=FEATURES)
manual_type = classify_glass(example)["predicted_type"].iat[0]
print(f"Пример ввода. Ручное правило — тип {manual_type or 'не определён'}. "
      f"Иллюстративное дерево — тип {illustration_tree.predict(example)[0]}.")

