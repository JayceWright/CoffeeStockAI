"""
CoffeeStock AI — ML-прогноз расхода сырья
==========================================
Скрипт для Jupyter Notebook: загрузка датасета, BOM-разузлование,
прогноз на 7 дней (Prophet), визуализация.

Запуск: python ml/forecast.py
Или: в Jupyter — выполнять ячейки последовательно.
"""

# %% [markdown]
# # CoffeeStock AI — Прогноз расхода сырья
# Загружаем транзакции, переводим в расход ингредиентов, обучаем Prophet.

# %% Импорты
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from prophet import Prophet

warnings.filterwarnings("ignore")

# Настройка matplotlib для красивых графиков
plt.rcParams.update({
    "figure.figsize": (14, 6),
    "figure.dpi": 120,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# %% [markdown]
# ## 1. Загрузка датасета

# %% Загрузка данных из Excel
DATA_PATH = Path(__file__).resolve().parent.parent / "Data" / "Coffee Shop Sales.xlsx"
print(f"📂 Загрузка данных: {DATA_PATH}")

df_raw: pd.DataFrame = pd.read_excel(DATA_PATH)
print(f"✅ Загружено {len(df_raw):,} транзакций, {df_raw['product_type'].nunique()} типов продуктов")
print(f"   Период: {df_raw['transaction_date'].min()} — {df_raw['transaction_date'].max()}")

# %% [markdown]
# ## 2. BOM-словарь (Bill of Materials)
# Переводим каждый `product_type` в расход ингредиентов.

# %% Словарь разузлования: product_type → {ингредиент: расход на 1 шт.}
BOM: dict[str, dict[str, float]] = {
    # ── Кофейные напитки ──────────────────────────────
    "Gourmet brewed coffee":    {"coffee_beans_g": 18, "water_ml": 250, "cup_pcs": 1, "lid_pcs": 1},
    "Premium brewed coffee":    {"coffee_beans_g": 20, "water_ml": 250, "cup_pcs": 1, "lid_pcs": 1},
    "Organic brewed coffee":    {"coffee_beans_g": 18, "water_ml": 250, "cup_pcs": 1, "lid_pcs": 1},
    "Drip coffee":              {"coffee_beans_g": 15, "water_ml": 200, "cup_pcs": 1, "lid_pcs": 1},
    "Barista Espresso":         {"coffee_beans_g": 14, "water_ml": 60,  "cup_pcs": 1, "lid_pcs": 1},

    # ── Горячий шоколад и какао ───────────────────────
    "Hot chocolate":            {"cocoa_g": 25, "milk_ml": 200, "sugar_g": 10, "cup_pcs": 1, "lid_pcs": 1},
    "Drinking Chocolate":       {"cocoa_g": 30, "milk_ml": 220, "sugar_g": 12, "cup_pcs": 1, "lid_pcs": 1},
    "Organic Chocolate":        {"cocoa_g": 28, "milk_ml": 200, "sugar_g": 8,  "cup_pcs": 1, "lid_pcs": 1},

    # ── Чай ───────────────────────────────────────────
    "Brewed Black tea":         {"tea_g": 3, "water_ml": 300, "cup_pcs": 1, "lid_pcs": 1},
    "Brewed Chai tea":          {"tea_g": 3, "milk_ml": 100, "water_ml": 200, "sugar_g": 5, "cup_pcs": 1, "lid_pcs": 1},
    "Brewed Green tea":         {"tea_g": 2.5, "water_ml": 300, "cup_pcs": 1, "lid_pcs": 1},
    "Brewed herbal tea":        {"tea_g": 3, "water_ml": 300, "cup_pcs": 1, "lid_pcs": 1},
    "Black tea":                {"tea_g": 3, "water_ml": 300, "cup_pcs": 1, "lid_pcs": 1},
    "Chai tea":                 {"tea_g": 3, "milk_ml": 100, "water_ml": 200, "sugar_g": 5, "cup_pcs": 1, "lid_pcs": 1},
    "Green tea":                {"tea_g": 2.5, "water_ml": 300, "cup_pcs": 1, "lid_pcs": 1},
    "Herbal tea":               {"tea_g": 3, "water_ml": 300, "cup_pcs": 1, "lid_pcs": 1},

    # ── Сиропы ────────────────────────────────────────
    "Regular syrup":            {"syrup_ml": 30, "cup_pcs": 0},  # добавка, не отдельный стакан
    "Sugar free syrup":         {"syrup_ml": 30, "cup_pcs": 0},

    # ── Выпечка ───────────────────────────────────────
    "Pastry":                   {"pastry_pcs": 1, "napkin_pcs": 1},
    "Scone":                    {"pastry_pcs": 1, "napkin_pcs": 1},
    "Biscotti":                 {"pastry_pcs": 1, "napkin_pcs": 1},

    # ── Зёрна в упаковке (розница) ────────────────────
    "Espresso Beans":           {"beans_pack_pcs": 1},
    "Gourmet Beans":            {"beans_pack_pcs": 1},
    "Green beans":              {"beans_pack_pcs": 1},
    "House blend Beans":        {"beans_pack_pcs": 1},
    "Organic Beans":            {"beans_pack_pcs": 1},
    "Premium Beans":            {"beans_pack_pcs": 1},

    # ── Мерч / прочее ─────────────────────────────────
    "Clothing":                 {"merch_pcs": 1},
    "Housewares":               {"merch_pcs": 1},
}

print(f"📋 BOM-словарь: {len(BOM)} продуктов → ингредиенты")

# %% [markdown]
# ## 3. Расчёт расхода сырья по транзакциям

# %% Конвертация продаж → расход ингредиентов
def explode_bom(df: pd.DataFrame, bom: dict[str, dict[str, float]]) -> pd.DataFrame:
    """
    Для каждой транзакции рассчитывает расход ингредиентов через BOM.
    Учитывает transaction_qty (количество в чеке).
    """
    rows: list[dict] = []
    unmapped: set[str] = set()

    for _, row in df.iterrows():
        product_type: str = row["product_type"]
        qty: int = row["transaction_qty"]
        sale_date = row["transaction_date"]

        if product_type not in bom:
            unmapped.add(product_type)
            continue

        for ingredient, amount in bom[product_type].items():
            rows.append({
                "date": sale_date,
                "ingredient": ingredient,
                "consumption": amount * qty,
            })

    if unmapped:
        print(f"⚠️  Не найдены в BOM: {unmapped}")

    return pd.DataFrame(rows)


df_consumption: pd.DataFrame = explode_bom(df_raw, BOM)
print(f"✅ Рассчитан расход: {len(df_consumption):,} записей")
print(f"   Ингредиенты: {sorted(df_consumption['ingredient'].unique())}")

# %% [markdown]
# ## 4. Группировка расхода по дням

# %% Агрегация: суточный расход каждого ингредиента
df_daily: pd.DataFrame = (
    df_consumption
    .groupby(["date", "ingredient"], as_index=False)["consumption"]
    .sum()
)
df_daily["date"] = pd.to_datetime(df_daily["date"])
print(f"✅ Агрегировано: {len(df_daily):,} записей (дневной расход)")

# %% Покажем пример — стаканчики (cup_pcs) по дням
target_ingredient: str = "cup_pcs"
df_target: pd.DataFrame = (
    df_daily[df_daily["ingredient"] == target_ingredient]
    .sort_values("date")
    .reset_index(drop=True)
)
print(f"\n📊 Целевой ингредиент для прогноза: {target_ingredient}")
print(f"   Записей: {len(df_target)}, среднее в день: {df_target['consumption'].mean():.0f} шт.")

# %% [markdown]
# ## 5. Обучение модели Prophet

# %% Подготовка данных для Prophet (столбцы ds, y)
df_prophet: pd.DataFrame = df_target[["date", "consumption"]].rename(
    columns={"date": "ds", "consumption": "y"}
)

print("🤖 Обучение Prophet...")
model = Prophet(
    yearly_seasonality=False,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05,  # гибкость тренда
    seasonality_mode="multiplicative",
)
model.fit(df_prophet)
print("✅ Модель обучена!")

# %% Прогноз на 7 дней вперёд
FORECAST_DAYS: int = 7
future: pd.DataFrame = model.make_future_dataframe(periods=FORECAST_DAYS, freq="D")
forecast: pd.DataFrame = model.predict(future)

# Разделяем исторические данные и прогноз
last_date = df_prophet["ds"].max()
df_hist_forecast = forecast[forecast["ds"] <= last_date]
df_future_forecast = forecast[forecast["ds"] > last_date]

print(f"🔮 Прогноз на {FORECAST_DAYS} дней:")
print(df_future_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_string(index=False))

# %% [markdown]
# ## 5.5 Train/Test Validation (MAPE)
# Разбиваем датасет: 80% обучение, 20% тестирование.
# Метрика 3.1 из KPI-дерева: Forecast Accuracy (MAPE < 15%).

# %% Train/Test split и расчёт MAPE
TRAIN_RATIO: float = 0.80
train_size: int = int(len(df_prophet) * TRAIN_RATIO)

df_train: pd.DataFrame = df_prophet.iloc[:train_size].copy()
df_test: pd.DataFrame  = df_prophet.iloc[train_size:].copy()

print(f"\n📊 Train/Test split ({TRAIN_RATIO:.0%}/{1-TRAIN_RATIO:.0%}):")
print(f"   Train: {len(df_train)} дней ({df_train['ds'].min().date()} — {df_train['ds'].max().date()})")
print(f"   Test:  {len(df_test)} дней ({df_test['ds'].min().date()} — {df_test['ds'].max().date()})")

# Обучаем на train-выборке
model_val = Prophet(
    yearly_seasonality=False,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05,
    seasonality_mode="multiplicative",
)
model_val.fit(df_train)

# Прогноз на период тестовой выборки
future_val = model_val.make_future_dataframe(periods=len(df_test), freq="D")
forecast_val = model_val.predict(future_val)
pred_test = forecast_val[forecast_val["ds"].isin(df_test["ds"])]["yhat"].values

# MAPE (цель в KPI 3.1: < 15%)
actual_test = df_test["y"].values
mape: float = float((abs(actual_test - pred_test) / actual_test).mean() * 100)

print(f"\n🎯 Точность прогноза (MAPE на test-выборке): {mape:.1f}%")
print(f"   Цель по KPI 3.1: MAPE < 15% — ", end="")
print("✅ Достигнуто!" if mape < 15 else "⚠️ Следует пересмотреть параметры")

# %% [markdown]
# ## 6. Визуализация: история + прогноз

# %% Красивый график для диплома
fig, ax = plt.subplots(figsize=(14, 6))

# --- Исторические данные (синим) ---
ax.plot(
    df_prophet["ds"], df_prophet["y"],
    color="#3B82F6", linewidth=1.2, alpha=0.7,
    label="Исторический расход"
)

# --- Тренд модели (тонким) ---
ax.plot(
    df_hist_forecast["ds"], df_hist_forecast["yhat"],
    color="#6366F1", linewidth=1.0, alpha=0.5,
    linestyle="--", label="Тренд (Prophet)"
)

# --- Прогноз на 7 дней (оранжевым) ---
ax.plot(
    df_future_forecast["ds"], df_future_forecast["yhat"],
    color="#F97316", linewidth=2.5,
    marker="o", markersize=6,
    label=f"Прогноз на {FORECAST_DAYS} дней"
)

# --- Доверительный интервал прогноза ---
ax.fill_between(
    df_future_forecast["ds"],
    df_future_forecast["yhat_lower"],
    df_future_forecast["yhat_upper"],
    color="#F97316", alpha=0.15,
    label="95% доверительный интервал"
)

# --- Вертикальная линия: граница «история / прогноз» ---
ax.axvline(x=last_date, color="#EF4444", linestyle=":", linewidth=1.5, alpha=0.8, label="Сегодня")

# --- Оформление ---
ax.set_title(
    f"CoffeeStock AI — Прогноз расхода: {target_ingredient.replace('_', ' ').title()}",
    fontsize=16, fontweight="bold", pad=15
)
ax.set_xlabel("Дата", fontsize=12)
ax.set_ylabel(f"Расход ({target_ingredient}), шт./день", fontsize=12)
ax.legend(loc="upper left", framealpha=0.9, fontsize=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.xticks(rotation=45)
plt.tight_layout()

# --- Сохранение графика ---
plot_path = Path(__file__).resolve().parent / "forecast_plot.png"
fig.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"\n💾 График сохранён: {plot_path}")
plt.show()

# %% [markdown]
# ## 7. Сводка прогноза по всем ключевым ингредиентам
# Краткая таблица для дашборда

# %% Прогноз по нескольким ингредиентам (для API)
KEY_INGREDIENTS: list[str] = ["cup_pcs", "milk_ml", "coffee_beans_g", "cocoa_g", "tea_g", "pastry_pcs"]

print("\n" + "=" * 65)
print("📋 СВОДКА: средний прогнозируемый расход на следующие 7 дней")
print("=" * 65)

summary_rows: list[dict] = []
for ing_name in KEY_INGREDIENTS:
    df_ing = df_daily[df_daily["ingredient"] == ing_name].sort_values("date").reset_index(drop=True)
    if len(df_ing) < 10:
        continue

    df_p = df_ing[["date", "consumption"]].rename(columns={"date": "ds", "consumption": "y"})
    m = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
    )
    m.fit(df_p)
    fut = m.make_future_dataframe(periods=FORECAST_DAYS, freq="D")
    fc = m.predict(fut)
    fc_future = fc[fc["ds"] > df_p["ds"].max()]
    avg_forecast = fc_future["yhat"].mean()

    summary_rows.append({
        "ingredient": ing_name,
        "avg_daily_actual": df_ing["consumption"].tail(30).mean(),
        "avg_daily_forecast": avg_forecast,
    })
    print(f"  {ing_name:20s} | факт (30д): {df_ing['consumption'].tail(30).mean():>8.0f} | прогноз: {avg_forecast:>8.0f}")

print("=" * 65)
print("✅ Прогноз завершён. Данные готовы для передачи в API.")
