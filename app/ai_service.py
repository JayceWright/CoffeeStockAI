"""
CoffeeStock AI — AI Service (OpenRouter API)
=============================================
Генерация объяснений аномалий для менеджера через OpenRouter
(бесплатные модели: DeepSeek, Llama 3, Mistral).
OpenRouter совместим с OpenAI SDK.
"""

import os
import traceback
from datetime import datetime

# ── API Keys (из env-переменных, fallback для локальной разработки) ──
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")   # заполни в .env!
OWM_API_KEY = os.getenv("OWM_API_KEY", "34076b265135c684510dd990edad57a3")

# Бесплатная маршрутизация OpenRouter
OPENROUTER_MODELS = [
    "openrouter/free"
]

# Координаты Нью-Йорка (из датасета)
NYC_LAT = 40.74
NYC_LON = -74.04


def get_weather_forecast_text() -> dict:
    """Получает текущую погоду и прогноз из OpenWeatherMap API."""
    import requests

    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/forecast"
            f"?lat={NYC_LAT}&lon={NYC_LON}&appid={OWM_API_KEY}"
            f"&units=metric&lang=ru&cnt=8"
        )
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        forecasts = []
        for item in data.get("list", [])[:8]:
            forecasts.append({
                "dt_txt": item["dt_txt"],
                "temp": item["main"]["temp"],
                "description": item["weather"][0]["description"],
                "rain_mm": item.get("rain", {}).get("3h", 0),
            })

        today = forecasts[0] if forecasts else {}
        tomorrow = forecasts[2] if len(forecasts) > 2 else (forecasts[0] if forecasts else {})

        return {
            "today_temp": today.get("temp", "N/A"),
            "today_desc": today.get("description", "N/A"),
            "tomorrow_temp": tomorrow.get("temp", "N/A"),
            "tomorrow_desc": tomorrow.get("description", "N/A"),
            "city": data.get("city", {}).get("name", "New York"),
            "raw": forecasts,
        }
    except Exception as e:
        print(f"⚠️ Weather API error: {e}. Используем mock-данные.")
        return {
            "today_temp": "22.5",
            "today_desc": "ясно",
            "tomorrow_temp": "24.0",
            "tomorrow_desc": "переменная облачность",
            "city": "New York",
            "raw": [],
        }


def get_upcoming_holidays() -> list[str]:
    """Возвращает ближайшие праздники/события."""
    holidays_2026 = {
        "2026-01-01": "Новый год",
        "2026-01-19": "День М.Л. Кинга",
        "2026-02-14": "День Святого Валентина 💝",
        "2026-02-16": "Presidents' Day",
        "2026-03-17": "День Святого Патрика 🍀",
        "2026-04-05": "Пасха 🐣",
        "2026-05-10": "День Матери 🌷",
        "2026-05-25": "Memorial Day",
        "2026-06-21": "День Отца",
        "2026-07-04": "День Независимости 🎆",
        "2026-09-07": "Labor Day",
        "2026-10-01": "Международный День Кофе ☕",
        "2026-10-31": "Хэллоуин 🎃",
        "2026-11-26": "День Благодарения 🦃",
        "2026-12-25": "Рождество 🎄",
    }

    today = datetime.now().date()
    upcoming = []
    for date_str, name in holidays_2026.items():
        from datetime import datetime as dt
        hdate = dt.strptime(date_str, "%Y-%m-%d").date()
        diff = (hdate - today).days
        if -1 <= diff <= 14:
            if diff == 0:
                upcoming.append(f"🎉 СЕГОДНЯ: {name}")
            elif diff == 1:
                upcoming.append(f"📅 ЗАВТРА: {name}")
            elif diff < 0:
                upcoming.append(f"📅 Вчера: {name}")
            else:
                upcoming.append(f"📅 Через {diff} дн.: {name}")
    return upcoming


async def explain_anomaly(
    ingredient: str,
    current_stock: float,
    min_stock: float,
    forecast_qty: float,
    recommended_qty: float,
    anomaly_reason: str,
    unit: str,
) -> str:
    """
    Генерирует объяснение аномалии через OpenRouter API.
    Использует DeepSeek / Llama / Mistral (бесплатные модели).
    """
    if not OPENROUTER_API_KEY:
        return ("⚠️ OPENROUTER_API_KEY не задан. "
                "Добавь ключ в .env: OPENROUTER_API_KEY=sk-or-...")

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )

        # Контекст: погода + праздники
        weather = get_weather_forecast_text()
        holidays = get_upcoming_holidays()
        holidays_text = "\n".join(holidays) if holidays else "Нет ближайших праздников"

        prompt = f"""Ты — AI-аналитик системы управления запасами кофейни в Нью-Йорке.
Менеджер видит аномалию в рекомендации по заказу и нуждается в объяснении.

КОНТЕКСТ:
- Ингредиент: {ingredient} ({unit})
- Текущий остаток: {current_stock} {unit}
- Минимальный порог: {min_stock} {unit}
- Прогноз Prophet на 7 дней: {forecast_qty} {unit}
- Рекомендация к заказу: {recommended_qty} {unit}
- Причина аномалии: {anomaly_reason}

ПОГОДА ({weather['city']}):
- Сейчас: {weather['today_temp']}°C, {weather['today_desc']}
- Завтра: {weather['tomorrow_temp']}°C, {weather['tomorrow_desc']}

БЛИЖАЙШИЕ СОБЫТИЯ:
{holidays_text}

ЗАДАЧА:
Объясни менеджеру простым языком почему система рекомендует такой объём заказа.
Учти влияние погоды и праздников на спрос. Дай конкретную рекомендацию.
Формат: 2-3 предложения, по-русски, конкретно и по делу."""

        # Пробуем модели по порядку
        last_error = ""
        for model_id in OPENROUTER_MODELS:
            try:
                response = await client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.7,
                    extra_headers={
                        "HTTP-Referer": "https://coffeestockai.railway.app",
                        "X-Title": "CoffeeStock AI",
                    },
                )
                content = response.choices[0].message.content
                if not content:
                    raise ValueError(f"Пустой ответ от модели {model_id}")
                text = content.strip()
                print(f"✅ AI ответ получен от: {model_id}")
                return text
            except Exception as model_err:
                last_error = str(model_err)
                print(f"⚠️ Модель {model_id} недоступна: {model_err}")
                continue

        print(f"⚠️ Все модели недоступны. Последняя ошибка: {last_error[:120]}")
        return ("[Автоматический анализ]: Заказ рассчитан с учетом "
                "прогнозируемой погоды и сезонности. AI-сервер временно "
                "недоступен (режим защиты от сетевых сбоев).")

    except Exception as e:
        traceback.print_exc()
        return ("[Автоматический анализ]: Заказ рассчитан штатно. "
                "Детальное AI-объяснение недоступно из-за таймаута.")
