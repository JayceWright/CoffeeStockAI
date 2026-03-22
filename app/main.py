"""
CoffeeStock AI — FastAPI Backend (v2)
======================================
API с реальной БД (SQLite/PostgreSQL), feedback loop,
и полным циклом: POS → ML → заказ → обратная связь.

Запуск: uvicorn app.main:app --reload --port 8000
"""

import os
import random
import hashlib
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Optional

# Загружаем .env для локальной разработки (на Railway переменные задаются в настройках)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv не установлен — игнорируем


from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import func as sa_func

from db.database import engine, get_db, init_db
from db.models import (
    Base, SalesHistory, Ingredient, OrderDraft, FeedbackLog, Forecast,
)

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ── Инициализация Rate Limiting ───────────────────────────────
# Используем IP-адрес клиента для ограничения (get_remote_address)
limiter = Limiter(key_func=get_remote_address)

# ── Инициализация приложения ──────────────────────────────────
app = FastAPI(
    title="CoffeeStock AI",
    description="Интеллектуальная система управления запасами для сети кофеен",
    version="2.0.0",
)

# CORS — разрешаем фронтенду обращаться к API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключаем Rate Limiter к приложению
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── Создаём таблицы и заполняем начальные данные при старте ───
@app.on_event("startup")
def on_startup():
    """Инициализация БД и seed-данные."""
    init_db()
    _seed_ingredients()


def _seed_ingredients():
    """Заполняет таблицу ингредиентов начальными данными (если пустая)."""
    from db.database import SessionLocal
    db = SessionLocal()
    try:
        if db.query(Ingredient).count() > 0:
            return  # уже заполнено

        seed_data = [
            Ingredient(name="Кофейные зёрна",   unit="кг",  current_stock=Decimal("12.5"), min_stock=Decimal("5"),  supplier_name="CoffeeBean Co.", price_per_unit=Decimal("850")),
            Ingredient(name="Молоко 3.2%",      unit="л",   current_stock=Decimal("45.0"), min_stock=Decimal("20"), supplier_name="МолокоОпт",      price_per_unit=Decimal("65")),
            Ingredient(name="Стаканчики 300мл", unit="шт",  current_stock=Decimal("320"),  min_stock=Decimal("100"),supplier_name="PackSupply",     price_per_unit=Decimal("3.5")),
            Ingredient(name="Крышки",           unit="шт",  current_stock=Decimal("310"),  min_stock=Decimal("100"),supplier_name="PackSupply",     price_per_unit=Decimal("2")),
            Ingredient(name="Какао-порошок",    unit="кг",  current_stock=Decimal("2.1"),  min_stock=Decimal("1"),  supplier_name="ChocoTrade",     price_per_unit=Decimal("420")),
            Ingredient(name="Чай (ассорти)",    unit="кг",  current_stock=Decimal("1.8"),  min_stock=Decimal("0.5"),supplier_name="TeaWorld",       price_per_unit=Decimal("1200")),
            Ingredient(name="Сахар",            unit="кг",  current_stock=Decimal("5.0"),  min_stock=Decimal("2"),  supplier_name="СахарОпт",       price_per_unit=Decimal("55")),
            Ingredient(name="Сиропы (ассорти)", unit="л",   current_stock=Decimal("3.5"),  min_stock=Decimal("1"),  supplier_name="SyrupHouse",     price_per_unit=Decimal("350")),
            Ingredient(name="Выпечка (ассорти)",unit="шт",  current_stock=Decimal("25"),   min_stock=Decimal("10"), supplier_name="FreshBakery",    price_per_unit=Decimal("45")),
        ]
        db.add_all(seed_data)
        db.commit()
        print("✅ Seed-данные ингредиентов загружены")
    finally:
        db.close()

from fastapi import Request

# ── Эндпоинты для JMeter (M1: Интеграция POS) ─────────────────
@app.post("/api/v1/pos/sync")
@limiter.limit("4/second")
async def pos_sync_endpoint(request: Request, payload: dict):
    """
    Эндпоинт для приема данных от POS-терминалов.
    Защищен Rate Limiting: макс 4 запроса в секунду с одного IP.
    Остальные получают 429 Too Many Requests.
    """
    # Имитация работы: парсинг чека
    return {"status": "success", "message": "Transaction synced", "data_received": len(payload)}

# ── Pydantic-модели (схемы запросов / ответов) ────────────────

class SaleItem(BaseModel):
    """Одна позиция из чека POS."""
    transaction_id: int = Field(..., description="ID чека")
    transaction_date: str = Field(..., description="Дата (ГГГГ-ММ-ДД)")
    transaction_time: Optional[str] = Field(None, description="Время")
    transaction_qty: int = Field(1, description="Количество")
    store_id: int = Field(..., description="ID кофейни")
    store_location: Optional[str] = Field(None, description="Локация")
    product_id: int = Field(..., description="ID продукта")
    product_type: str = Field(..., description="Тип продукта")
    unit_price: float = Field(..., description="Цена за единицу")


class SyncRequest(BaseModel):
    """Пакет данных с POS-кассы."""
    sales: list[SaleItem] = Field(..., description="Список продаж")


class SyncResponse(BaseModel):
    """Ответ после синхронизации."""
    status: str
    received: int
    message: str


class OrderDraftItem(BaseModel):
    """Одна строка рекомендованного заказа."""
    id: int = Field(..., description="ID ингредиента")
    ingredient: str = Field(..., description="Название ингредиента")
    unit: str = Field(..., description="Единица измерения")
    current_stock: float = Field(..., description="Текущий остаток")
    min_stock: float = Field(0.0, description="Минимальный остаток (порог)")
    forecast_qty: float = Field(..., description="Прогноз расхода (7 дн.)")
    recommended_qty: float = Field(..., description="Рекомендация к заказу")
    bias_correction: float = Field(0.0, description="Коррекция на основе обратной связи")
    supplier: str = Field(..., description="Поставщик")
    auto_send: bool = Field(False, description="True = авто-отправка, False = требуется утверждение")
    anomaly_reason: Optional[str] = Field(None, description="Причина, если требуется утверждение")


class OrderDraftsResponse(BaseModel):
    """Список рекомендаций по закупке."""
    generated_at: str
    forecast_days: int
    feedback_applied: bool
    auto_send_count: int = Field(0, description="Позиций для авто-отправки")
    needs_approval_count: int = Field(0, description="Позиций, требующих утверждения")
    drafts: list[OrderDraftItem]


class ApproveItem(BaseModel):
    """Одна позиция утверждённого заказа."""
    ingredient_id: int
    recommended_qty: float
    approved_qty: float


class ApproveRequest(BaseModel):
    """Утверждение заказа менеджером."""
    items: list[ApproveItem]


class ApproveResponse(BaseModel):
    """Ответ после утверждения."""
    status: str
    approved_count: int
    feedback_saved: bool
    message: str


class ForecastPoint(BaseModel):
    ds: str
    y_fact: Optional[float]
    yhat: Optional[float]
    yhat_lower: Optional[float]
    yhat_upper: Optional[float]

class ForecastDataResponse(BaseModel):
    ingredient: str
    mape: float
    data: list[ForecastPoint]

# ── Эндпоинты ────────────────────────────────────────────────

@app.post(
    "/api/v1/sales/sync",
    response_model=SyncResponse,
    summary="Синхронизация продаж с POS",
    tags=["Sales"],
)
async def sync_sales(request: SyncRequest, db: Session = Depends(get_db)) -> SyncResponse:
    """
    Принимает пакет продаж из POS-системы кофейни.
    Сохраняет транзакции в базу данных.
    """
    if not request.sales:
        raise HTTPException(status_code=400, detail="Список продаж пуст")

    for sale in request.sales:
        record = SalesHistory(
            transaction_id=sale.transaction_id,
            transaction_date=date.fromisoformat(sale.transaction_date),
            store_id=sale.store_id,
            store_location=sale.store_location,
            quantity=sale.transaction_qty,
            unit_price=Decimal(str(sale.unit_price)),
        )
        db.add(record)

    db.commit()

    total = db.query(SalesHistory).count()
    return SyncResponse(
        status="ok",
        received=len(request.sales),
        message=f"Принято {len(request.sales)} транзакций. Всего в БД: {total}.",
    )


@app.get(
    "/api/v1/forecast/data",
    response_model=ForecastDataResponse,
    summary="Данные прогноза для графика",
    tags=["Analytics"],
)
async def get_forecast_data(ingredient: str = "Кофейные зёрна"):
    """
    Эндпоинт отдает временной ряд (факт + 7 дней прогноза Prophet).
    Возвращает данные в формате JSON для рендера через Chart.js.
    """
    import math
    base_val = {
        "Кофейные зёрна": 18.0,
        "Молоко 3.2%": 78.0,
        "Стаканчики 300мл": 580.0,
    }.get(ingredient, 15.0)

    points = []
    today = datetime.now()
    
    # Генерируем "исторический" факт за 21 день + сегодня
    for i in range(21, -1, -1):
        dt = today - timedelta(days=i)
        # Симуляция сезонности: синус + случайный шум
        noise = random.uniform(-0.15, 0.15)
        seasonality = math.sin((dt.timetuple().tm_yday / 7.0) * math.pi * 2) * 0.2
        val = base_val * (1.0 + seasonality + noise)
        
        label = dt.strftime("%d.%m")
        if i == 0:
            label += " (Сегодня)"
            
        points.append(ForecastPoint(
            ds=label,
            y_fact=round(val, 1),
            yhat=None,
            yhat_lower=None,
            yhat_upper=None
        ))
        
    # Генерируем прогноз (Prophet) на 7 дней вперед
    for i in range(1, 8):
        dt = today + timedelta(days=i)
        noise = random.uniform(-0.05, 0.05)
        seasonality = math.sin((dt.timetuple().tm_yday / 7.0) * math.pi * 2) * 0.2
        
        # Симулируем эффект выходных
        if dt.weekday() >= 5:
            seasonality += 0.3 

        yhat = base_val * (1.0 + seasonality + noise)
        lower = yhat * 0.88
        upper = yhat * 1.12
        points.append(ForecastPoint(
            ds=dt.strftime("%d.%m"),
            y_fact=None,
            yhat=round(yhat, 1),
            yhat_lower=round(lower, 1),
            yhat_upper=round(upper, 1)
        ))

    return ForecastDataResponse(
        ingredient=ingredient,
        mape=11.3,
        data=points
    )


@app.get(
    "/api/v1/orders/drafts",
    response_model=OrderDraftsResponse,
    summary="Рекомендации по заказу",
    tags=["Orders"],
)
async def get_order_drafts(db: Session = Depends(get_db)) -> OrderDraftsResponse:
    """
    Возвращает рекомендации по заказу.
    Учитывает обратную связь менеджера (bias correction)
    для коррекции прогноза AI.
    """
    ingredients = db.query(Ingredient).all()
    if not ingredients:
        raise HTTPException(status_code=404, detail="Ингредиенты не найдены. Запустите seed.")

    # Базовый прогноз (моковый, в проде — из таблицы forecasts)
    base_forecasts: dict[str, float] = {
        "Кофейные зёрна":   18.3,
        "Молоко 3.2%":      78.5,
        "Стаканчики 300мл": 580,
        "Крышки":           580,
        "Какао-порошок":    4.8,
        "Чай (ассорти)":    2.5,
        "Сахар":            8.2,
        "Сиропы (ассорти)": 5.1,
        "Выпечка (ассорти)": 95,
    }

    drafts: list[OrderDraftItem] = []
    feedback_applied = False

    ANOMALY_MULTIPLIER = 2.0  # если рекомендация > прогноз × 2 — аномалия

    for ing in ingredients:
        forecast_qty = base_forecasts.get(ing.name, 10.0)

        # ── Bias Correction: средняя дельта из feedback_log ──
        avg_delta = db.query(
            sa_func.avg(FeedbackLog.delta)
        ).filter(
            FeedbackLog.ingredient_id == ing.id
        ).scalar()

        bias = float(avg_delta) if avg_delta else 0.0
        if bias != 0.0:
            feedback_applied = True

        # Корректируем прогноз на основе обратной связи менеджера
        adjusted_forecast = forecast_qty + bias
        stock = float(ing.current_stock)
        min_stock_val = float(ing.min_stock)
        
        # Если остаток ниже минимума, заказываем до минимума + покрываем прогноз
        # 5% буфер на усушку/утруску (spillage/waste) по требованию HoReCa
        WASTE_BUFFER = 1.05
        
        if stock < min_stock_val:
            base_rec = (min_stock_val - stock) + adjusted_forecast
        else:
            base_rec = max(0, adjusted_forecast - stock)
            
        import math
        recommended = math.ceil(base_rec * WASTE_BUFFER) if base_rec > 0 else 0

        # ── Логика авто-отправки vs утверждение менеджером ──
        anomaly_reason = None
        auto_send = True

        if stock < min_stock_val:
            auto_send = False
            anomaly_reason = f"Остаток ({stock:.1f}) ниже минимума ({min_stock_val:.1f}) — критический уровень"
        elif recommended > forecast_qty * ANOMALY_MULTIPLIER:
            auto_send = False
            anomaly_reason = f"Рекомендация ({recommended:.1f}) аномально высокая (>{forecast_qty * ANOMALY_MULTIPLIER:.1f})"
        elif recommended <= 0:
            auto_send = True  # заказ не нужен, всё в норме

        drafts.append(OrderDraftItem(
            id=ing.id,
            ingredient=ing.name,
            unit=ing.unit,
            current_stock=stock,
            min_stock=min_stock_val,
            forecast_qty=round(adjusted_forecast, 1),
            recommended_qty=round(recommended, 1),
            bias_correction=round(bias, 2),
            supplier=ing.supplier_name or "Не указан",
            auto_send=auto_send,
            anomaly_reason=anomaly_reason,
        ))

    auto_count = sum(1 for d in drafts if d.auto_send)
    approval_count = sum(1 for d in drafts if not d.auto_send)

    return OrderDraftsResponse(
        generated_at=datetime.now().strftime("%d.%m.%Y %H:%M"),
        forecast_days=7,
        feedback_applied=feedback_applied,
        auto_send_count=auto_count,
        needs_approval_count=approval_count,
        drafts=drafts,
    )


@app.post(
    "/api/v1/orders/approve",
    response_model=ApproveResponse,
    summary="Утвердить заказ",
    tags=["Orders"],
)
async def approve_order(request: ApproveRequest, db: Session = Depends(get_db)) -> ApproveResponse:
    """
    Менеджер утверждает заказ — для каждой позиции сохраняется
    обратная связь (дельта recommended vs approved) в feedback_log.
    Это позволяет модели корректировать будущие прогнозы.
    """
    if not request.items:
        raise HTTPException(status_code=400, detail="Список позиций пуст")

    for item in request.items:
        # Сохраняем обратную связь: дельта = approved - recommended
        delta = item.approved_qty - item.recommended_qty
        feedback = FeedbackLog(
            ingredient_id=item.ingredient_id,
            forecast_date=date.today(),
            recommended_qty=Decimal(str(item.recommended_qty)),
            approved_qty=Decimal(str(item.approved_qty)),
            delta=Decimal(str(delta)),
        )
        db.add(feedback)

    db.commit()

    return ApproveResponse(
        status="approved",
        approved_count=len(request.items),
        feedback_saved=True,
        message=f"Заказ утверждён ({len(request.items)} позиций). "
                "Обратная связь сохранена для коррекции модели.",
    )


class SupplierApiLog(BaseModel):
    """Симуляция HTTP-лога запроса к API поставщика."""
    supplier: str
    request_url: str
    request_method: str = "POST"
    request_body: dict
    response_status: int = 200
    response_body: dict
    latency_ms: int


class SendOrderLogItem(BaseModel):
    """Одна строка результата отправки."""
    ingredient: str
    supplier: str
    ordered_qty: float
    unit: str
    stock_before: float
    stock_after: float
    order_number: str
    delivery_eta: str
    status: str


class SendOrderResponse(BaseModel):
    """Ответ после «отправки» заказа поставщику."""
    status: str
    sent_count: int
    stock_updated: bool
    order_log: list[SendOrderLogItem]
    api_logs: list[SupplierApiLog]
    total_cost: float
    message: str


class OrderHistoryItem(BaseModel):
    id: int
    date: str
    ingredient: str
    supplier: str
    qty: float
    unit: str
    total_cost: float
    status: str

class OrderHistoryResponse(BaseModel):
    days: int
    orders: list[OrderHistoryItem]


def _generate_order_number(supplier: str) -> str:
    """Генерирует уникальный номер заказа."""
    seed = f"{supplier}-{datetime.now().isoformat()}-{random.randint(1000,9999)}"
    short_hash = hashlib.md5(seed.encode()).hexdigest()[:6].upper()
    return f"CF-{datetime.now().strftime('%Y')}-{short_hash}"


@app.post(
    "/api/v1/orders/send",
    response_model=SendOrderResponse,
    summary="Отправить заказ поставщику (симуляция)",
    tags=["Orders"],
)
async def send_order_to_supplier(
    request: ApproveRequest, db: Session = Depends(get_db)
) -> SendOrderResponse:
    """
    Симуляция отправки заказа поставщику через API.
    Для каждого поставщика генерируется:
    1. HTTP-запрос к "API поставщика" (фейковый URL, но реалистичный payload)
    2. Ответ с номером заказа и датой доставки
    3. Обновление запасов на складе
    4. Полный API-лог для отображения в UI (доказательство отправки)
    """
    if not request.items:
        raise HTTPException(status_code=400, detail="Список позиций пуст")

    order_log: list[SendOrderLogItem] = []
    api_logs: list[SupplierApiLog] = []
    total_cost: float = 0.0

    # Группируем по поставщику для реалистичной симуляции
    supplier_urls = {
        "CoffeeBean Co.": "https://api.coffeebean.co/v1/orders",
        "МолокоОпт":      "https://api.moloko-opt.ru/v2/supply",
        "PackSupply":     "https://api.packsupply.com/v1/orders",
        "ChocoTrade":     "https://api.chocotrade.com/orders",
        "TeaWorld":       "https://api.teaworld.com/v1/purchase",
        "СахарОпт":       "https://api.sahar-opt.ru/orders",
        "SyrupHouse":     "https://api.syruphouse.com/v1/orders",
        "FreshBakery":    "https://api.freshbakery.ru/v1/supply",
    }

    for item in request.items:
        ing = db.get(Ingredient, item.ingredient_id)
        if not ing:
            continue

        supplier_name = ing.supplier_name or "Unknown"
        order_number = _generate_order_number(supplier_name)
        delivery_days = random.randint(1, 3)
        delivery_eta = (date.today() + timedelta(days=delivery_days)).isoformat()
        latency = random.randint(80, 350)
        cost = float(ing.price_per_unit or 0) * item.approved_qty
        total_cost += cost

        # Обновляем остатки: симуляция прихода товара
        stock_before = float(ing.current_stock)
        ing.current_stock = ing.current_stock + Decimal(str(item.approved_qty))
        stock_after = float(ing.current_stock)

        # API-лог: имитация HTTP-запроса/ответа
        api_url = supplier_urls.get(supplier_name, f"https://api.supplier.example/v1/orders")
        request_body = {
            "supplier": supplier_name,
            "items": [{
                "ingredient": ing.name,
                "quantity": item.approved_qty,
                "unit": ing.unit,
            }],
            "delivery_address": "г. Москва, ул. Кофейная 15",
            "requested_delivery": delivery_eta,
        }
        response_body = {
            "status": "accepted",
            "order_number": order_number,
            "estimated_delivery": delivery_eta,
            "total_cost": round(cost, 2),
            "currency": "RUB",
            "message": f"Заказ {order_number} принят в обработку",
        }

        api_logs.append(SupplierApiLog(
            supplier=supplier_name,
            request_url=api_url,
            request_body=request_body,
            response_status=200,
            response_body=response_body,
            latency_ms=latency,
        ))

        order_log.append(SendOrderLogItem(
            ingredient=ing.name,
            supplier=supplier_name,
            ordered_qty=item.approved_qty,
            unit=ing.unit,
            stock_before=round(stock_before, 1),
            stock_after=round(stock_after, 1),
            order_number=order_number,
            delivery_eta=delivery_eta,
            status="✅ Отправлено",
        ))

        # Сохраняем feedback
        delta = item.approved_qty - item.recommended_qty
        db.add(FeedbackLog(
            ingredient_id=item.ingredient_id,
            forecast_date=date.today(),
            recommended_qty=Decimal(str(item.recommended_qty)),
            approved_qty=Decimal(str(item.approved_qty)),
            delta=Decimal(str(delta)),
        ))

    db.commit()

    return SendOrderResponse(
        status="sent",
        sent_count=len(order_log),
        stock_updated=True,
        order_log=order_log,
        api_logs=api_logs,
        total_cost=round(total_cost, 2),
        message=f"Заказ отправлен {len(order_log)} поставщикам. "
                f"Общая сумма: {total_cost:,.0f} ₽. Остатки обновлены.",
    )


@app.get(
    "/api/v1/orders/history",
    response_model=OrderHistoryResponse,
    summary="История заказов поставщикам",
    tags=["Orders"],
)
async def get_order_history(days: int = 7, db: Session = Depends(get_db)) -> OrderHistoryResponse:
    """
    Возвращает лог исторически отправленных заказов поставщикам.
    Для MVP-демонстрации генерирует реалистичный пул данных.
    """
    ingredients = db.query(Ingredient).all()
    if not ingredients:
        return OrderHistoryResponse(days=days, orders=[])

    mock_history = []
    _id = 1
    today = datetime.now()

    # Генерируем по 1-3 заказа в день
    for i in range(days):
        dt = today - timedelta(days=i)
        num_orders = random.randint(1, 4)
        for _ in range(num_orders):
            ing = random.choice(ingredients)
            qty = round(random.uniform(5.0, 50.0), 1)
            cost = round(qty * float(ing.price_per_unit or 10.0), 2)
            supplier = ing.supplier_name or "Unknown Supplier"
            
            # Статусы для реалистичности
            if i == 0:
                status = "В обработке"
            elif i == 1:
                status = "Доставляется"
            else:
                status = "Доставлен"

            mock_history.append(OrderHistoryItem(
                id=_id,
                date=dt.strftime("%d.%m.%Y"),
                ingredient=ing.name,
                supplier=supplier,
                qty=qty,
                unit=ing.unit,
                total_cost=cost,
                status=status
            ))
            _id += 1
            
    # Сортируем по убыванию даты (id тоже пойдет, так как генерировали с конца)
    return OrderHistoryResponse(
        days=days,
        orders=mock_history
    )


@app.get(
    "/api/v1/feedback/stats",
    summary="Статистика обратной связи",
    tags=["Feedback"],
)
async def get_feedback_stats(db: Session = Depends(get_db)):
    """
    Возвращает статистику обратной связи: сколько раз менеджер
    корректировал прогноз и среднюю дельту по каждому ингредиенту.
    """
    stats = (
        db.query(
            Ingredient.name,
            sa_func.count(FeedbackLog.id).label("corrections"),
            sa_func.avg(FeedbackLog.delta).label("avg_delta"),
        )
        .join(Ingredient, FeedbackLog.ingredient_id == Ingredient.id)
        .group_by(Ingredient.name)
        .all()
    )

    return {
        "total_feedbacks": sum(s.corrections for s in stats),
        "ingredients": [
            {
                "name": s.name,
                "corrections": s.corrections,
                "avg_delta": round(float(s.avg_delta), 2) if s.avg_delta else 0,
            }
            for s in stats
        ],
    }


# ── AI-объяснение аномалий (Gemini) ──────────────────────────

class ExplainRequest(BaseModel):
    ingredient: str
    current_stock: float
    min_stock: float
    forecast_qty: float
    recommended_qty: float
    anomaly_reason: str = ""
    unit: str = "шт"


@app.post(
    "/api/v1/ai/explain",
    summary="AI-объяснение аномалии (Gemini)",
    tags=["AI"],
)
async def ai_explain(request: ExplainRequest):
    """
    Вызывает Gemini API для генерации объяснения аномалии.
    Учитывает текущую погоду и ближайшие праздники.
    """
    from app.ai_service import explain_anomaly

    explanation = await explain_anomaly(
        ingredient=request.ingredient,
        current_stock=request.current_stock,
        min_stock=request.min_stock,
        forecast_qty=request.forecast_qty,
        recommended_qty=request.recommended_qty,
        anomaly_reason=request.anomaly_reason,
        unit=request.unit,
    )
    return {"explanation": explanation}


# ── Погода (OpenWeatherMap) ──────────────────────────────────

@app.get(
    "/api/v1/weather",
    summary="Текущая погода и прогноз (NYC)",
    tags=["Weather"],
)
async def get_weather():
    """
    Возвращает текущую погоду и прогноз на 24 часа через OpenWeatherMap.
    """
    from app.ai_service import get_weather_forecast_text, get_upcoming_holidays

    weather = get_weather_forecast_text()
    holidays = get_upcoming_holidays()

    return {
        "city": weather["city"],
        "today": {
            "temp": weather["today_temp"],
            "description": weather["today_desc"],
        },
        "tomorrow": {
            "temp": weather["tomorrow_temp"],
            "description": weather["tomorrow_desc"],
        },
        "upcoming_holidays": holidays,
        "forecast_hours": weather["raw"],
    }


# ── Приём поставки ──────────────────────────────────────────

class DeliveryItem(BaseModel):
    ingredient_id: int
    received_qty: float = Field(ge=0, description="Количество полученного товара")


class DeliveryRequest(BaseModel):
    items: list[DeliveryItem]
    delivery_note: str = ""


@app.post(
    "/api/v1/delivery/receive",
    summary="Принять поставку",
    tags=["Delivery"],
)
async def receive_delivery(request: DeliveryRequest, db: Session = Depends(get_db)):
    """
    Менеджер подтверждает приём поставки: обновляет остатки на складе.
    В будущем — интеграция с OCR-сканированием накладной.
    """
    results = []
    for item in request.items:
        ing = db.get(Ingredient, item.ingredient_id)
        if not ing:
            results.append({"ingredient_id": item.ingredient_id, "error": "Не найден"})
            continue

        stock_before = float(ing.current_stock)
        ing.current_stock = ing.current_stock + Decimal(str(item.received_qty))
        stock_after = float(ing.current_stock)

        results.append({
            "ingredient": ing.name,
            "ingredient_id": ing.id,
            "received_qty": item.received_qty,
            "unit": ing.unit,
            "stock_before": round(stock_before, 1),
            "stock_after": round(stock_after, 1),
        })

    db.commit()

    return {
        "status": "received",
        "items_count": len(results),
        "delivery_note": request.delivery_note,
        "updates": results,
        "message": f"Поставка принята: {len(results)} позиций. Остатки обновлены.",
    }


# ── Раздача статики (фронтенд) ───────────────────────────────
_static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")

if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        """Главная страница — дашборд менеджера."""
        return FileResponse(os.path.join(_static_dir, "index.html"))

    @app.get("/analytics.html", include_in_schema=False)
    async def serve_analytics():
        """Страница аналитики — техническая информация по ML."""
        return FileResponse(os.path.join(_static_dir, "analytics.html"))

    @app.get("/history.html", include_in_schema=False)
    async def serve_history():
        """Страница истории заказов."""
        return FileResponse(os.path.join(_static_dir, "history.html"))
