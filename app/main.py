"""
CoffeeStock AI — FastAPI Backend (v2)
======================================
API с реальной БД (SQLite/PostgreSQL), feedback loop,
и полным циклом: POS → ML → заказ → обратная связь.

Запуск: uvicorn app.main:app --reload --port 8000
"""

import os
from datetime import date, datetime
from decimal import Decimal
from typing import Optional

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
    forecast_qty: float = Field(..., description="Прогноз расхода (7 дн.)")
    recommended_qty: float = Field(..., description="Рекомендация к заказу")
    bias_correction: float = Field(0.0, description="Коррекция на основе обратной связи")
    supplier: str = Field(..., description="Поставщик")


class OrderDraftsResponse(BaseModel):
    """Список рекомендаций по закупке."""
    generated_at: str
    forecast_days: int
    feedback_applied: bool
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
        recommended = max(0, adjusted_forecast - stock)

        drafts.append(OrderDraftItem(
            id=ing.id,
            ingredient=ing.name,
            unit=ing.unit,
            current_stock=stock,
            forecast_qty=round(adjusted_forecast, 1),
            recommended_qty=round(recommended, 1),
            bias_correction=round(bias, 2),
            supplier=ing.supplier_name or "Не указан",
        ))

    return OrderDraftsResponse(
        generated_at=datetime.now().strftime("%d.%m.%Y %H:%M"),
        forecast_days=7,
        feedback_applied=feedback_applied,
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


class SendOrderResponse(BaseModel):
    """Ответ после «отправки» заказа поставщику."""
    status: str
    sent_count: int
    stock_updated: bool
    order_log: list[dict]
    message: str


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
    Симуляция отправки заказа поставщику.
    После «отправки»:
    1. Статус заказа → 'sent'
    2. Остатки на складе (current_stock) увеличиваются на approved_qty
       — имитация прихода товара.
    3. Возвращается лог операции для отображения в UI.
    """
    if not request.items:
        raise HTTPException(status_code=400, detail="Список позиций пуст")

    order_log: list[dict] = []

    for item in request.items:
        ing = db.get(Ingredient, item.ingredient_id)
        if not ing:
            continue

        # Обновляем остатки: симуляция прихода товара
        stock_before = float(ing.current_stock)
        ing.current_stock = ing.current_stock + Decimal(str(item.approved_qty))
        stock_after = float(ing.current_stock)

        order_log.append({
            "ingredient": ing.name,
            "supplier": ing.supplier_name,
            "ordered_qty": item.approved_qty,
            "unit": ing.unit,
            "stock_before": round(stock_before, 1),
            "stock_after": round(stock_after, 1),
            "status": "✅ Отправлено",
        })

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
        message=f"Заказ отправлен {len(order_log)} поставщикам. "
                "Остатки на складе обновлены.",
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


# ── Раздача статики (фронтенд) ───────────────────────────────
_static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")

if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        """Главная страница — дашборд менеджера."""
        return FileResponse(os.path.join(_static_dir, "index.html"))
