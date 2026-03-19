"""
CoffeeStock AI — Модели SQLAlchemy 2.0
Все таблицы для системы управления запасами кофейни.
"""

from datetime import date, datetime, time
from decimal import Decimal
from typing import List, Optional

from sqlalchemy import (
    ForeignKey,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)


# ── Базовый класс ────────────────────────────────────────────
class Base(DeclarativeBase):
    """Базовый класс для всех моделей."""
    pass


# ── Пользователи (менеджеры) ─────────────────────────────────
class User(Base):
    """Менеджеры и администраторы системы."""
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(100))
    role: Mapped[str] = mapped_column(String(20), default="manager")  # admin / manager
    store_id: Mapped[Optional[int]] = mapped_column()
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    # Связь: заказы, утверждённые пользователем
    approved_orders: Mapped[List["OrderDraft"]] = relationship(back_populates="approved_by_user")

    def __repr__(self) -> str:
        return f"<User {self.username} ({self.role})>"


# ── Готовые напитки / товары ──────────────────────────────────
class Product(Base):
    """Справочник напитков и товаров (из POS)."""
    __tablename__ = "products"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    product_type: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    unit_price: Mapped[Optional[Decimal]] = mapped_column()
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    # Связи
    recipes: Mapped[List["RecipeBOM"]] = relationship(back_populates="product", cascade="all, delete-orphan")
    sales: Mapped[List["SalesHistory"]] = relationship(back_populates="product")

    def __repr__(self) -> str:
        return f"<Product {self.product_type}>"


# ── Сырьё / Ингредиенты ──────────────────────────────────────
class Ingredient(Base):
    """Справочник сырья: молоко, зёрна, стаканчики..."""
    __tablename__ = "ingredients"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    unit: Mapped[str] = mapped_column(String(20), nullable=False)  # ml, g, pcs
    min_stock: Mapped[Decimal] = mapped_column(default=Decimal("0"))
    current_stock: Mapped[Decimal] = mapped_column(default=Decimal("0"))
    supplier_name: Mapped[Optional[str]] = mapped_column(String(100))
    price_per_unit: Mapped[Optional[Decimal]] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    # Связи
    recipes: Mapped[List["RecipeBOM"]] = relationship(back_populates="ingredient")
    forecasts: Mapped[List["Forecast"]] = relationship(back_populates="ingredient")
    order_drafts: Mapped[List["OrderDraft"]] = relationship(back_populates="ingredient")

    def __repr__(self) -> str:
        return f"<Ingredient {self.name} ({self.unit})>"


# ── Рецептура / BOM (Bill of Materials) ──────────────────────
class RecipeBOM(Base):
    """
    Разузлование: связь «многие-ко-многим» между Product и Ingredient.
    Указывает расход ингредиента на 1 единицу продукта.
    """
    __tablename__ = "recipe_bom"
    __table_args__ = (
        UniqueConstraint("product_id", "ingredient_id", name="uq_product_ingredient"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), nullable=False)
    ingredient_id: Mapped[int] = mapped_column(ForeignKey("ingredients.id"), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(nullable=False)  # расход на 1 шт.

    # Связи
    product: Mapped["Product"] = relationship(back_populates="recipes")
    ingredient: Mapped["Ingredient"] = relationship(back_populates="recipes")

    def __repr__(self) -> str:
        return f"<BOM product={self.product_id} → ingredient={self.ingredient_id}: {self.quantity}>"


# ── История продаж (данные из POS-кассы) ─────────────────────
class SalesHistory(Base):
    """Транзакции продаж, синхронизированные из POS."""
    __tablename__ = "sales_history"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    transaction_id: Mapped[int] = mapped_column(nullable=False)
    transaction_date: Mapped[date] = mapped_column(nullable=False, index=True)
    transaction_time: Mapped[Optional[time]] = mapped_column()
    store_id: Mapped[int] = mapped_column(nullable=False, index=True)
    store_location: Mapped[Optional[str]] = mapped_column(String(100))
    product_id: Mapped[Optional[int]] = mapped_column(ForeignKey("products.id"))
    quantity: Mapped[int] = mapped_column(default=1)
    unit_price: Mapped[Optional[Decimal]] = mapped_column()
    synced_at: Mapped[datetime] = mapped_column(server_default=func.now())

    # Связь
    product: Mapped[Optional["Product"]] = relationship(back_populates="sales")

    def __repr__(self) -> str:
        return f"<Sale #{self.transaction_id} @ {self.transaction_date}>"


# ── Прогнозы ML ──────────────────────────────────────────────
class Forecast(Base):
    """Результат работы ML-модели: прогноз расхода сырья."""
    __tablename__ = "forecasts"
    __table_args__ = (
        UniqueConstraint("ingredient_id", "forecast_date", name="uq_ingredient_forecast_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ingredient_id: Mapped[int] = mapped_column(ForeignKey("ingredients.id"), nullable=False)
    forecast_date: Mapped[date] = mapped_column(nullable=False)
    predicted_qty: Mapped[Decimal] = mapped_column(nullable=False)
    lower_bound: Mapped[Optional[Decimal]] = mapped_column()
    upper_bound: Mapped[Optional[Decimal]] = mapped_column()
    model_name: Mapped[str] = mapped_column(String(50), default="Prophet")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    # Связь
    ingredient: Mapped["Ingredient"] = relationship(back_populates="forecasts")

    def __repr__(self) -> str:
        return f"<Forecast {self.ingredient_id} @ {self.forecast_date}: {self.predicted_qty}>"


# ── Черновик заказа поставщику ────────────────────────────────
class OrderDraft(Base):
    """Автоматически сформированная заявка на закупку."""
    __tablename__ = "order_drafts"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ingredient_id: Mapped[int] = mapped_column(ForeignKey("ingredients.id"), nullable=False)
    current_stock: Mapped[Optional[Decimal]] = mapped_column()
    forecast_qty: Mapped[Optional[Decimal]] = mapped_column()
    recommended_qty: Mapped[Decimal] = mapped_column(nullable=False)
    approved_qty: Mapped[Optional[Decimal]] = mapped_column()
    status: Mapped[str] = mapped_column(String(20), default="draft", index=True)  # draft / approved / sent
    approved_by: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now(), index=True)
    approved_at: Mapped[Optional[datetime]] = mapped_column()

    # Связи
    ingredient: Mapped["Ingredient"] = relationship(back_populates="order_drafts")
    approved_by_user: Mapped[Optional["User"]] = relationship(back_populates="approved_orders")

    def __repr__(self) -> str:
        return f"<OrderDraft {self.ingredient_id} qty={self.recommended_qty} [{self.status}]>"


# ── Лог обратной связи (для самообучения модели) ─────────────
class FeedbackLog(Base):
    """
    Цикл обратной связи: менеджер корректирует прогноз AI.
    Дельта (approved - recommended) используется для bias-коррекции
    будущих прогнозов Prophet.
    """
    __tablename__ = "feedback_log"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ingredient_id: Mapped[int] = mapped_column(ForeignKey("ingredients.id"), nullable=False)
    forecast_date: Mapped[date] = mapped_column(nullable=False)
    recommended_qty: Mapped[Decimal] = mapped_column(nullable=False)  # что предложил AI
    approved_qty: Mapped[Decimal] = mapped_column(nullable=False)     # что утвердил менеджер
    delta: Mapped[Decimal] = mapped_column(nullable=False)            # approved - recommended
    reason: Mapped[Optional[str]] = mapped_column(String(255))        # комментарий менеджера
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    # Связь
    ingredient: Mapped["Ingredient"] = relationship()

    def __repr__(self) -> str:
        return f"<Feedback {self.ingredient_id}: delta={self.delta}>"
