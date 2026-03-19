"""
CoffeeStock AI — Конфигурация базы данных
==========================================
SQLite (dev/demo) ↔ PostgreSQL (production/Railway).
Переключение — через переменную окружения DATABASE_URL.
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.models import Base

# Если задана DATABASE_URL (Railway пробросит автоматически) — используем PostgreSQL.
# Иначе — SQLite (файл coffeestock.db в корне проекта).
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "sqlite:///coffeestock.db"
)

# Railway выдаёт postgres:// а SQLAlchemy 2.0 требует postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL,
    echo=False,
    # Для SQLite — check_same_thread=False (FastAPI многопоточный)
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Создаёт все таблицы из models.py (если не существуют)."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Генератор сессии для FastAPI Depends()."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
