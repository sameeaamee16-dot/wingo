from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from config import config


engine = create_engine(config.database.database_url, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True)


def create_database_if_missing() -> None:
    server_engine = create_engine(config.database.server_url, pool_pre_ping=True, future=True)
    with server_engine.begin() as conn:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{config.database.name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
    server_engine.dispose()


@contextmanager
def get_session() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
