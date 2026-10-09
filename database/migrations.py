from __future__ import annotations

import logging
from decimal import Decimal

from sqlalchemy import inspect, select, text
from sqlalchemy.exc import OperationalError

from config import config
from database.connection import create_database_if_missing, engine, get_session
from database.models import AppSettings, Base, TableCapture

logger = logging.getLogger(__name__)


def ensure_capture_metadata() -> None:
    """Add metadata columns to existing table_captures installations safely."""
    additions = {
        "previous_period": "VARCHAR(64) NULL",
        "previous_number": "INTEGER NULL",
        "data_status": "VARCHAR(24) NULL",
        "source_status": "VARCHAR(24) NULL",
    }
    existing = {column["name"] for column in inspect(engine).get_columns("table_captures")}
    with engine.begin() as connection:
        for name, sql_type in additions.items():
            if name not in existing:
                connection.execute(text(f"ALTER TABLE table_captures ADD COLUMN {name} {sql_type}"))

    with get_session() as session:
        rows = list(session.scalars(select(TableCapture).order_by(TableCapture.captured_at, TableCapture.id)))
        previous = None
        for row in rows:
            if row.data_status is None:
                number_ok = isinstance(row.number, int) and 0 <= row.number <= 9
                size_ok = row.size == ("Small" if number_ok and row.number < 5 else "Big") if number_ok else False
                period_ok = str(row.period).isdigit() and 10 <= len(str(row.period)) <= 24
                if not (number_ok and size_ok and period_ok):
                    row.data_status = "Invalid result"
                elif previous is not None:
                    delta = int(row.period) - int(previous.period) if len(str(row.period)) == len(str(previous.period)) else None
                    row.data_status = "Valid" if delta == 1 else "Missing period" if delta is not None and delta > 1 else "Abnormal jump"
                else:
                    row.data_status = "Valid"
                row.source_status = "LEGACY"
                if period_ok and number_ok and size_ok and row.data_status != "Abnormal jump":
                    row.previous_period = previous.period if previous else None
                    row.previous_number = previous.number if previous else None
                    previous = row
        logger.info("Capture metadata migration checked %s historical rows", len(rows))


def ensure_wingo_analysis_outcome_width() -> None:
    """Widen the outcome column for descriptive skipped-result statuses."""
    inspector = inspect(engine)
    if not inspector.has_table("wingo_analysis"):
        return
    outcome_column = next(
        (column for column in inspector.get_columns("wingo_analysis") if column["name"] == "outcome"),
        None,
    )
    if outcome_column is None:
        return
    current_length = getattr(outcome_column["type"], "length", None)
    if engine.dialect.name == "mysql" and (current_length is None or current_length < 32):
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE wingo_analysis MODIFY COLUMN outcome VARCHAR(32) NULL"))
        logger.info("Widened wingo_analysis.outcome to VARCHAR(32)")


def initialize_database() -> None:
    try:
        create_database_if_missing()
        Base.metadata.create_all(bind=engine)
        ensure_capture_metadata()
        ensure_wingo_analysis_outcome_width()
        with get_session() as session:
            settings = session.get(AppSettings, 1)
            if settings is None:
                session.add(AppSettings(id=1, loss_threshold=config.default_loss_threshold))
            elif settings.loss_threshold == Decimal("2.00"):
                settings.loss_threshold = config.default_loss_threshold
        logger.info("Database initialized: %s", config.database.name)
    except OperationalError as exc:
        raise RuntimeError(
            "Could not connect to MySQL. Check that MySQL is running and .env contains valid DB credentials."
        ) from exc


if __name__ == "__main__":
    initialize_database()
    print("Database initialized.")
