from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import DateTime, ForeignKey, Index, Integer, Numeric, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class SessionModel(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_name: Mapped[str] = mapped_column(String(120), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    total_results: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_wins: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_losses: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    maximum_loss_streak: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    maximum_win_streak: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    results: Mapped[list["GameResult"]] = relationship(back_populates="session")


class GameResult(Base):
    __tablename__ = "game_results"
    __table_args__ = (
        UniqueConstraint("fingerprint", name="uq_game_results_fingerprint"),
        Index("ix_game_results_timestamp", "timestamp"),
        Index("ix_game_results_session_id", "session_id"),
        Index("ix_game_results_result_type", "result_type"),
        Index("ix_game_results_multiplier", "multiplier"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), nullable=False)
    result_number: Mapped[int] = mapped_column(Integer, nullable=False)
    multiplier: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    result_type: Mapped[str] = mapped_column(String(8), nullable=False)
    loss_streak_before: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    win_streak_before: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    ocr_confidence: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)
    raw_ocr_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    original_multiplier: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    corrected_multiplier: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    corrected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    session: Mapped[SessionModel] = relationship(back_populates="results")


class AppSettings(Base):
    __tablename__ = "settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    screen_x: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    screen_y: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    screen_width: Mapped[int] = mapped_column(Integer, default=500, nullable=False)
    screen_height: Mapped[int] = mapped_column(Integer, default=250, nullable=False)
    ocr_mode: Mapped[str] = mapped_column(String(24), default="both", nullable=False)
    minimum_interval: Mapped[Decimal] = mapped_column(Numeric(5, 2), default=Decimal("1.00"), nullable=False)
    ocr_confidence_threshold: Mapped[Decimal] = mapped_column(Numeric(5, 2), default=Decimal("80.00"), nullable=False)
    loss_threshold: Mapped[Decimal] = mapped_column(Numeric(10, 2), default=Decimal("1.50"), nullable=False)
    capture_interval: Mapped[Decimal] = mapped_column(Numeric(5, 2), default=Decimal("0.75"), nullable=False)
    max_sanity_multiplier: Mapped[Decimal] = mapped_column(Numeric(10, 2), default=Decimal("10000.00"), nullable=False)
    allow_low_confidence: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)


class TableCapture(Base):
    __tablename__ = "table_captures"
    __table_args__ = (UniqueConstraint("period", name="uq_table_captures_period"), Index("ix_table_captures_timestamp", "captured_at"))

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    period: Mapped[str] = mapped_column(String(64), nullable=False)
    number: Mapped[int] = mapped_column(Integer, nullable=False)
    size: Mapped[str] = mapped_column(String(8), nullable=False)
    colors: Mapped[str] = mapped_column(String(128), nullable=False)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    confidence: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    previous_period: Mapped[str | None] = mapped_column(String(64), nullable=True)
    previous_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    data_status: Mapped[str | None] = mapped_column(String(24), nullable=True)
    source_status: Mapped[str | None] = mapped_column(String(24), nullable=True)


class PatternIdentifier(Base):
    __tablename__ = "pattern_identifiers"
    __table_args__ = (UniqueConstraint("pattern", name="uq_pattern_identifiers_pattern"), Index("ix_pattern_identifiers_occurrences", "occurrences"))

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pattern: Mapped[str] = mapped_column(String(64), nullable=False)
    occurrences: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    next_big_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    next_small_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)


class PredictionSignal(Base):
    __tablename__ = "prediction_signals"
    __table_args__ = (UniqueConstraint("target_period", name="uq_prediction_signals_target_period"), Index("ix_prediction_signals_issued_at", "issued_at"))

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    target_period: Mapped[str] = mapped_column(String(64), nullable=False)
    predicted_size: Mapped[str] = mapped_column(String(8), nullable=False)
    pattern: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[Decimal] = mapped_column(Numeric(5, 2), nullable=False)
    matching_records: Mapped[int] = mapped_column(Integer, nullable=False)
    issued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    actual_size: Mapped[str | None] = mapped_column(String(8), nullable=True)
    outcome: Mapped[str | None] = mapped_column(String(8), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class WingoAnalysis(Base):
    __tablename__ = "wingo_analysis"
    __table_args__ = (UniqueConstraint("analysis_period", name="uq_wingo_analysis_period"), Index("ix_wingo_analysis_created_at", "created_at"))

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_period: Mapped[str] = mapped_column(String(64), nullable=False)
    actual_period: Mapped[str | None] = mapped_column(String(64), nullable=True)
    big_score: Mapped[Decimal] = mapped_column(Numeric(6, 2), nullable=False)
    small_score: Mapped[Decimal] = mapped_column(Numeric(6, 2), nullable=False)
    signal: Mapped[str] = mapped_column(String(24), nullable=False)
    signal_strength: Mapped[str] = mapped_column(String(16), nullable=False)
    sample_size: Mapped[int] = mapped_column(Integer, nullable=False)
    data_quality: Mapped[str] = mapped_column(String(16), nullable=False)
    backtest_accuracy: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)
    actual_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    actual_size: Mapped[str | None] = mapped_column(String(8), nullable=True)
    # Includes resolution states such as SKIPPED_MISSING_PERIOD.
    outcome: Mapped[str | None] = mapped_column(String(32), nullable=True)
    evaluation_mode: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class CaptureAudit(Base):
    __tablename__ = "capture_audits"
    __table_args__ = (Index("ix_capture_audits_created_at", "created_at"), Index("ix_capture_audits_period", "period"))

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    period: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    message: Mapped[str] = mapped_column(String(240), nullable=False)
    source_status: Mapped[str] = mapped_column(String(24), default="OCR", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
