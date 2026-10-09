from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / "logs" / "matplotlib"))
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")


@dataclass(frozen=True)
class DatabaseConfig:
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "3307"))
    name: str = os.getenv("DB_NAME", "multiplier_tracker")
    user: str = os.getenv("DB_USER", "root")
    password: str = os.getenv("DB_PASSWORD", "Admin")

    @property
    def server_url(self) -> str:
        return f"mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}"

    @property
    def database_url(self) -> str:
        return f"{self.server_url}/{self.name}?charset=utf8mb4"


@dataclass(frozen=True)
class AppConfig:
    database: DatabaseConfig = DatabaseConfig()
    app_name: str = "Multiplier Tracker"
    subtitle: str = "Automatic Game Result Recorder & Statistics Analyzer"
    default_loss_threshold: Decimal = Decimal("1.50")
    default_minimum_interval: float = 1.0
    default_capture_interval: float = 0.75
    default_confidence_threshold: float = 80.0
    max_sanity_multiplier: Decimal = Decimal(os.getenv("MAX_SANITY_MULTIPLIER", "10000.00"))
    logs_dir: Path = BASE_DIR / "logs"
    tesseract_cmd: str = os.getenv("TESSERACT_CMD", "")


@dataclass(frozen=True)
class AnalyticsConfig:
    """Central settings for descriptive, out-of-sample analysis."""

    threshold: Decimal = Decimal(os.getenv("ANALYSIS_THRESHOLD", "1.50"))
    min_pattern_sample: int = int(os.getenv("MIN_PATTERN_SAMPLE", "20"))
    initial_training_size: int = int(os.getenv("INITIAL_TRAINING_SIZE", "300"))
    test_window_size: int = int(os.getenv("TEST_WINDOW_SIZE", "100"))
    pattern_length: int = int(os.getenv("PATTERN_LENGTH", "3"))
    min_historical_accuracy: float = float(os.getenv("MIN_HISTORICAL_ACCURACY", "0.55"))
    recent_windows: tuple[int, ...] = (50, 100, 250, 500)


config = AppConfig()
analytics_config = AnalyticsConfig()
