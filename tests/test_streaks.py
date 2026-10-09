from decimal import Decimal

from statistics.analytics_engine import loss_streaks
from statistics.streak_engine import StreakEngine


def test_complete_loss_streak_only() -> None:
    engine = StreakEngine()
    for value in [Decimal("1.21"), Decimal("1.35"), Decimal("1.18"), Decimal("2.41")]:
        engine.add_multiplier(value)
    assert engine.state.completed_loss_streaks == [(3, Decimal("2.41"), "WIN")]


def test_loss_streak_analysis_uses_exact_preceding_sequences() -> None:
    values = [Decimal("1.21"), Decimal("1.35"), Decimal("1.18"), Decimal("2.41"), Decimal("1.14"), Decimal("1.74"), Decimal("6.75")]
    report = loss_streaks(values, 1.5)
    assert report[2]["sample_size"] == 1
    assert report[1]["sample_size"] == 1
    assert report[0]["sample_size"] == 2
