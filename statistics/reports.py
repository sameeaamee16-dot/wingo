from __future__ import annotations

from decimal import Decimal

from database.models import GameResult
from statistics.analytics import loss_streak_analysis, results_to_dataframe


def build_loss_streak_report(rows: list[GameResult], threshold: Decimal = Decimal("1.50")):
    multipliers = [Decimal(row.multiplier) for row in rows]
    return loss_streak_analysis(multipliers, threshold=threshold)


def build_results_report(rows: list[GameResult]):
    return results_to_dataframe(rows)
