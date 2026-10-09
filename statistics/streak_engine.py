from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from ocr.validator import classify_multiplier


@dataclass
class StreakState:
    current_loss_streak: int = 0
    current_win_streak: int = 0
    max_loss_streak: int = 0
    max_win_streak: int = 0
    completed_loss_streaks: list[tuple[int, Decimal, str]] = field(default_factory=list)
    pending_loss_streak: int = 0


class StreakEngine:
    def __init__(self, threshold: Decimal = Decimal("1.50")) -> None:
        self.threshold = threshold
        self.state = StreakState()

    def add_multiplier(self, multiplier: Decimal) -> tuple[str, int, int]:
        result_type = classify_multiplier(multiplier, self.threshold)
        before_loss = self.state.current_loss_streak
        before_win = self.state.current_win_streak

        if result_type == "LOSS":
            self.state.current_loss_streak += 1
            self.state.current_win_streak = 0
            self.state.pending_loss_streak = self.state.current_loss_streak
        else:
            if self.state.pending_loss_streak > 0:
                self.state.completed_loss_streaks.append((self.state.pending_loss_streak, multiplier, result_type))
            self.state.current_win_streak += 1
            self.state.current_loss_streak = 0
            self.state.pending_loss_streak = 0

        self.state.max_loss_streak = max(self.state.max_loss_streak, self.state.current_loss_streak)
        self.state.max_win_streak = max(self.state.max_win_streak, self.state.current_win_streak)
        return result_type, before_loss, before_win

    @classmethod
    def from_results(cls, multipliers: list[Decimal], threshold: Decimal = Decimal("1.50")) -> "StreakEngine":
        engine = cls(threshold)
        for multiplier in multipliers:
            engine.add_multiplier(multiplier)
        return engine


def compute_result_streaks(multipliers: list[Decimal], threshold: Decimal = Decimal("1.50")) -> list[tuple[str, int, int]]:
    engine = StreakEngine(threshold)
    rows: list[tuple[str, int, int]] = []
    for multiplier in multipliers:
        rows.append(engine.add_multiplier(multiplier))
    return rows
