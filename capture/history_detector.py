from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal


@dataclass
class HistoryDetector:
    previous: list[Decimal] = field(default_factory=list)
    tolerance: Decimal = Decimal("0.01")

    def detect_new(self, current: list[Decimal]) -> Decimal | None:
        if not current:
            return None
        if not self.previous:
            self.previous = current
            return current[0]

        for candidate in current:
            if not any(abs(candidate - old) <= self.tolerance for old in self.previous[: min(5, len(self.previous))]):
                self.previous = current
                return candidate

        self.previous = current
        return None
