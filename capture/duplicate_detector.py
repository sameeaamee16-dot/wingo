from __future__ import annotations

import time
from decimal import Decimal


class DuplicateDetector:
    def __init__(self, minimum_interval: float = 1.0, tolerance: Decimal = Decimal("0.01")) -> None:
        self.minimum_interval = minimum_interval
        self.tolerance = tolerance
        self.previous_multiplier: Decimal | None = None
        self.previous_time = 0.0

    def is_duplicate(self, multiplier: Decimal, now: float | None = None) -> bool:
        now = now or time.time()
        if self.previous_multiplier is None:
            self.previous_multiplier = multiplier
            self.previous_time = now
            return False
        same_value = abs(multiplier - self.previous_multiplier) <= self.tolerance
        too_soon = now - self.previous_time < self.minimum_interval
        if same_value or too_soon:
            return True
        self.previous_multiplier = multiplier
        self.previous_time = now
        return False
