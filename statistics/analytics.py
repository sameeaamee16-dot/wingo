from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal

import pandas as pd

from ocr.validator import classify_multiplier


@dataclass(frozen=True)
class SummaryStats:
    total: int
    wins: int
    losses: int
    win_pct: float
    loss_pct: float
    current_streak_label: str
    max_loss_streak: int
    max_win_streak: int


def summarize_results(rows: list[object], threshold: Decimal = Decimal("1.50")) -> SummaryStats:
    total = len(rows)
    # Reclassify raw multipliers so a changed analysis threshold applies consistently
    # to historical rows instead of trusting the classification stored at record time.
    classifications = [classify_multiplier(Decimal(row.multiplier), threshold) for row in rows]
    wins = classifications.count("WIN")
    losses = total - wins
    max_loss = 0
    max_win = 0
    cur_loss = 0
    cur_win = 0
    for result_type in classifications:
        if result_type == "LOSS":
            cur_loss += 1
            cur_win = 0
        else:
            cur_win += 1
            cur_loss = 0
        max_loss = max(max_loss, cur_loss)
        max_win = max(max_win, cur_win)
    current = f"{cur_loss} LOSS" if cur_loss else (f"{cur_win} WIN" if cur_win else "0")
    return SummaryStats(
        total=total,
        wins=wins,
        losses=losses,
        win_pct=(wins / total * 100) if total else 0.0,
        loss_pct=(losses / total * 100) if total else 0.0,
        current_streak_label=current,
        max_loss_streak=max_loss,
        max_win_streak=max_win,
    )


def loss_streak_analysis(multipliers: list[Decimal], threshold: Decimal = Decimal("1.50"), max_bucket: int = 8) -> pd.DataFrame:
    buckets: dict[str, list[Decimal]] = defaultdict(list)
    loss_count = 0
    for multiplier in multipliers:
        result_type = classify_multiplier(multiplier, threshold)
        if result_type == "LOSS":
            loss_count += 1
            continue
        if loss_count > 0:
            label = f"{max_bucket}+" if loss_count >= max_bucket else str(loss_count)
            buckets[label].append(multiplier)
            loss_count = 0

    records = []
    ordered_labels = [str(i) for i in range(1, max_bucket)] + [f"{max_bucket}+"]
    for label in ordered_labels:
        values = buckets.get(label, [])
        wins = sum(1 for value in values if value >= threshold)
        losses = len(values) - wins
        records.append(
            {
                "Loss Streak": label,
                "Occurrences": len(values),
                "Next Result >= 1.5x Count": wins,
                "Next Result < 1.5x Count": losses,
                "Percentage >= 1.5x": round((wins / len(values) * 100), 2) if values else 0.0,
                "Next Multiplier Average": float(sum(values) / len(values)) if values else 0.0,
                "Highest Next Multiplier": float(max(values)) if values else 0.0,
                "Lowest Next Multiplier": float(min(values)) if values else 0.0,
            }
        )
    return pd.DataFrame(records)


def results_to_dataframe(rows: list[object]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": row.id,
                "session": row.session_id,
                "round": row.result_number,
                "timestamp": row.timestamp,
                "multiplier": float(row.multiplier),
                "result": row.result_type,
                "loss_streak": row.loss_streak_before,
                "win_streak": row.win_streak_before,
                "ocr_confidence": float(row.ocr_confidence or 0),
            }
            for row in rows
        ]
    )
