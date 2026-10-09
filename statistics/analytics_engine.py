"""Chronological historical analytics with no future-data leakage."""
from __future__ import annotations

import math
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from typing import Iterable

from config import analytics_config


def _number(value: object) -> float | None:
    try:
        number = float(Decimal(str(value)))
    except (InvalidOperation, ValueError, TypeError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def normalize_values(values: Iterable[object]) -> tuple[list[float], int]:
    """Keep valid positive values in supplied chronological order."""
    valid, invalid = [], 0
    for value in values:
        number = _number(value)
        if number is None:
            invalid += 1
        else:
            valid.append(number)
    return valid, invalid


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def reliability(sample: int) -> str:
    if sample < 20: return "INSUFFICIENT SAMPLE"
    if sample < 50: return "LOW SAMPLE"
    if sample < 100: return "MODERATE SAMPLE"
    if sample < 250: return "GOOD SAMPLE"
    return "LARGE SAMPLE"


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator * 100, 2) if denominator else None


def _target_stats(targets: list[float], threshold: float, baseline: float | None) -> dict:
    wins = sum(value >= threshold for value in targets)
    rate = _rate(wins, len(targets))
    return {
        "sample_size": len(targets), "win_count": wins, "loss_count": len(targets) - wins,
        "historical_rate": rate, "baseline_rate": baseline,
        "difference_points": round(rate - baseline, 2) if rate is not None and baseline is not None else None,
        "average_multiplier": round(_mean(targets), 4) if targets else None,
        "median_multiplier": round(_median(targets), 4) if targets else None,
        "p90_multiplier": round(percentile(targets, 90), 4) if targets else None,
        "p95_multiplier": round(percentile(targets, 95), 4) if targets else None,
        "minimum_multiplier": round(min(targets), 4) if targets else None,
        "maximum_multiplier": round(max(targets), 4) if targets else None,
        "reliability_status": "NO DATA" if not targets else reliability(len(targets)),
    }


def overview(values: Iterable[object], threshold: float) -> dict:
    valid, invalid = normalize_values(values)
    wins = sum(value >= threshold for value in valid)
    return {
        "total_rounds": len(valid), "invalid_rounds": invalid, "wins": wins, "losses": len(valid) - wins,
        "threshold": threshold, "baseline_rate": _rate(wins, len(valid)),
        "win_percentage": _rate(wins, len(valid)), "loss_percentage": _rate(len(valid) - wins, len(valid)),
        "minimum_multiplier": round(min(valid), 4) if valid else None,
        "maximum_multiplier": round(max(valid), 4) if valid else None,
        "average_multiplier": round(_mean(valid), 4) if valid else None,
        "median_multiplier": round(_median(valid), 4) if valid else None,
        "p90_multiplier": round(percentile(valid, 90), 4) if valid else None,
        "p95_multiplier": round(percentile(valid, 95), 4) if valid else None,
    }


def loss_streaks(values: Iterable[object], threshold: float, max_streak: int = 8) -> list[dict]:
    valid, _ = normalize_values(values); base = overview(valid, threshold)["baseline_rate"]
    buckets: dict[int, list[float]] = defaultdict(list)
    preceding = 0
    for target in valid:
        # preceding is derived solely from earlier rounds, before target is classified.
        if preceding:
            buckets[preceding].append(target)
        preceding = preceding + 1 if target < threshold else 0
    records = []
    for streak in range(1, max_streak + 1):
        record = _target_stats(buckets[streak], threshold, base)
        record["streak"] = streak
        records.append(record)
    return records


def patterns(values: Iterable[object], threshold: float, length: int | None = None) -> list[dict]:
    valid, _ = normalize_values(values); length = length or analytics_config.pattern_length
    baseline = overview(valid, threshold)["baseline_rate"]
    buckets: dict[str, list[float]] = defaultdict(list)
    labels = ["W" if value >= threshold else "L" for value in valid]
    for index in range(length, len(valid)):
        pattern = "-".join(labels[index - length:index])  # target is excluded
        buckets[pattern].append(valid[index])
    records = []
    for name in sorted(buckets):
        record = _target_stats(buckets[name], threshold, baseline); record["pattern"] = name; records.append(record)
    return records


def recent_comparison(values: Iterable[object], threshold: float) -> list[dict]:
    valid, _ = normalize_values(values); windows = [None, 500, 250, 100, 50]
    report = []
    for window in windows:
        slice_ = valid if window is None else valid[-window:]
        item = overview(slice_, threshold); item["window"] = "ALL DATA" if window is None else f"LAST {window}"
        report.append(item)
    return report


def neutral_drawdown(scores: Iterable[int]) -> dict:
    current = peak = maximum = 0
    for score in scores:
        current += score; peak = max(peak, current); maximum = max(maximum, peak - current)
    return {"final_neutral_score": current, "maximum_neutral_drawdown": maximum}


def walk_forward(values: Iterable[object], threshold: float, length: int | None = None,
                 initial_training_size: int | None = None, test_window: int | None = None,
                 min_sample: int | None = None, min_accuracy: float | None = None) -> dict:
    valid, invalid = normalize_values(values)
    length = length or analytics_config.pattern_length; initial_training_size = initial_training_size or analytics_config.initial_training_size
    test_window = test_window or analytics_config.test_window_size; min_sample = min_sample or analytics_config.min_pattern_sample
    min_accuracy = min_accuracy if min_accuracy is not None else analytics_config.min_historical_accuracy
    if len(valid) <= initial_training_size:
        return {"status": "NOT ENOUGH DATA", "required": initial_training_size, "available": len(valid), "invalid_rounds": invalid, "windows": [], "signals": 0}
    labels = ["W" if v >= threshold else "L" for v in valid]; all_signals = []; signal_targets = []; windows = []
    for test_start in range(initial_training_size, len(valid), test_window):
        test_end = min(test_start + test_window, len(valid)); training_end = test_start
        buckets: dict[str, list[float]] = defaultdict(list)
        for target_index in range(length, training_end):
            buckets["-".join(labels[target_index - length:target_index])].append(valid[target_index])
        window_signals = []
        for target_index in range(test_start, test_end):
            if target_index < length: continue
            prior_targets = buckets.get("-".join(labels[target_index - length:target_index]), [])
            rate = _rate(sum(v >= threshold for v in prior_targets), len(prior_targets))
            if len(prior_targets) >= min_sample and rate is not None and rate / 100 >= min_accuracy:
                correct = valid[target_index] >= threshold
                window_signals.append(correct); all_signals.append(correct); signal_targets.append(valid[target_index])
        baseline = _rate(sum(v >= threshold for v in valid[test_start:test_end]), test_end - test_start)
        windows.append({"training_rounds": training_end, "test_start": test_start + 1, "test_end": test_end,
                        "signals": len(window_signals), "correct": sum(window_signals), "incorrect": len(window_signals)-sum(window_signals),
                        "accuracy": _rate(sum(window_signals), len(window_signals)), "same_test_baseline": baseline})
    accuracy = _rate(sum(all_signals), len(all_signals))
    same_period = _rate(sum(value >= threshold for value in signal_targets), len(signal_targets))
    scores = [1 if correct else -1 for correct in all_signals]
    result = {"status": "OK", "training_window": initial_training_size, "test_window": test_window, "signals": len(all_signals),
              "correct": sum(all_signals), "incorrect": len(all_signals)-sum(all_signals), "walk_forward_accuracy": accuracy,
              "same_test_baseline": same_period,
              "difference_points": round(accuracy - same_period, 2) if accuracy is not None and same_period is not None else None,
              "windows": windows, "invalid_rounds": invalid}
    result.update(neutral_drawdown(scores)); return result
