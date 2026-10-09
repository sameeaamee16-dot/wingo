"""Transparent historical WinGo analysis with abstention and walk-forward scoring."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Sequence


WINDOWS = (5, 10, 20, 30, 50, 100)
ANALYSIS_WINDOW = 100


def classify_number(number: int) -> str:
    if not isinstance(number, int) or not 0 <= number <= 9:
        raise ValueError("WinGo result number must be an integer from 0 through 9")
    return "Small" if number <= 4 else "Big"


def period_delta(previous: Any, current: Any) -> int | None:
    try:
        a, b = str(previous).strip(), str(current).strip()
        if not (a.isdigit() and b.isdigit() and len(a) == len(b)):
            return None
        return int(b) - int(a)
    except (TypeError, ValueError):
        return None


def row_is_valid(row: Any) -> bool:
    number = getattr(row, "number", None)
    period = str(getattr(row, "period", ""))
    status = getattr(row, "data_status", None)
    if not isinstance(number, int) or not 0 <= number <= 9 or not period.isdigit() or not 10 <= len(period) <= 24:
        return False
    if getattr(row, "size", None) != classify_number(number):
        return False
    return status not in {"Invalid result", "Abnormal jump", "Duplicate"}


def _count_pct(count: int, total: int) -> float:
    return round(count * 100 / total, 1) if total else 0.0


def _runs(rows: Sequence[Any]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for row in rows:
        if not row_is_valid(row):
            continue
        if runs and period_delta(runs[-1]["last_period"], row.period) == 1 and runs[-1]["size"] == row.size:
            runs[-1]["length"] += 1
            runs[-1]["last_period"] = row.period
        else:
            runs.append({"size": row.size, "length": 1, "first_period": row.period, "last_period": row.period})
    return runs


def _snapshot(rows: Sequence[Any], total_valid: int | None = None) -> dict[str, Any]:
    chronological = list(rows)
    valid = [row for row in chronological if row_is_valid(row)]
    recent = valid[-ANALYSIS_WINDOW:]
    observed_total = len(valid) if total_valid is None else total_valid
    size_values = [row.size for row in recent]
    big_total, small_total = size_values.count("Big"), size_values.count("Small")
    windows: dict[str, dict[str, Any]] = {}
    for width in WINDOWS:
        window_rows = recent[-width:]
        values = [row.size for row in window_rows]
        if not window_rows:
            windows[str(width)] = {"available": False, "sample_size": 0, "big": 0, "small": 0, "big_percentage": 0.0, "small_percentage": 0.0, "number_frequency": {str(n): {"count": 0, "percentage": 0.0} for n in range(10)}, "odd": 0, "even": 0, "high": 0, "low": 0, "alternation_percentage": 0.0, "repetition_percentage": 0.0, "pair_sample_size": 0, "current_streak": {"type": None, "length": 0}, "longest_streak": {"Big": 0, "Small": 0}}
            continue
        big, small = values.count("Big"), values.count("Small")
        numbers = [row.number for row in window_rows]
        odds = sum(number % 2 for number in numbers)
        pairs_in_window = [(a, b) for a, b in zip(window_rows, window_rows[1:]) if period_delta(a.period, b.period) == 1]
        alternations_in_window = sum(a.size != b.size for a, b in pairs_in_window)
        runs_in_window = _runs(window_rows)
        current_run = runs_in_window[-1] if runs_in_window else None
        windows[str(width)] = {
            "available": len(values) == width,
            "sample_size": len(values),
            "big": big,
            "small": small,
            "big_percentage": _count_pct(big, len(values)),
            "small_percentage": _count_pct(small, len(values)),
            "number_frequency": {str(n): {"count": numbers.count(n), "percentage": _count_pct(numbers.count(n), len(numbers))} for n in range(10)},
            "odd": odds,
            "even": len(numbers) - odds,
            "high": sum(number >= 5 for number in numbers),
            "low": sum(number < 5 for number in numbers),
            "alternation_percentage": _count_pct(alternations_in_window, len(pairs_in_window)),
            "repetition_percentage": _count_pct(len(pairs_in_window) - alternations_in_window, len(pairs_in_window)),
            "pair_sample_size": len(pairs_in_window),
            "current_streak": {"type": current_run["size"].upper(), "length": current_run["length"]} if current_run else {"type": None, "length": 0},
            "longest_streak": {size: max((run["length"] for run in runs_in_window if run["size"] == size), default=0) for size in ("Big", "Small")},
        }

    number_values = [row.number for row in recent]
    number_counts = Counter(number_values)
    number_frequency = {str(n): {"count": number_counts[n], "percentage": _count_pct(number_counts[n], len(number_values))} for n in range(10)}
    odd = sum(number % 2 for number in number_values)
    even = len(number_values) - odd
    high = sum(number >= 5 for number in number_values)
    low = len(number_values) - high

    runs = _runs(recent)
    current = runs[-1] if runs else None
    longest = {size: max((run["length"] for run in runs if run["size"] == size), default=0) for size in ("Big", "Small")}
    pairs = [(a, b) for a, b in zip(recent, recent[1:]) if period_delta(a.period, b.period) == 1]
    alternations = sum(a.size != b.size for a, b in pairs)
    repetitions = len(pairs) - alternations

    transition_counts: dict[str, Counter[str]] = {"Big": Counter(), "Small": Counter()}
    for a, b in pairs:
        transition_counts[a.size][b.size] += 1
    transition = {}
    for previous in ("Big", "Small"):
        total = sum(transition_counts[previous].values())
        transition[previous] = {
            "sample_size": total,
            "next_big": transition_counts[previous]["Big"],
            "next_small": transition_counts[previous]["Small"],
            "big_percentage": _count_pct(transition_counts[previous]["Big"], total),
            "small_percentage": _count_pct(transition_counts[previous]["Small"], total),
        }

    sequence_counts: dict[str, dict[str, Any]] = {}
    for start in range(len(recent)):
        for length in range(2, min(5, len(recent) - start) + 1):
            seq = recent[start:start + length]
            if any(period_delta(seq[j].period, seq[j + 1].period) != 1 for j in range(len(seq) - 1)):
                continue
            key = " ".join(row.size for row in seq)
            entry = sequence_counts.setdefault(key, {"occurrences": 0, "next_big": 0, "next_small": 0})
            entry["occurrences"] += 1
            next_index = start + length
            if next_index < len(recent) and period_delta(recent[next_index - 1].period, recent[next_index].period) == 1:
                entry[f"next_{recent[next_index].size.lower()}"] += 1
    sequences = []
    for key, counts in sequence_counts.items():
        followed = counts["next_big"] + counts["next_small"]
        quality = "LOW SAMPLE" if followed < 5 else "LIMITED" if followed < 20 else "MORE USEFUL"
        sequences.append({"pattern": key, **counts, "followed_count": followed, "next_big_percentage": _count_pct(counts["next_big"], followed), "next_small_percentage": _count_pct(counts["next_small"], followed), "quality": quality})
    sequences.sort(key=lambda item: (-item["followed_count"], -item["occurrences"], item["pattern"]))

    components: list[dict[str, Any]] = []
    weights = {"10": 0.30, "20": 0.25, "50": 0.20}
    for width, weight in weights.items():
        record = windows[width]
        if record["available"]:
            components.append({"name": f"Last {width}", "big_rate": record["big_percentage"], "sample_size": record["sample_size"], "weight": weight})

    latest_size = recent[-1].size if recent else None
    latest_transition = transition.get(latest_size, {}) if latest_size else {}
    if latest_transition.get("sample_size", 0) >= 20:
        components.append({"name": f"After {latest_size}", "big_rate": latest_transition["big_percentage"], "sample_size": latest_transition["sample_size"], "weight": 0.15})

    sequence_component = None
    for length in range(min(5, len(recent)), 1, -1):
        suffix = recent[-length:]
        if any(period_delta(suffix[j].period, suffix[j + 1].period) != 1 for j in range(length - 1)):
            continue
        key = " ".join(row.size for row in suffix)
        record = sequence_counts.get(key)
        followed = record["next_big"] + record["next_small"] if record else 0
        if followed >= 20:
            sequence_component = {"name": f"Sequence {key}", "big_rate": _count_pct(record["next_big"], followed), "sample_size": followed, "weight": 0.10}
            components.append(sequence_component)
            break

    weighted_delta = sum((item["big_rate"] / 100 - 0.5) * 2 * item["weight"] for item in components)
    weight_total = sum(item["weight"] for item in components)
    for item in components:
        item["applied_weight"] = item["weight"] / weight_total if weight_total else 0.0
    big_score = 50 + (50 * weighted_delta / weight_total if weight_total else 0)
    small_score = 100 - big_score
    for item in components:
        rate = item["big_rate"]
        item["direction"] = "BIG" if rate >= 55 else "SMALL" if rate <= 45 else "NEUTRAL"
    directional = [item for item in components if item["direction"] != "NEUTRAL"]
    directions = {item["direction"] for item in directional}
    conflicts = len(directions) > 1
    chosen = "BIG" if big_score > small_score else "SMALL" if small_score > big_score else "NO CLEAR SIGNAL"
    agreeing = sum(item["direction"] == chosen for item in directional) if chosen in {"BIG", "SMALL"} else 0
    agreement = _count_pct(agreeing, len(directional)) if directional else 0.0

    statuses = [getattr(row, "data_status", None) for row in chronological[-ANALYSIS_WINDOW:]]
    low_confidence = sum(getattr(row, "confidence", None) is not None and float(row.confidence) < 65 for row in chronological[-ANALYSIS_WINDOW:])
    quality_flags = []
    if any(status in {"Missing period", "Abnormal jump", "Invalid result", "Duplicate"} for status in statuses):
        quality_flags.append("period or result validation issue")
    if low_confidence:
        quality_flags.append(f"{low_confidence} low-confidence OCR row(s)")
    if observed_total < 20:
        quality_flags.append("fewer than 20 valid results")
    data_quality = "LOW" if quality_flags else "GOOD"

    # Two separate, deterministic analysis agents: one evaluates rolling
    # distribution; the other evaluates transitions and matching sequences.
    history_windows = [(windows[str(width)], weight) for width, weight in ((10, 0.35), (20, 0.30), (50, 0.20), (100, 0.15)) if windows[str(width)]["sample_size"] >= 10]
    history_weight = sum(weight for _, weight in history_windows)
    history_big_score = (sum(window["big_percentage"] * weight for window, weight in history_windows) / history_weight) if history_weight else 50.0
    history_agent_signal = "BIG" if observed_total >= 20 and history_big_score >= 55 else "SMALL" if observed_total >= 20 and history_big_score <= 45 else "NO CLEAR SIGNAL"

    pattern_evidence = []
    if latest_transition.get("sample_size", 0) >= 10:
        pattern_evidence.append((latest_transition["big_percentage"], 0.4, latest_transition["sample_size"]))
    if sequence_component:
        pattern_evidence.append((sequence_component["big_rate"], 0.6, sequence_component["sample_size"]))
    pattern_weight = sum(weight for _, weight, _ in pattern_evidence)
    pattern_big_score = (sum(rate * weight for rate, weight, _ in pattern_evidence) / pattern_weight) if pattern_weight else 50.0
    pattern_sample = sum(sample for _, _, sample in pattern_evidence)
    pattern_agent_signal = "BIG" if pattern_sample >= 10 and pattern_big_score >= 55 else "SMALL" if pattern_sample >= 10 and pattern_big_score <= 45 else "NO CLEAR SIGNAL"
    agents = [
        {"name": "History Agent", "focus": "Rolling Big/Small distribution across 10, 20, 50, and 100 results", "signal": history_agent_signal, "big_score": round(history_big_score, 1), "small_score": round(100 - history_big_score, 1), "sample_size": sum(window["sample_size"] for window, _ in history_windows), "evidence_count": len(history_windows)},
        {"name": "Pattern Agent", "focus": "Next-result transitions and matching sequence continuations", "signal": pattern_agent_signal, "big_score": round(pattern_big_score, 1), "small_score": round(100 - pattern_big_score, 1), "sample_size": pattern_sample, "evidence_count": len(pattern_evidence)},
    ]

    max_strength = "VERY LOW" if observed_total < 6 else "LOW" if observed_total < 20 else "MEDIUM" if observed_total < 50 else "HIGH"
    strength_by_agreement = "VERY LOW" if agreement < 50 else "LOW" if agreement < 60 else "MEDIUM" if agreement < 70 else "HIGH"
    rank = {"VERY LOW": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
    strength = min((max_strength, strength_by_agreement), key=lambda label: rank[label])
    if history_agent_signal in {"BIG", "SMALL"} and pattern_agent_signal in {"BIG", "SMALL"} and history_agent_signal != pattern_agent_signal:
        signal, status = "NO CLEAR SIGNAL", "AGENTS DISAGREE"
    elif data_quality == "LOW":
        signal, status = "NO CLEAR SIGNAL", "DATA QUALITY LOW"
    elif len(valid) < 20 or history_agent_signal == "NO CLEAR SIGNAL" or pattern_agent_signal == "NO CLEAR SIGNAL":
        signal, status = "NO CLEAR SIGNAL", "INSUFFICIENT EVIDENCE"
    elif len(directional) < 2 or agreement < 70 or abs(big_score - small_score) < 8:
        signal, status = "NO CLEAR SIGNAL", "INSUFFICIENT EVIDENCE"
    else:
        signal, status = history_agent_signal, "TWO-AGENT CONSENSUS"

    latest_period = chronological[-1].period if chronological else None
    next_period = None
    if latest_period and str(latest_period).isdigit():
        next_value = str(int(latest_period) + 1)
        if len(next_value) == len(str(latest_period)):
            next_period = next_value

    recent10 = windows["10"]
    recent_trend = "BALANCED" if not recent10["available"] or abs(recent10["big_percentage"] - 50) <= 10 else "BIG" if recent10["big_percentage"] > 50 else "SMALL"
    long_window = windows["100"] if windows["100"]["available"] else windows["50"]
    long_term = "BALANCED" if not long_window["available"] or abs(long_window["big_percentage"] - 50) <= 5 else "BIG" if long_window["big_percentage"] > 50 else "SMALL"
    return {
        "period": chronological[-1].period if chronological else None,
        "next_period": next_period,
        "last_result": chronological[-1].number if chronological else None,
        "last_size": classify_number(chronological[-1].number) if chronological and row_is_valid(chronological[-1]) else None,
        "last10": [{"period": row.period, "number": row.number, "size": row.size.upper()} for row in recent[-10:]],
        "windows": windows,
        "number_frequency": number_frequency,
        "odd_even": {"odd": odd, "even": even, "sample_size": len(number_values)},
        "high_low": {"high": high, "low": low, "sample_size": len(number_values)},
        "current_streak": {"type": current["size"].upper(), "length": current["length"]} if current else {"type": None, "length": 0},
        "longest_streak": {"big": longest["Big"], "small": longest["Small"]},
        "recent_runs": [{"size": run["size"].upper(), "length": run["length"]} for run in runs[-20:]],
        "alternation_percentage": _count_pct(alternations, len(pairs)),
        "repetition_percentage": _count_pct(repetitions, len(pairs)),
        "transition": transition,
        "sequence_patterns": sequences[:40],
        "components": components,
        "agents": agents,
        "big_score": round(big_score, 1),
        "small_score": round(small_score, 1),
        "score_difference": round(abs(big_score - small_score), 1),
        "signal_agreement": agreement,
        "signal": signal,
        "signal_strength": strength,
        "signal_status": status,
        "historical_signal_strength": "LOW" if signal == "NO CLEAR SIGNAL" else strength,
        "recent_trend": recent_trend,
        "long_term_trend": long_term,
        "sample_size": observed_total,
        "data_quality": "LOW" if quality_flags else "GOOD",
        "data_quality_flags": quality_flags,
        "latest_data_status": getattr(chronological[-1], "data_status", "Waiting for result") if chronological else "Waiting for result",
        "analysis_label": "Historical pattern analysis; not a guaranteed forecast.",
    }


def analyze_history(rows: Sequence[Any]) -> dict[str, Any]:
    """Analyze captured results and run strict one-step-ahead walk-forward scoring."""
    chronological = list(rows)
    current = _snapshot(chronological)
    backtest_records: list[dict[str, Any]] = []
    scored: list[dict[str, Any]] = []
    baseline_hits = 0
    baseline_total = 0
    by_window: dict[str, list[bool]] = {str(width): [] for width in WINDOWS}
    during_streak: list[bool] = []
    during_alternation: list[bool] = []
    conflicting_rounds = 0
    valid_prefix = [0]
    big_prefix = [0]
    small_prefix = [0]
    for row in chronological:
        valid = row_is_valid(row)
        valid_prefix.append(valid_prefix[-1] + int(valid))
        big_prefix.append(big_prefix[-1] + int(valid and row.size == "Big"))
        small_prefix.append(small_prefix[-1] + int(valid and row.size == "Small"))
    for index in range(1, len(chronological)):
        source, target = chronological[index - 1], chronological[index]
        if not row_is_valid(source) or not row_is_valid(target) or period_delta(source.period, target.period) != 1:
            continue
        prefix = chronological[max(0, index - ANALYSIS_WINDOW):index]
        forecast = _snapshot(prefix, total_valid=valid_prefix[index])
        signal = forecast["signal"]
        actual = classify_number(target.number).upper()
        correct = signal == actual if signal in {"BIG", "SMALL"} else None
        if forecast["signal_status"] == "SIGNALS CONFLICTING":
            conflicting_rounds += 1
        big_before = big_prefix[index]
        small_before = small_prefix[index]
        baseline = "BIG" if big_before >= small_before else "SMALL"
        baseline_hits += baseline == actual
        baseline_total += 1
        record = {"analysis_period": source.period, "actual_period": target.period, "actual_number": target.number, "signal": signal, "actual_size": actual, "outcome": "NO SIGNAL" if correct is None else "WIN" if correct else "LOSS", "correct": correct, "signal_strength": forecast["signal_strength"], "sample_size": forecast["sample_size"], "big_score": forecast["big_score"], "small_score": forecast["small_score"], "data_quality": forecast["data_quality"], "backtest_accuracy": _count_pct(sum(item["correct"] is True for item in scored), len(scored)) if scored else None, "created_at": getattr(source, "captured_at", None)}
        backtest_records.append(record)
        if correct is not None:
            scored.append(record)
            if forecast["current_streak"]["length"] >= 3:
                during_streak.append(correct)
            if forecast["alternation_percentage"] >= 70:
                during_alternation.append(correct)
        for width in WINDOWS:
            window = forecast["windows"][str(width)]
            if window["available"] and window["big"] != window["small"]:
                tendency = "BIG" if window["big"] > window["small"] else "SMALL"
                by_window[str(width)].append(tendency == actual)

    correct_count = sum(record["correct"] is True for record in scored)
    incorrect_count = len(scored) - correct_count
    by_strength = {}
    for strength in ("VERY LOW", "LOW", "MEDIUM", "HIGH"):
        group = [record for record in scored if record["signal_strength"] == strength]
        wins = sum(record["correct"] is True for record in group)
        by_strength[strength] = {"tested": len(group), "correct": wins, "incorrect": len(group) - wins, "accuracy": _count_pct(wins, len(group))}
    return {
        **current,
        "backtest": {
            "tested": len(scored),
            "correct": correct_count,
            "incorrect": incorrect_count,
            "accuracy": _count_pct(correct_count, len(scored)),
            "by_signal_strength": by_strength,
            "by_window": {width: {"tested": len(values), "correct": sum(values), "incorrect": len(values) - sum(values), "accuracy": _count_pct(sum(values), len(values))} for width, values in by_window.items()},
            "during_streak": {"tested": len(during_streak), "correct": sum(during_streak), "incorrect": len(during_streak) - sum(during_streak), "accuracy": _count_pct(sum(during_streak), len(during_streak))},
            "during_alternation": {"tested": len(during_alternation), "correct": sum(during_alternation), "incorrect": len(during_alternation) - sum(during_alternation), "accuracy": _count_pct(sum(during_alternation), len(during_alternation))},
            "conflicting_rounds": conflicting_rounds,
            "majority_baseline_tested": baseline_total,
            "majority_baseline_correct": baseline_hits,
            "majority_baseline_accuracy": _count_pct(baseline_hits, baseline_total),
        },
        "backtest_records": backtest_records,
    }
