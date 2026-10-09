from statistics.analytics_engine import (loss_streaks, neutral_drawdown, overview, patterns,
                                         percentile, reliability, walk_forward)


def test_classification_and_baseline() -> None:
    report = overview([1.49, 1.50, 2.0, 1.10], 1.5)
    assert (report["wins"], report["losses"], report["baseline_rate"]) == (2, 2, 50.0)


def test_exact_preceding_three_loss_streak_has_no_lookahead() -> None:
    rows = loss_streaks([1.20, 1.30, 1.10, 2.00], 1.5)
    three = rows[2]
    assert three["sample_size"] == 1 and three["win_count"] == 1
    # The first two loss targets have preceding streaks one and two, respectively.
    assert rows[0]["sample_size"] == 1 and rows[1]["sample_size"] == 1


def test_small_perfect_sample_is_not_confident() -> None:
    assert reliability(2) == "INSUFFICIENT SAMPLE"


def test_percentiles_and_pattern_target_exclusion() -> None:
    assert percentile([1, 2, 3, 4, 5], 90) == 4.6
    report = patterns([1.1, 1.1, 1.1, 2.0], 1.5, length=3)
    assert report[0]["pattern"] == "L-L-L" and report[0]["sample_size"] == 1
    assert report[0]["historical_rate"] == 100.0


def test_neutral_drawdown() -> None:
    assert neutral_drawdown([1, 1, -1, -1, -1, 1]) == {"final_neutral_score": 0, "maximum_neutral_drawdown": 3}


def test_walk_forward_needs_training_data_and_never_leaks() -> None:
    short = walk_forward([1.0] * 10, 1.5, initial_training_size=20)
    assert short["status"] == "NOT ENOUGH DATA"
    report = walk_forward([1.0, 2.0] * 30, 1.5, length=1, initial_training_size=20, test_window=10, min_sample=2, min_accuracy=0)
    assert report["status"] == "OK"
    assert all(item["test_start"] > item["training_rounds"] for item in report["windows"])
