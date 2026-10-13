"""Pure (database-free) core of the analyzer agent.

The agent looks at the *current scenario* (the last few Big/Small results, or
the current streak), finds every earlier moment in the captured history that
had the same scenario, and checks what followed. If enough past cases exist
and they agree strongly enough, it issues a BIG or SMALL signal for the next
period; otherwise it stays silent (NO SIGNAL).

Everything is evaluated walk-forward: a prediction only ever uses records that
existed *before* the period being predicted.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

DEFAULT_STRATEGIES: list[dict[str, Any]] = [
    {"name": "Seq-3", "kind": "seq", "k": 3, "min_matches": 8, "min_agreement": 62.0},
    {"name": "Seq-4", "kind": "seq", "k": 4, "min_matches": 6, "min_agreement": 65.0},
    {"name": "Seq-5", "kind": "seq", "k": 5, "min_matches": 5, "min_agreement": 70.0},
    {"name": "Streak", "kind": "streak", "k": 4, "min_matches": 10, "min_agreement": 62.0},
]

MIN_HISTORY = 30          # valid consecutive results needed before any signal
EVOLVE_EVERY = 10         # review strategies after this many settled signals
MIN_VALIDATION_TESTS = 20 # minimum out-of-sample tests to accept a change
DISABLE_MIN_TESTS = 40    # minimum tests before a strategy can be switched off


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson score interval for a win rate, as fractions."""
    if n <= 0:
        return None
    p = hits / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - m), min(1.0, c + m)


def build_segments(sizes_and_periods: Sequence[tuple[str, bool, int | None]]) -> list[list[str]]:
    """Split history into runs of consecutive valid periods.

    Each item is (size_letter 'B'/'S', is_valid, delta_from_previous_period).
    A segment ends at an invalid row or a period gap, so patterns never join
    results across missing data.
    """
    segments: list[list[str]] = []
    current: list[str] = []
    for letter, valid, delta in sizes_and_periods:
        if not valid:
            if current:
                segments.append(current)
            current = []
            continue
        if current and delta != 1:
            segments.append(current)
            current = []
        current.append(letter)
    if current:
        segments.append(current)
    return segments


def context_key(strategy: dict[str, Any], seg: Sequence[str], i: int):
    """Scenario description using only seg[:i] (everything before position i)."""
    k = int(strategy["k"])
    if strategy["kind"] == "seq":
        return tuple(seg[i - k:i]) if i >= k else None
    if i < 1:
        return None
    last, run = seg[i - 1], 1
    while i - 1 - run >= 0 and seg[i - 1 - run] == last:
        run += 1
    return (last, min(run, k))


def describe_context(strategy: dict[str, Any], key) -> str:
    if key is None:
        return "—"
    if strategy["kind"] == "seq":
        return " ".join("Big" if c == "B" else "Small" for c in key)
    return f"{'Big' if key[0] == 'B' else 'Small'} streak ≥{key[1]}" if key[1] >= int(strategy["k"]) else f"{'Big' if key[0] == 'B' else 'Small'} streak of {key[1]}"


def _predict(counts: dict, key, strategy: dict[str, Any]) -> dict[str, Any] | None:
    if key is None:
        return None
    big, small = counts.get(key, (0, 0))
    n = big + small
    if n < int(strategy["min_matches"]) or big == small:
        return None
    agreement = max(big, small) * 100 / n
    if agreement < float(strategy["min_agreement"]):
        return None
    # Significance gate: the matched cases must beat a coin flip with 95% confidence,
    # otherwise a lucky-looking pattern in a handful of cases would fire constantly.
    if wilson(max(big, small), n)[0] <= 0.5:
        return None
    return {"predicted": "BIG" if big > small else "SMALL", "matches": n, "agreement": round(agreement, 1), "big": big, "small": small}


def walk_forward(strategy: dict[str, Any], segments: Sequence[Sequence[str]], lo: float = 0.0, hi: float = 1.0) -> dict[str, int]:
    """Replay history in order; predict each step from earlier steps only.

    Only positions in the [lo, hi) fraction of the timeline are scored, which
    lets the evolver tune on early data and validate on later, unseen data.
    """
    total = sum(len(s) for s in segments)
    start, stop = int(total * lo), int(total * hi)
    counts: dict = {}
    position = tested = hits = 0
    for seg in segments:
        for i, letter in enumerate(seg):
            key = context_key(strategy, seg, i)
            if key is not None:
                if start <= position < stop:
                    pred = _predict(counts, key, strategy)
                    if pred:
                        tested += 1
                        hits += pred["predicted"][0] == letter
                big, small = counts.get(key, (0, 0))
                counts[key] = (big + (letter == "B"), small + (letter == "S"))
            position += 1
    return {"tested": tested, "hits": hits}


def build_counts(strategy: dict[str, Any], segments: Sequence[Sequence[str]]) -> dict:
    counts: dict = {}
    for seg in segments:
        for i, letter in enumerate(seg):
            key = context_key(strategy, seg, i)
            if key is not None:
                big, small = counts.get(key, (0, 0))
                counts[key] = (big + (letter == "B"), small + (letter == "S"))
    return counts


def edge_label(hits: int, tested: int) -> str:
    interval = wilson(hits, tested)
    if interval is None or tested < MIN_VALIDATION_TESTS:
        return "TOO FEW TESTS"
    if interval[0] > 0.5:
        return "EDGE FOUND"
    if interval[1] < 0.5:
        return "WORSE THAN CHANCE"
    return "NOT PROVEN"


def issue_signal(strategies: Sequence[dict[str, Any]], segments: Sequence[Sequence[str]]) -> dict[str, Any]:
    """Compare the current scenario with all earlier records and decide."""
    total = sum(len(s) for s in segments)
    if total < MIN_HISTORY or not segments or not segments[-1]:
        return {"signal": "NO SIGNAL", "status": f"Collecting history ({total}/{MIN_HISTORY} consecutive results)"}
    tail = list(segments[-1])
    candidates: list[dict[str, Any]] = []
    for strategy in strategies:
        if not strategy.get("enabled", True):
            continue
        key = context_key(strategy, tail, len(tail))
        # History for matching = every earlier record, including the open segment's own past.
        counts = build_counts(strategy, segments)
        # Remove the current open-ended position: counts only contain *completed* (context -> next) pairs.
        pred = _predict(counts, key, strategy)
        if not pred:
            continue
        wf = walk_forward(strategy, segments)
        interval = wilson(wf["hits"], wf["tested"])
        candidates.append({**pred, "strategy": strategy["name"], "version": strategy.get("version", 1), "context": describe_context(strategy, key),
                           "wf_tested": wf["tested"], "wf_hits": wf["hits"], "wf_lower": interval[0] if interval else 0.0, "edge": edge_label(wf["hits"], wf["tested"])})
    if not candidates:
        return {"signal": "NO SIGNAL", "status": "Current scenario has no strong enough match in past records"}
    candidates.sort(key=lambda c: (c["wf_lower"], c["agreement"], c["matches"]), reverse=True)
    best = candidates[0]
    agreeing = sum(c["predicted"] == best["predicted"] for c in candidates)
    return {"signal": best["predicted"], "status": "Pattern matched past records", "strategies_agreeing": agreeing, "strategies_firing": len(candidates), **best}


def explain_win(sig: dict[str, Any], actual: str) -> tuple[str, str]:
    cause = "pattern held"
    text = (f"WIN: {sig['strategy']} saw scenario “{sig['context']}” {sig['matches']} times before; {sig['agreement']}% were followed by {sig['predicted']}. "
            f"This time {actual} followed, so the pattern held.")
    if sig["matches"] < 15:
        text += " Sample was small, so this win may be luck rather than proof."
        cause = "pattern held (small sample)"
    return cause, text


def explain_loss(sig: dict[str, Any], actual: str, recent_hits: Sequence[bool], strategy: dict[str, Any]) -> tuple[str, str, str]:
    """Return (cause, explanation, upgrade idea) for a lost signal."""
    min_matches, min_agree = int(strategy["min_matches"]), float(strategy["min_agreement"])
    if sig["matches"] < min_matches * 2:
        cause = "thin evidence"
        idea = f"Require more past matches for {sig['strategy']} (min_matches {min_matches} → {min_matches + 2})."
        why = f"only {sig['matches']} past cases backed the signal"
    elif sig["agreement"] < min_agree + 8:
        cause = "weak agreement"
        idea = f"Require stronger agreement for {sig['strategy']} (min_agreement {min_agree:g}% → {min_agree + 3:g}%)."
        why = f"past cases agreed only {sig['agreement']}%, barely above the {min_agree:g}% cutoff"
    elif len(recent_hits) >= 5 and sum(recent_hits[-5:]) <= 2:
        cause = "strategy cooling"
        idea = f"Re-validate {sig['strategy']} on recent data; pause it if it stays below chance."
        why = f"{sig['strategy']} won only {sum(recent_hits[-5:])} of its last {len(recent_hits[-5:])} signals"
    else:
        cause = "normal variance"
        idea = "No rule change justified: a pattern that agreed %.0f%% of the time is still expected to miss about %.0f%% of the time." % (sig["agreement"], 100 - sig["agreement"])
        why = f"the pattern was strong ({sig['matches']} cases, {sig['agreement']}% agreement) but the other side came up"
    text = f"LOSS: predicted {sig['predicted']} from “{sig['context']}” but {actual} came — {why}."
    return cause, text, idea


def _neighbors(strategy: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for dm in (-2, 2):
        m = int(strategy["min_matches"]) + dm
        if 3 <= m <= 40:
            out.append({**strategy, "min_matches": m})
    for da in (-3.0, 3.0):
        a = float(strategy["min_agreement"]) + da
        if 52 <= a <= 90:
            out.append({**strategy, "min_agreement": a})
    return out


def evolve(strategies: Sequence[dict[str, Any]], segments: Sequence[Sequence[str]], live_stats: dict[str, dict[str, int]]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Review every strategy and return (new_strategies, log_entries).

    A parameter change is accepted only if it is better on the 30% most recent
    data that was NOT used to choose it, and clears a minimum sample size.
    Strategies that are demonstrably worse than chance are switched off.
    """
    updated: list[dict[str, Any]] = []
    log: list[dict[str, str]] = []
    for strategy in strategies:
        current = dict(strategy)
        name = current["name"]
        live = live_stats.get(name, {"tested": 0, "hits": 0})
        interval = wilson(live["hits"], live["tested"])
        if current.get("enabled", True) and live["tested"] >= DISABLE_MIN_TESTS and interval and interval[1] < 0.5:
            current["enabled"] = False
            log.append({"kind": "DISABLE", "strategy": name, "text": f"{name} switched OFF: {live['hits']}/{live['tested']} live wins, the 95% range ({interval[0]*100:.0f}–{interval[1]*100:.0f}%) is entirely below 50%."})
            updated.append(current)
            continue
        if not current.get("enabled", True):
            updated.append(current)
            continue
        base_train = walk_forward(current, segments, 0.0, 0.7)
        base_hold = walk_forward(current, segments, 0.7, 1.0)
        best, best_train_rate = None, (base_train["hits"] / base_train["tested"]) if base_train["tested"] else 0.0
        for candidate in _neighbors(current):
            train = walk_forward(candidate, segments, 0.0, 0.7)
            if train["tested"] >= MIN_VALIDATION_TESTS and train["hits"] / train["tested"] > best_train_rate + 0.01:
                best, best_train_rate = candidate, train["hits"] / train["tested"]
        if best is not None:
            hold = walk_forward(best, segments, 0.7, 1.0)
            base_rate = base_hold["hits"] / base_hold["tested"] if base_hold["tested"] else 0.0
            hold_rate = hold["hits"] / hold["tested"] if hold["tested"] else 0.0
            if hold["tested"] >= MIN_VALIDATION_TESTS and hold_rate >= base_rate + 0.02 and hold_rate > 0.5:
                new_version = int(current.get("version", 1)) + 1
                change = ", ".join(f"{k} {current[k]:g}→{best[k]:g}" for k in ("min_matches", "min_agreement") if best[k] != current[k])
                best["version"] = new_version
                log.append({"kind": "UPGRADE", "strategy": name, "text": f"{name} upgraded to v{new_version} ({change}). Unseen-data check: {hold_rate*100:.0f}% on {hold['tested']} tests vs {base_rate*100:.0f}% before."})
                updated.append(best)
                continue
            log.append({"kind": "NOCHANGE", "strategy": name, "text": f"{name}: tried a tweak but it did not hold up on unseen data ({hold['hits']}/{hold['tested']}); kept v{current.get('version', 1)}."})
        else:
            log.append({"kind": "NOCHANGE", "strategy": name, "text": f"{name}: no parameter change beat the current rules on training data; kept v{current.get('version', 1)}."})
        updated.append(current)
    return updated, log
