"""Database glue for the analyzer agent: issue, settle, learn, upgrade."""

from __future__ import annotations

import json
from datetime import timezone
from decimal import Decimal
from typing import Any, Sequence

from sqlalchemy import select

from agent import engine
from database.models import AgentLesson, AgentSignal, AgentStrategy, utc_now
from statistics.wingo_analyzer import period_delta, row_is_valid


def load_strategies(db) -> list[dict[str, Any]]:
    rows = list(db.scalars(select(AgentStrategy).order_by(AgentStrategy.id)))
    if not rows:
        for params in engine.DEFAULT_STRATEGIES:
            db.add(AgentStrategy(name=params["name"], version=1, params=json.dumps(params), enabled=1))
        db.flush()
        rows = list(db.scalars(select(AgentStrategy).order_by(AgentStrategy.id)))
    return [{**json.loads(r.params), "name": r.name, "version": r.version, "enabled": bool(r.enabled)} for r in rows]


def reset_agent(db) -> None:
    for model in (AgentLesson, AgentSignal, AgentStrategy):
        for row in db.scalars(select(model)):
            db.delete(row)
    db.flush()


def _segments(rows: Sequence[Any]) -> list[list[str]]:
    items, previous = [], None
    for row in rows:
        valid = row_is_valid(row)
        delta = period_delta(previous.period, row.period) if previous is not None else None
        items.append((row.size[0].upper(), valid, delta))
        previous = row if valid else None
    return engine.build_segments(items)


def _next_period(rows: Sequence[Any]) -> str | None:
    if not rows or not str(rows[-1].period).isdigit():
        return None
    value = str(int(rows[-1].period) + 1)
    return value if len(value) == len(str(rows[-1].period)) else None


def _live_stats(db) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {}
    for sig in db.scalars(select(AgentSignal).where(AgentSignal.outcome.in_(["WIN", "LOSS"]))):
        entry = stats.setdefault(sig.strategy_name, {"tested": 0, "hits": 0})
        entry["tested"] += 1
        entry["hits"] += sig.outcome == "WIN"
    return stats


def _settle(db, rows: Sequence[Any], strategies: list[dict[str, Any]]) -> None:
    by_period = {r.period: r for r in rows}
    latest = rows[-1].period if rows else None
    strategy_map = {s["name"]: s for s in strategies}
    pending = list(db.scalars(select(AgentSignal).where(AgentSignal.outcome.is_(None)).order_by(AgentSignal.id)))
    for sig in pending:
        row = by_period.get(sig.target_period)
        if row is None:
            if latest and period_delta(sig.target_period, latest) not in (None,) and period_delta(sig.target_period, latest) > 0:
                sig.outcome, sig.resolved_at = "SKIPPED", utc_now()
                sig.reason = "SKIPPED: the target period was never captured, so this signal cannot be judged."
            continue
        if not row_is_valid(row):
            sig.outcome, sig.resolved_at, sig.reason = "SKIPPED", utc_now(), "SKIPPED: the captured result failed validation."
            continue
        actual = row.size.upper()
        sig.actual_number, sig.actual_size, sig.resolved_at = row.number, row.size, utc_now()
        sig.outcome = "WIN" if actual == sig.predicted else "LOSS"
        info = {"strategy": sig.strategy_name, "context": sig.context, "matches": sig.matches, "agreement": float(sig.agreement), "predicted": sig.predicted}
        if sig.outcome == "WIN":
            cause, text = engine.explain_win(info, actual)
            idea = None
        else:
            recent = [s.outcome == "WIN" for s in db.scalars(select(AgentSignal).where(AgentSignal.strategy_name == sig.strategy_name, AgentSignal.outcome.in_(["WIN", "LOSS"])).order_by(AgentSignal.id))]
            cause, text, idea = engine.explain_loss(info, actual, recent, strategy_map.get(sig.strategy_name, engine.DEFAULT_STRATEGIES[0]))
        sig.reason = text
        db.flush()
        db.add(AgentLesson(signal_id=sig.id, kind=sig.outcome, strategy_name=sig.strategy_name, cause=cause, text=text, upgrade_idea=idea))
    db.flush()


def _maybe_evolve(db, strategies: list[dict[str, Any]], segments: list[list[str]]) -> None:
    last_review = db.scalar(select(AgentLesson.created_at).where(AgentLesson.kind == "REVIEW").order_by(AgentLesson.id.desc()).limit(1))
    query = select(AgentSignal).where(AgentSignal.outcome.in_(["WIN", "LOSS"]))
    if last_review is not None:
        query = query.where(AgentSignal.resolved_at > last_review)
    settled_since = len(list(db.scalars(query)))
    if settled_since < engine.EVOLVE_EVERY:
        return
    new_strategies, log = engine.evolve(strategies, segments, _live_stats(db))
    stored = {r.name: r for r in db.scalars(select(AgentStrategy))}
    for strategy in new_strategies:
        row = stored[strategy["name"]]
        row.version, row.enabled = int(strategy.get("version", 1)), int(bool(strategy.get("enabled", True)))
        row.params = json.dumps({k: strategy[k] for k in ("name", "kind", "k", "min_matches", "min_agreement")})
    for entry in log:
        db.add(AgentLesson(kind=entry["kind"], strategy_name=entry["strategy"], text=entry["text"]))
    changed = sum(e["kind"] in {"UPGRADE", "DISABLE"} for e in log)
    db.add(AgentLesson(kind="REVIEW", text=f"Strategy review after {settled_since} settled signals: {changed} change(s) applied, {len(log) - changed} kept as-is."))
    db.flush()


def tick(db, rows: Sequence[Any], quality_ok: bool, quality_note: str = "") -> dict[str, Any]:
    """Settle finished signals, learn from them, issue the next signal."""
    strategies = load_strategies(db)
    segments = _segments(rows)
    if rows:
        _settle(db, rows, strategies)
        _maybe_evolve(db, strategies, segments)
        strategies = load_strategies(db)

    target = _next_period(rows)
    decision: dict[str, Any] = {"signal": "NO SIGNAL", "status": "Waiting for captured results"}
    if rows:
        latest_ok = row_is_valid(rows[-1]) and any(float(r.confidence or 100) >= 65 for r in rows[-1:])
        existing = db.scalar(select(AgentSignal).where(AgentSignal.target_period == target)) if target else None
        if not quality_ok or not latest_ok:
            decision = {"signal": "NO SIGNAL", "status": quality_note or "Data quality too low to signal"}
        elif existing is not None:
            decision = {"signal": existing.predicted, "status": "Pattern matched past records", "strategy": existing.strategy_name, "version": existing.strategy_version,
                        "context": existing.context, "matches": existing.matches, "agreement": float(existing.agreement)}
        else:
            decision = engine.issue_signal(strategies, segments)
            if decision["signal"] in {"BIG", "SMALL"} and target:
                db.add(AgentSignal(target_period=target, strategy_name=decision["strategy"], strategy_version=int(decision["version"]), predicted=decision["signal"],
                                   context=decision["context"][:120], matches=decision["matches"], agreement=Decimal(str(decision["agreement"]))))
                db.flush()
        if decision["signal"] in {"BIG", "SMALL"}:
            chosen = next((s for s in strategies if s["name"] == decision.get("strategy")), None)
            if chosen:
                wf = engine.walk_forward(chosen, segments)
                decision.update(wf_tested=wf["tested"], wf_hits=wf["hits"], edge=engine.edge_label(wf["hits"], wf["tested"]))
    age = None
    if rows and getattr(rows[-1], "captured_at", None) is not None:
        captured = rows[-1].captured_at
        captured = captured if captured.tzinfo else captured.replace(tzinfo=timezone.utc)
        age = int((utc_now() - captured).total_seconds())
    decision["health"] = {"results": len(rows), "clean_tail": len(segments[-1]) if segments else 0, "seconds_since_last_capture": age,
                          "quality_ok": bool(quality_ok), "latest_confidence": float(rows[-1].confidence) if rows and rows[-1].confidence is not None else None}
    decision["diagnostics"] = engine.diagnose(strategies, segments) if rows else []
    return {"target_period": target, "decision": decision, "strategies": strategies}


def snapshot(db, rows: Sequence[Any], limit: int = 120) -> dict[str, Any]:
    """KPIs, bottom record log, and learning log for the UI."""
    signals = list(db.scalars(select(AgentSignal).order_by(AgentSignal.id)))
    by_period = {s.target_period: s for s in signals}
    wins = sum(s.outcome == "WIN" for s in signals)
    losses = sum(s.outcome == "LOSS" for s in signals)
    judged = wins + losses
    interval = engine.wilson(wins, judged)
    lessons = list(db.scalars(select(AgentLesson).order_by(AgentLesson.id.desc()).limit(60)))
    records = []
    for row in reversed(list(rows)[-limit:]):
        sig = by_period.get(row.period)
        records.append({"period": row.period, "number": row.number, "size": row.size.upper(),
                        "signal": sig.predicted if sig else "—", "outcome": (sig.outcome or "PENDING") if sig else "—",
                        "strategy": sig.strategy_name if sig else "", "reason": sig.reason if sig and sig.reason else ""})
    return {
        "kpis": {"signals": len(signals), "wins": wins, "losses": losses, "pending": sum(s.outcome is None for s in signals), "skipped": sum(s.outcome == "SKIPPED" for s in signals),
                 "win_rate": round(wins * 100 / judged, 1) if judged else None,
                 "interval": [round(interval[0] * 100, 1), round(interval[1] * 100, 1)] if interval else None,
                 "verdict": engine.edge_label(wins, judged)},
        "records": records,
        "lessons": [{"kind": l.kind, "strategy": l.strategy_name or "", "text": l.text, "idea": l.upgrade_idea or "", "time": l.created_at.astimezone().strftime("%d %b %H:%M")} for l in lessons],
    }
