import random
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from agent import engine, service
from database.models import AgentLesson, AgentSignal, Base, TableCapture


def letters(n, seed=3, planted=False):
    rnd, seq = random.Random(seed), []
    for _ in range(n):
        if planted and seq[-3:] == list("BBB"):
            seq.append("S" if rnd.random() < 0.85 else "B")
        else:
            seq.append(rnd.choice("BS"))
    return seq


def test_segments_split_on_gaps_and_invalid_rows():
    items = [("B", True, None), ("S", True, 1), ("B", True, 3), ("S", False, 1), ("B", True, 1)]
    assert engine.build_segments(items) == [["B", "S"], ["B"], ["B"]]


def test_no_signal_before_enough_history():
    assert engine.issue_signal(engine.DEFAULT_STRATEGIES, [list("BSBSB")])["signal"] == "NO SIGNAL"


def test_walk_forward_never_peeks_ahead():
    seq = letters(300)
    s = engine.DEFAULT_STRATEGIES[0]
    full = engine.walk_forward(s, [seq])
    # Appending future data cannot change scores already earned on the earlier part.
    longer = engine.walk_forward(s, [seq + letters(200, seed=9)], 0.0, 300 / 500)
    assert longer["tested"] == full["tested"] and longer["hits"] == full["hits"]


def test_agent_finds_planted_pattern_and_signals_small():
    seq = letters(1200, planted=True)
    seq[-3:] = list("BBB")
    result = engine.issue_signal(engine.DEFAULT_STRATEGIES, [seq])
    assert result["signal"] == "SMALL" and result["agreement"] >= 70


def test_loss_reasons_cover_causes():
    strat = {"min_matches": 8, "min_agreement": 62.0}
    info = {"strategy": "Seq-3", "context": "Big Big Big", "matches": 9, "agreement": 66.0, "predicted": "SMALL"}
    cause, text, idea = engine.explain_loss(info, "BIG", [], strat)
    assert cause == "thin evidence" and "min_matches" in idea
    info.update(matches=40, agreement=90.0)
    assert engine.explain_loss(info, "BIG", [], strat)[0] == "normal variance"


def test_evolve_disables_strategy_that_is_provably_worse_than_chance():
    strategies = [{**engine.DEFAULT_STRATEGIES[0], "version": 1, "enabled": True}]
    new, log = engine.evolve(strategies, [letters(200)], {"Seq-3": {"tested": 60, "hits": 15}})
    assert new[0]["enabled"] is False and log[0]["kind"] == "DISABLE"


@pytest.fixture()
def db():
    eng = create_engine("sqlite://")
    Base.metadata.create_all(eng)
    with sessionmaker(eng)() as session:
        yield session


def feed(session, seq, start=20260101000):
    for i, c in enumerate(seq):
        number = 7 if c == "B" else 2
        session.add(TableCapture(period=str(start + i), number=number, size="Big" if c == "B" else "Small", colors="", confidence=Decimal("90"), data_status="Valid"))
    session.flush()


def rows_of(session):
    return list(session.scalars(select(TableCapture).order_by(TableCapture.period, TableCapture.id)))


def test_end_to_end_signals_settle_and_write_lessons(db):
    seq = letters(900, planted=True)
    feed(db, seq[:200])
    for i in range(200, 900):
        service.tick(db, rows_of(db), quality_ok=True)
        feed(db, [seq[i]], start=20260101000 + i)
    service.tick(db, rows_of(db), quality_ok=True)
    signals = list(db.scalars(select(AgentSignal)))
    settled = [s for s in signals if s.outcome in {"WIN", "LOSS"}]
    assert settled, "agent should have issued and settled signals"
    assert all(s.reason for s in settled)
    kinds = {l.kind for l in db.scalars(select(AgentLesson))}
    assert {"WIN", "LOSS", "REVIEW"} <= kinds
    wins = sum(s.outcome == "WIN" for s in settled)
    assert wins / len(settled) > 0.6          # planted pattern is learnable
    snap = service.snapshot(db, rows_of(db))
    assert snap["kpis"]["wins"] + snap["kpis"]["losses"] == len(settled)
    assert snap["records"][0]["period"] > snap["records"][-1]["period"]


def test_data_quality_gate_blocks_new_signals(db):
    feed(db, letters(300, planted=True))
    out = service.tick(db, rows_of(db), quality_ok=False, quality_note="paused")
    assert out["decision"]["signal"] == "NO SIGNAL"
    assert not list(db.scalars(select(AgentSignal)))


def test_diagnose_explains_every_blocker():
    seg = [list("BSBSBSBS" * 6)]
    seg[0][-3:] = list("BBB")
    rows = engine.diagnose(engine.DEFAULT_STRATEGIES, seg)
    assert len(rows) == 4 and all(r["blocker"] for r in rows)
    off = engine.diagnose([{**engine.DEFAULT_STRATEGIES[0], "enabled": False}], seg)
    assert "OFF" in off[0]["blocker"]


def test_upgrades_cannot_exceed_bounds():
    s = {**engine.DEFAULT_STRATEGIES[0], "min_matches": engine.MAX_MIN_MATCHES, "min_agreement": engine.MAX_MIN_AGREEMENT}
    for n in engine._neighbors(s):
        assert n["min_matches"] <= engine.MAX_MIN_MATCHES and n["min_agreement"] <= engine.MAX_MIN_AGREEMENT


def test_tick_reports_health_and_reasons_when_silent(db):
    feed(db, letters(300))
    out = service.tick(db, rows_of(db), quality_ok=True)
    d = out["decision"]
    assert d["health"]["results"] == 300 and len(d["diagnostics"]) == 4
