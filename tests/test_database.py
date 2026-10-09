from decimal import Decimal

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import Base, GameResult, SessionModel
from capture.duplicate_detector import DuplicateDetector


def test_duplicate_detector_same_visible_value() -> None:
    detector = DuplicateDetector(minimum_interval=1.0)
    assert detector.is_duplicate(Decimal("1.21"), now=100.0) is False
    assert detector.is_duplicate(Decimal("1.21"), now=100.2) is True
    assert detector.is_duplicate(Decimal("1.21"), now=100.8) is True


def test_duplicate_detector_new_result() -> None:
    detector = DuplicateDetector(minimum_interval=1.0)
    assert detector.is_duplicate(Decimal("1.21"), now=100.0) is False
    assert detector.is_duplicate(Decimal("2.43"), now=101.2) is False


def test_models_can_insert_results_in_memory() -> None:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session_model = SessionModel(session_name="test")
        session.add(session_model)
        session.flush()
        session.add(
            GameResult(
                session_id=session_model.id,
                result_number=1,
                multiplier=Decimal("1.21"),
                result_type="LOSS",
                loss_streak_before=1,
                win_streak_before=0,
                source="test",
                ocr_confidence=Decimal("96.00"),
                raw_ocr_text="x1.21",
                fingerprint="abc",
            )
        )
        session.commit()
        assert session.query(GameResult).count() == 1
