from decimal import Decimal

from ocr.validator import classify_multiplier, parse_first_multiplier


def test_classification_threshold() -> None:
    assert classify_multiplier(Decimal("1.49")) == "LOSS"
    assert classify_multiplier(Decimal("1.50")) == "WIN"
    assert classify_multiplier(Decimal("2.00")) == "WIN"


def test_ocr_parsing_common_forms() -> None:
    for text in ["x1.21", "1.21x", "1.21", "X 1.21"]:
        parsed = parse_first_multiplier(text)
        assert parsed is not None
        assert parsed.value == Decimal("1.21")
