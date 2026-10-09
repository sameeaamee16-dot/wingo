from decimal import Decimal

from ocr.validator import parse_multipliers


def test_parse_history_values() -> None:
    parsed = parse_multipliers("1.35 1.18 1 1 6.75 6.7 1.14 1.74")
    assert [item.value for item in parsed] == [
        Decimal("1.35"),
        Decimal("1.18"),
        Decimal("1.00"),
        Decimal("1.00"),
        Decimal("6.75"),
        Decimal("6.70"),
        Decimal("1.14"),
        Decimal("1.74"),
    ]
