from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP


MULTIPLIER_RE = re.compile(r"(?i)(?:x\s*)?(\d{1,5}(?:[.,]\d{1,2})?)(?:\s*x)?")


@dataclass(frozen=True)
class ParsedMultiplier:
    value: Decimal
    raw: str


def parse_multipliers(text: str, minimum: Decimal = Decimal("1.00"), maximum: Decimal = Decimal("10000.00")) -> list[ParsedMultiplier]:
    values: list[ParsedMultiplier] = []
    normalized_text = text.replace("|", "1").replace("O", "0").replace("o", "0")
    for match in MULTIPLIER_RE.finditer(normalized_text):
        token = match.group(1).replace(",", ".")
        try:
            value = Decimal(token).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        except (InvalidOperation, ValueError):
            continue
        if minimum <= value <= maximum:
            values.append(ParsedMultiplier(value=value, raw=match.group(0).strip()))
    return values


def parse_first_multiplier(text: str, minimum: Decimal = Decimal("1.00"), maximum: Decimal = Decimal("10000.00")) -> ParsedMultiplier | None:
    values = parse_multipliers(text, minimum=minimum, maximum=maximum)
    return values[0] if values else None


def classify_multiplier(multiplier: Decimal, threshold: Decimal = Decimal("1.50")) -> str:
    return "LOSS" if multiplier < threshold else "WIN"
