from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal

import pytesseract
from PIL import Image

from config import config
from ocr.preprocessing import preprocess_fast_variants, preprocess_variants
from ocr.tesseract_config import configure_tesseract, tesseract_help_message
from ocr.validator import parse_multipliers

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OCRResult:
    multiplier: Decimal | None
    raw_text: str
    confidence: float
    all_multipliers: list[Decimal]


class MultiplierReader:
    def __init__(self, maximum: Decimal | None = None) -> None:
        self.maximum = maximum or config.max_sanity_multiplier
        configure_tesseract()

    def read_center(self, image: Image.Image, fast: bool = False) -> OCRResult:
        candidates = self._read_candidates(image, fast=fast)
        if not candidates:
            return OCRResult(None, "", 0.0, [])
        best = max(candidates, key=lambda item: item.confidence)
        return best

    def read_history(self, image: Image.Image, fast: bool = True) -> OCRResult:
        candidates = self._read_candidates(image, all_values=True, fast=fast)
        if not candidates and fast:
            candidates = self._read_candidates(image, all_values=True, fast=False)
        if not candidates:
            return OCRResult(None, "", 0.0, [])
        best = max(candidates, key=lambda item: (len(item.all_multipliers), item.confidence))
        first = best.all_multipliers[0] if best.all_multipliers else None
        return OCRResult(first, best.raw_text, best.confidence, best.all_multipliers)

    def _read_candidates(self, image: Image.Image, all_values: bool = False, fast: bool = False) -> list[OCRResult]:
        results: list[OCRResult] = []
        variants = preprocess_fast_variants(image) if fast else preprocess_variants(image)
        for variant in variants:
            try:
                data = pytesseract.image_to_data(
                    variant,
                    config="--psm 6 -c tessedit_char_whitelist=xX0123456789., ",
                    output_type=pytesseract.Output.DICT,
                )
            except pytesseract.TesseractNotFoundError:
                raise RuntimeError(tesseract_help_message())
            words = [word for word in data.get("text", []) if word and word.strip()]
            confidences = []
            for conf in data.get("conf", []):
                try:
                    value = float(conf)
                except ValueError:
                    continue
                if value >= 0:
                    confidences.append(value)
            raw_text = " ".join(words)
            parsed = parse_multipliers(raw_text, maximum=self.maximum)
            if not parsed:
                continue
            confidence = sum(confidences) / len(confidences) if confidences else 0.0
            multipliers = [item.value for item in parsed]
            selected = multipliers[0] if all_values else parsed[0].value
            results.append(OCRResult(selected, raw_text, confidence, multipliers))
        return results
