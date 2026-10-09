from __future__ import annotations

import shutil
from pathlib import Path

import pytesseract

from config import config


COMMON_WINDOWS_PATHS = [
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
]


def configure_tesseract() -> str | None:
    candidates: list[Path] = []
    if config.tesseract_cmd:
        candidates.append(Path(config.tesseract_cmd))
    candidates.extend(COMMON_WINDOWS_PATHS)

    for candidate in candidates:
        if candidate.exists():
            pytesseract.pytesseract.tesseract_cmd = str(candidate)
            return str(candidate)

    path_value = shutil.which("tesseract")
    if path_value:
        pytesseract.pytesseract.tesseract_cmd = path_value
        return path_value

    return None


def tesseract_help_message() -> str:
    return (
        "Tesseract OCR is not installed or the app cannot find tesseract.exe.\n\n"
        "Install Tesseract OCR for Windows, then either restart the app or set this in multiplier_tracker\\.env:\n\n"
        'TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    )
