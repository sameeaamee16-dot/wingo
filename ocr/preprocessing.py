from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def pil_to_cv(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def cv_to_pil(image: np.ndarray) -> Image.Image:
    if len(image.shape) == 2:
        return Image.fromarray(image)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def preprocess_variants(image: Image.Image, scale: int = 3) -> list[Image.Image]:
    # Limit the enlarged image to avoid huge memory usage
    max_pixels = 50_000_000

    new_width = image.width * scale
    new_height = image.height * scale

    if new_width * new_height > max_pixels:
        factor = (max_pixels / (image.width * image.height)) ** 0.5
        scale = max(1, int(scale * factor))

        new_width = image.width * scale
        new_height = image.height * scale

    enlarged = image.convert("RGB").resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    gray = enlarged.convert("L")
    contrast = ImageEnhance.Contrast(gray).enhance(2.0)
    sharp = contrast.filter(ImageFilter.SHARPEN)

    arr = np.array(sharp)

    _, binary = cv2.threshold(
        arr, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    adaptive = cv2.adaptiveThreshold(
        arr, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 7
    )

    inverted = cv2.bitwise_not(binary)

    return [
        enlarged,
        contrast,
        sharp,
        cv_to_pil(binary),
        cv_to_pil(adaptive),
        cv_to_pil(inverted),
    ]


def preprocess_fast_variants(image: Image.Image, scale: int = 3) -> list[Image.Image]:
    max_pixels = 50_000_000

    new_width = image.width * scale
    new_height = image.height * scale

    if new_width * new_height > max_pixels:
        factor = (max_pixels / (image.width * image.height)) ** 0.5
        scale = max(1, int(scale * factor))

        new_width = image.width * scale
        new_height = image.height * scale

    enlarged = image.convert("RGB").resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    gray = enlarged.convert("L")
    contrast = ImageEnhance.Contrast(gray).enhance(2.4)
    sharp = contrast.filter(ImageFilter.SHARPEN)

    arr = np.array(sharp)

    _, binary = cv2.threshold(
        arr, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return [sharp, cv_to_pil(binary)]