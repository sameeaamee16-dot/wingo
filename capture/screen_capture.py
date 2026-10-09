from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageGrab


@dataclass(frozen=True)
class ScreenRegion:
    x: int
    y: int
    width: int
    height: int
    monitor: int = 0

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)


class ScreenCapture:
    def capture(self, region: ScreenRegion) -> Image.Image:
        if region.width <= 0 or region.height <= 0:
            raise ValueError("Screen region width and height must be positive.")
        return ImageGrab.grab(bbox=region.bbox, all_screens=True)
