from __future__ import annotations

import tkinter as tk
from typing import Callable

from capture.screen_capture import ScreenRegion


class RegionSelector(tk.Toplevel):
    def __init__(self, master: tk.Misc, on_selected: Callable[[ScreenRegion], None]) -> None:
        super().__init__(master)
        self.on_selected = on_selected
        self.start_x = 0
        self.start_y = 0
        self.rect_id: int | None = None
        self.attributes("-fullscreen", True)
        self.attributes("-alpha", 0.28)
        self.attributes("-topmost", True)
        self.configure(bg="black")
        self.canvas = tk.Canvas(self, cursor="crosshair", bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self._start)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._finish)
        self.bind("<Escape>", lambda _event: self.destroy())

    def _start(self, event: tk.Event) -> None:
        self.start_x = self.winfo_pointerx()
        self.start_y = self.winfo_pointery()
        if self.rect_id is not None:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="#39ff88", width=3)

    def _drag(self, event: tk.Event) -> None:
        if self.rect_id is not None:
            self.canvas.coords(self.rect_id, self.canvas.canvasx(event.x), self.canvas.canvasy(event.y), self.canvas.canvasx(self.start_x), self.canvas.canvasy(self.start_y))

    def _finish(self, _event: tk.Event) -> None:
        end_x = self.winfo_pointerx()
        end_y = self.winfo_pointery()
        x1, y1 = min(self.start_x, end_x), min(self.start_y, end_y)
        x2, y2 = max(self.start_x, end_x), max(self.start_y, end_y)
        if x2 - x1 > 10 and y2 - y1 > 10:
            self.on_selected(ScreenRegion(x=x1, y=y1, width=x2 - x1, height=y2 - y1))
        self.destroy()
