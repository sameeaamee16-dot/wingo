from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class HistoryPanel(ttk.LabelFrame):
    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master, text="Live Results")
        columns = ("round", "time", "multiplier", "result", "loss", "win", "confidence")
        self.tree = ttk.Treeview(self, columns=columns, show="headings", height=12)
        headings = {
            "round": "#",
            "time": "Time",
            "multiplier": "Multiplier",
            "result": "Result",
            "loss": "Loss Streak",
            "win": "Win Streak",
            "confidence": "OCR Confidence",
        }
        for key, label in headings.items():
            self.tree.heading(key, text=label)
            self.tree.column(key, width=105, anchor="center")
        scroll = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=8)
        scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=8)

    def set_rows(self, rows: list[object]) -> None:
        self.tree.delete(*self.tree.get_children())
        reversed_rows = list(reversed(rows))
        for row in reversed_rows[:100]:
            time_value = row.timestamp.astimezone().strftime("%H:%M:%S") if row.timestamp else ""
            self.tree.insert(
                "",
                "end",
                values=(
                    row.result_number,
                    time_value,
                    f"{row.multiplier}x",
                    row.result_type,
                    row.loss_streak_before,
                    row.win_streak_before,
                    f"{float(row.ocr_confidence or 0):.0f}%",
                ),
            )
