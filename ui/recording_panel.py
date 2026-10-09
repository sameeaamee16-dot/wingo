from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class RecordingPanel(ttk.LabelFrame):
    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master, text="Live Recording")
        self.status_var = tk.StringVar(value="● STOPPED")
        self.current_var = tk.StringVar(value="Current Multiplier: --")
        self.result_var = tk.StringVar(value="Result: --")
        self.streak_var = tk.StringVar(value="Current Streak: --")
        self.last_var = tk.StringVar(value="Last Result: --")
        self.previous_var = tk.StringVar(value="Previous: --")

        ttk.Label(self, textvariable=self.status_var, font=("Segoe UI", 14, "bold")).pack(anchor="w", padx=12, pady=(10, 4))
        for var in [self.current_var, self.result_var, self.streak_var, self.last_var, self.previous_var]:
            ttk.Label(self, textvariable=var, font=("Segoe UI", 11)).pack(anchor="w", padx=12, pady=2)

    def update_state(self, recording: bool, current: str = "--", result: str = "--", streak: str = "--", previous: list[str] | None = None) -> None:
        self.status_var.set("● RECORDING" if recording else "● STOPPED")
        self.current_var.set(f"Current Multiplier: {current}")
        self.result_var.set(f"Result: {result}")
        self.streak_var.set(f"Current Streak: {streak}")
        self.last_var.set(f"Last Result: {current}")
        self.previous_var.set("Previous: " + ", ".join(previous or []))
