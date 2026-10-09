from __future__ import annotations

import tkinter as tk
from decimal import Decimal
from tkinter import ttk

from database.models import AppSettings


class SettingsPanel(ttk.LabelFrame):
    def __init__(self, master: tk.Misc, on_save) -> None:
        super().__init__(master, text="Settings")
        self.on_save = on_save
        self.mode_var = tk.StringVar(value="both")
        self.interval_var = tk.StringVar(value="1.00")
        self.conf_var = tk.StringVar(value="80.00")
        self.threshold_var = tk.StringVar(value="1.50")
        self.capture_var = tk.StringVar(value="0.75")

        ttk.Label(self, text="OCR Mode").grid(row=0, column=0, sticky="w", padx=8, pady=4)
        ttk.Combobox(self, textvariable=self.mode_var, values=["center", "history", "both"], state="readonly", width=12).grid(row=0, column=1, padx=8, pady=4)
        for row, (label, var) in enumerate(
            [
                ("Minimum interval", self.interval_var),
                ("Confidence threshold", self.conf_var),
                ("Loss threshold", self.threshold_var),
                ("Capture interval", self.capture_var),
            ],
            start=1,
        ):
            ttk.Label(self, text=label).grid(row=row, column=0, sticky="w", padx=8, pady=4)
            ttk.Entry(self, textvariable=var, width=14).grid(row=row, column=1, padx=8, pady=4)
        ttk.Button(self, text="Save Settings", command=self._save).grid(row=5, column=0, columnspan=2, sticky="ew", padx=8, pady=8)

    def load(self, settings: AppSettings) -> None:
        self.mode_var.set(settings.ocr_mode)
        self.interval_var.set(str(settings.minimum_interval))
        self.conf_var.set(str(settings.ocr_confidence_threshold))
        self.threshold_var.set(str(settings.loss_threshold))
        self.capture_var.set(str(settings.capture_interval))

    def _save(self) -> None:
        self.on_save(
            {
                "ocr_mode": self.mode_var.get(),
                "minimum_interval": Decimal(self.interval_var.get()),
                "ocr_confidence_threshold": Decimal(self.conf_var.get()),
                "loss_threshold": Decimal(self.threshold_var.get()),
                "capture_interval": Decimal(self.capture_var.get()),
            }
        )
