from __future__ import annotations

import tkinter as tk
from decimal import Decimal
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from statistics.analytics_engine import loss_streaks


class AnalyticsPanel(ttk.LabelFrame):
    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master, text="Historical Statistics")
        self.notice = ttk.Label(
            self,
            text="Historical descriptive statistics only. This does not predict or guarantee the next result.",
        )
        self.notice.pack(anchor="w", padx=8, pady=(8, 0))
        self.figure = Figure(figsize=(8, 4), dpi=100, facecolor="#1f232a")
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    def redraw(self, rows: list[object], threshold: Decimal = Decimal("2.00")) -> None:
        self.figure.clear()
        multipliers = [Decimal(row.multiplier) for row in rows]
        results = [row.result_type for row in rows]

        ax1 = self.figure.add_subplot(221)
        ax2 = self.figure.add_subplot(222)
        ax3 = self.figure.add_subplot(223)
        ax4 = self.figure.add_subplot(224)
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_facecolor("#252a33")
            ax.tick_params(colors="#d8dee9", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#596273")

        ax1.hist([float(value) for value in multipliers], bins=20, color="#5eead4")
        ax1.set_title("Multiplier Distribution", color="#f8fafc", fontsize=9)
        ax2.bar(["Wins", "Losses"], [results.count("WIN"), results.count("LOSS")], color=["#22c55e", "#ef4444"])
        ax2.set_title("Wins vs Losses", color="#f8fafc", fontsize=9)

        analysis = loss_streaks(multipliers, threshold=float(threshold))
        ax3.bar([item["streak"] for item in analysis], [item["sample_size"] for item in analysis], color="#f59e0b")
        ax3.set_title("Preceding Loss-Streak Samples", color="#f8fafc", fontsize=9)
        ax4.plot([float(value) for value in multipliers[-100:]], color="#60a5fa")
        ax4.set_title("Timeline Last 100", color="#f8fafc", fontsize=9)
        self.figure.tight_layout()
        self.canvas.draw_idle()
