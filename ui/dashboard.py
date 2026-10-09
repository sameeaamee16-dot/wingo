from __future__ import annotations

import hashlib
import logging
import queue
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from tkinter import filedialog, messagebox, simpledialog, ttk

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from capture.duplicate_detector import DuplicateDetector
from capture.history_detector import HistoryDetector
from capture.region_selector import RegionSelector
from capture.screen_capture import ScreenCapture, ScreenRegion
from config import config
from database.connection import get_session
from database.models import AppSettings, GameResult, SessionModel, utc_now
from exports.csv_export import export_csv
from exports.excel_export import export_excel
from ocr.multiplier_reader import MultiplierReader, OCRResult
from ocr.validator import classify_multiplier
from statistics.analytics import summarize_results
from statistics.streak_engine import StreakEngine
from ui.analytics_panel import AnalyticsPanel
from ui.history_panel import HistoryPanel
from ui.recording_panel import RecordingPanel
from ui.settings_panel import SettingsPanel

logger = logging.getLogger(__name__)


@dataclass
class WorkerEvent:
    kind: str
    message: str = ""
    ocr: OCRResult | None = None
    saved: bool = False


class RecordingWorker(threading.Thread):
    def __init__(self, session_id: int, events: queue.Queue[WorkerEvent], stop_event: threading.Event) -> None:
        super().__init__(daemon=True)
        self.session_id = session_id
        self.events = events
        self.stop_event = stop_event
        self.capture = ScreenCapture()
        self.reader = MultiplierReader()
        self.history = HistoryDetector()

    def run(self) -> None:
        duplicate = DuplicateDetector()
        while not self.stop_event.is_set():
            try:
                with get_session() as db:
                    settings = db.get(AppSettings, 1) or AppSettings(id=1)
                    region = ScreenRegion(settings.screen_x, settings.screen_y, settings.screen_width, settings.screen_height)
                    duplicate.minimum_interval = float(settings.minimum_interval)
                    threshold = Decimal(settings.loss_threshold)
                    confidence_threshold = float(settings.ocr_confidence_threshold)
                    interval = float(settings.capture_interval)
                    mode = settings.ocr_mode
                    allow_low = bool(settings.allow_low_confidence)
                image = self.capture.capture(region)
                ocr = self._read_by_mode(mode, image)
                if ocr.multiplier is None:
                    self.events.put(WorkerEvent("warning", "OCR cannot detect the multiplier. Please select the multiplier region again.", ocr))
                    time.sleep(interval)
                    continue
                if ocr.confidence < confidence_threshold and not allow_low:
                    self.events.put(WorkerEvent("warning", f"UNCONFIRMED OCR result ignored: {ocr.raw_text} ({ocr.confidence:.0f}%).", ocr))
                    time.sleep(interval)
                    continue
                if duplicate.is_duplicate(ocr.multiplier):
                    self.events.put(WorkerEvent("duplicate", "Duplicate visible result ignored.", ocr))
                    time.sleep(interval)
                    continue
                saved = self._save_result(ocr, threshold, mode)
                self.events.put(WorkerEvent("result", "Saved result." if saved else "Duplicate database fingerprint ignored.", ocr, saved=saved))
                time.sleep(interval)
            except Exception as exc:
                logger.exception("Recording worker error")
                self.events.put(WorkerEvent("error", str(exc)))
                time.sleep(1.5)

    def _read_by_mode(self, mode: str, image) -> OCRResult:
        if mode == "history":
            history_ocr = self.reader.read_history(image)
            new_value = self.history.detect_new(history_ocr.all_multipliers)
            return OCRResult(new_value, history_ocr.raw_text, history_ocr.confidence, history_ocr.all_multipliers)
        if mode == "both":
            history_ocr = self.reader.read_history(image)
            new_value = self.history.detect_new(history_ocr.all_multipliers)
            if new_value is not None:
                return OCRResult(new_value, history_ocr.raw_text, history_ocr.confidence, history_ocr.all_multipliers)
        return self.reader.read_center(image)

    def _save_result(self, ocr: OCRResult, threshold: Decimal, source: str) -> bool:
        assert ocr.multiplier is not None
        with get_session() as db:
            session_model = db.get(SessionModel, self.session_id)
            if session_model is None or session_model.ended_at is not None:
                return False
            prior = list(db.scalars(select(GameResult).where(GameResult.session_id == self.session_id).order_by(GameResult.result_number)))
            engine = StreakEngine(threshold)
            for row in prior:
                engine.add_multiplier(Decimal(row.multiplier))
            result_type, loss_before, win_before = engine.add_multiplier(ocr.multiplier)
            result_number = len(prior) + 1
            now = utc_now()
            fingerprint = hashlib.sha256(f"{self.session_id}:{result_number}:{ocr.multiplier}:{source}".encode()).hexdigest()
            row = GameResult(
                session_id=self.session_id,
                result_number=result_number,
                multiplier=ocr.multiplier,
                result_type=result_type,
                loss_streak_before=loss_before + (1 if result_type == "LOSS" else 0),
                win_streak_before=win_before + (1 if result_type == "WIN" else 0),
                timestamp=now,
                source=source,
                ocr_confidence=Decimal(str(round(ocr.confidence, 2))),
                raw_ocr_text=ocr.raw_text,
                fingerprint=fingerprint,
            )
            db.add(row)
            try:
                db.flush()
            except IntegrityError:
                db.rollback()
                return False
            all_rows = prior + [row]
            summary = summarize_results(all_rows, threshold)
            session_model.total_results = summary.total
            session_model.total_wins = summary.wins
            session_model.total_losses = summary.losses
            session_model.maximum_loss_streak = summary.max_loss_streak
            session_model.maximum_win_streak = summary.max_win_streak
            return True


class MultiplierTrackerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(config.app_name)
        self.geometry("1280x820")
        self.configure(bg="#151922")
        self.events: queue.Queue[WorkerEvent] = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: RecordingWorker | None = None
        self.active_session_id: int | None = None
        self.last_rows: list[GameResult] = []
        self._style()
        self._build()
        self.refresh()
        self.after(500, self._poll_events)

    def _style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background="#151922", foreground="#e5e7eb", fieldbackground="#252a33", font=("Segoe UI", 10))
        style.configure("TFrame", background="#151922")
        style.configure("TLabelframe", background="#151922", foreground="#f8fafc", bordercolor="#384152")
        style.configure("TLabelframe.Label", background="#151922", foreground="#f8fafc", font=("Segoe UI", 10, "bold"))
        style.configure("TButton", background="#273244", foreground="#f8fafc", padding=7)
        style.map("TButton", background=[("active", "#334155")])
        style.configure("Treeview", background="#202631", foreground="#e5e7eb", fieldbackground="#202631", rowheight=26)
        style.configure("Treeview.Heading", background="#334155", foreground="#f8fafc")

    def _build(self) -> None:
        header = ttk.Frame(self)
        header.pack(fill=tk.X, padx=14, pady=10)
        ttk.Label(header, text="MULTIPLIER TRACKER", font=("Segoe UI", 18, "bold")).pack(side=tk.LEFT)
        self.header_status = ttk.Label(header, text="● Connected to MySQL   ● OCR Ready   ● Recording: OFF")
        self.header_status.pack(side=tk.RIGHT)

        controls = ttk.Frame(self)
        controls.pack(fill=tk.X, padx=14, pady=(0, 10))
        for label, command in [
            ("SELECT REGION", self.select_region),
            ("TEST OCR", self.test_ocr),
            ("START RECORDING", self.start_recording),
            ("PAUSE", self.pause_recording),
            ("STOP", self.stop_recording),
            ("VIEW STATISTICS", self.refresh),
            ("EDIT RESULT", self.edit_result),
            ("EXPORT CSV", self.export_csv_dialog),
            ("EXPORT EXCEL", self.export_excel_dialog),
        ]:
            ttk.Button(controls, text=label, command=command).pack(side=tk.LEFT, padx=4)
        self.filter_var = tk.StringVar(value="all")
        ttk.Combobox(
            controls,
            textvariable=self.filter_var,
            values=["all", "wins", "losses", "loss_streak_3", "loss_streak_5", "multiplier_10", "between_2_5"],
            state="readonly",
            width=16,
        ).pack(side=tk.RIGHT)
        ttk.Button(controls, text="APPLY FILTER", command=self.refresh).pack(side=tk.RIGHT, padx=4)

        self.cards = ttk.Frame(self)
        self.cards.pack(fill=tk.X, padx=14, pady=(0, 10))
        self.card_vars = {}
        for name in ["TOTAL", "WINS", "LOSSES", "WIN %", "LOSS %", "CURRENT STREAK", "MAX LOSS STREAK", "MAX WIN STREAK"]:
            frame = ttk.LabelFrame(self.cards, text=name)
            frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
            var = tk.StringVar(value="0")
            self.card_vars[name] = var
            ttk.Label(frame, textvariable=var, font=("Segoe UI", 14, "bold")).pack(padx=8, pady=10)

        body = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 14))
        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=2)
        body.add(right, weight=3)

        self.recording_panel = RecordingPanel(left)
        self.recording_panel.pack(fill=tk.X, pady=(0, 8))
        self.settings_panel = SettingsPanel(left, self.save_settings)
        self.settings_panel.pack(fill=tk.X)
        self.history_panel = HistoryPanel(right)
        self.history_panel.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        self.analytics_panel = AnalyticsPanel(right)
        self.analytics_panel.pack(fill=tk.BOTH, expand=True)

    def select_region(self) -> None:
        RegionSelector(self, self._save_region)

    def _save_region(self, region: ScreenRegion) -> None:
        with get_session() as db:
            settings = db.get(AppSettings, 1) or AppSettings(id=1)
            settings.screen_x = region.x
            settings.screen_y = region.y
            settings.screen_width = region.width
            settings.screen_height = region.height
            db.merge(settings)
        messagebox.showinfo("Region saved", f"Saved region: x={region.x}, y={region.y}, width={region.width}, height={region.height}")
        self.refresh()

    def test_ocr(self) -> None:
        try:
            with get_session() as db:
                settings = db.get(AppSettings, 1)
                region = ScreenRegion(settings.screen_x, settings.screen_y, settings.screen_width, settings.screen_height)
            image = ScreenCapture().capture(region)
            result = MultiplierReader().read_history(image, fast=True) if settings.ocr_mode == "history" else MultiplierReader().read_center(image, fast=True)
            parsed = f"{result.multiplier}x" if result.multiplier is not None else "None"
            messagebox.showinfo("OCR Calibration", f"Detected text: {result.raw_text}\nParsed value: {parsed}\nConfidence: {result.confidence:.0f}%")
        except Exception as exc:
            messagebox.showerror("OCR test failed", str(exc))

    def start_recording(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        with get_session() as db:
            session_model = SessionModel(session_name=f"Session {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            db.add(session_model)
            db.flush()
            self.active_session_id = session_model.id
        self.stop_event.clear()
        self.worker = RecordingWorker(self.active_session_id, self.events, self.stop_event)
        self.worker.start()
        self.header_status.configure(text="● Connected to MySQL   ● OCR Ready   ● Recording: ON")
        self.recording_panel.update_state(True)

    def pause_recording(self) -> None:
        self.stop_event.set()
        self.header_status.configure(text="● Connected to MySQL   ● OCR Ready   ● Recording: PAUSED")

    def stop_recording(self) -> None:
        self.stop_event.set()
        if self.active_session_id is not None:
            with get_session() as db:
                session_model = db.get(SessionModel, self.active_session_id)
                if session_model and session_model.ended_at is None:
                    session_model.ended_at = utc_now()
            self.active_session_id = None
        self.header_status.configure(text="● Connected to MySQL   ● OCR Ready   ● Recording: OFF")
        self.recording_panel.update_state(False)
        self.refresh()

    def save_settings(self, values: dict) -> None:
        with get_session() as db:
            settings = db.get(AppSettings, 1) or AppSettings(id=1)
            for key, value in values.items():
                setattr(settings, key, value)
            db.merge(settings)
        self.refresh()

    def refresh(self) -> None:
        with get_session() as db:
            settings = db.get(AppSettings, 1) or AppSettings(id=1)
            self.settings_panel.load(settings)
            query = select(GameResult).order_by(GameResult.timestamp)
            rows = list(db.scalars(query))
            rows = self._apply_filter(rows)
            self.last_rows = rows
        summary = summarize_results(rows, Decimal(settings.loss_threshold))
        self.card_vars["TOTAL"].set(f"{summary.total:,}")
        self.card_vars["WINS"].set(f"{summary.wins:,}")
        self.card_vars["LOSSES"].set(f"{summary.losses:,}")
        self.card_vars["WIN %"].set(f"{summary.win_pct:.2f}%")
        self.card_vars["LOSS %"].set(f"{summary.loss_pct:.2f}%")
        self.card_vars["CURRENT STREAK"].set(summary.current_streak_label)
        self.card_vars["MAX LOSS STREAK"].set(str(summary.max_loss_streak))
        self.card_vars["MAX WIN STREAK"].set(str(summary.max_win_streak))
        self.history_panel.set_rows(rows[-100:])
        self.analytics_panel.redraw(rows, Decimal(settings.loss_threshold))

    def _apply_filter(self, rows: list[GameResult]) -> list[GameResult]:
        mode = self.filter_var.get()
        if mode == "wins":
            return [row for row in rows if row.result_type == "WIN"]
        if mode == "losses":
            return [row for row in rows if row.result_type == "LOSS"]
        if mode == "loss_streak_3":
            return [row for row in rows if row.loss_streak_before >= 3]
        if mode == "loss_streak_5":
            return [row for row in rows if row.loss_streak_before >= 5]
        if mode == "multiplier_10":
            return [row for row in rows if Decimal(row.multiplier) >= Decimal("10.00")]
        if mode == "between_2_5":
            return [row for row in rows if Decimal("2.00") <= Decimal(row.multiplier) <= Decimal("5.00")]
        return rows

    def edit_result(self) -> None:
        if not self.last_rows:
            messagebox.showinfo("Edit Result", "No results are available.")
            return
        result_id = simpledialog.askinteger("Edit Result", "Enter result ID to correct:")
        if result_id is None:
            return
        new_value_text = simpledialog.askstring("Edit Result", "Enter corrected multiplier:")
        if not new_value_text:
            return
        new_value = Decimal(new_value_text).quantize(Decimal("0.01"))
        with get_session() as db:
            row = db.get(GameResult, result_id)
            if row is None:
                messagebox.showerror("Edit Result", "Result ID not found.")
                return
            row.original_multiplier = row.original_multiplier or row.multiplier
            row.corrected_multiplier = new_value
            row.corrected_at = utc_now()
            row.multiplier = new_value
            settings = db.get(AppSettings, 1)
            row.result_type = classify_multiplier(new_value, Decimal(settings.loss_threshold))
        self.refresh()

    def export_csv_dialog(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            export_csv(self.last_rows, path)
            messagebox.showinfo("Export CSV", f"Exported {len(self.last_rows)} rows.")

    def export_excel_dialog(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
        if path:
            export_excel(self.last_rows, path)
            messagebox.showinfo("Export Excel", f"Exported {len(self.last_rows)} rows.")

    def _poll_events(self) -> None:
        changed = False
        previous = [f"{row.multiplier}x" for row in self.last_rows[-5:]]
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            if event.kind in {"result", "warning", "error"} and event.ocr:
                current = f"{event.ocr.multiplier}x" if event.ocr.multiplier else "--"
                result_type = classify_multiplier(event.ocr.multiplier) if event.ocr.multiplier else "--"
                self.recording_panel.update_state(True, current=current, result=result_type, streak=self.card_vars["CURRENT STREAK"].get(), previous=previous)
            if event.kind == "error":
                self.header_status.configure(text="● Connected to MySQL   ● OCR Issue   ● Recording: ON")
            if event.saved:
                changed = True
        if changed:
            self.refresh()
        self.after(500, self._poll_events)
