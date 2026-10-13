from __future__ import annotations

import base64
import hashlib
import io
import logging
import queue
import threading
import time
from datetime import datetime
from decimal import Decimal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from capture.duplicate_detector import DuplicateDetector
from capture.history_detector import HistoryDetector
from capture.screen_capture import ScreenCapture, ScreenRegion
from config import config
from database.connection import get_session
from database.migrations import initialize_database
from database.models import AppSettings, CaptureAudit, GameResult, PatternIdentifier, SessionModel, TableCapture, WingoAnalysis, utc_now
from ocr.multiplier_reader import MultiplierReader, OCRResult
from ocr.validator import classify_multiplier
from statistics.analytics import summarize_results
from statistics.analytics_engine import loss_streaks, overview, patterns, recent_comparison, walk_forward
from statistics.streak_engine import StreakEngine
from agent import service as agent_service
from statistics.wingo_analyzer import analyze_history, classify_number, period_delta, row_is_valid
import pytesseract
import re
from ocr.tesseract_config import configure_tesseract


logger = logging.getLogger(__name__)
app = FastAPI(title=config.app_name)


class RegionPayload(BaseModel):
    x: int
    y: int
    width: int
    height: int


class BrowserFramePayload(BaseModel):
    image: str


class OCRError(Exception):
    pass


class SettingsPayload(BaseModel):
    ocr_mode: str = "history"
    minimum_interval: float = 1.0
    ocr_confidence_threshold: float = 65.0
    loss_threshold: float = 1.5
    capture_interval: float = 0.4


class WorkerEvent(BaseModel):
    kind: str
    message: str = ""
    multiplier: str | None = None
    confidence: float = 0.0
    raw_text: str = ""
    saved: bool = False
    period: str | None = None


events: queue.Queue[WorkerEvent] = queue.Queue()
stop_event = threading.Event()
signal_issue_lock = threading.Lock()
pattern_index_latest_id: int | None = None
analysis_cache_latest_id: tuple[int, int] | None = None
analysis_cache: dict | None = None
analysis_cache_lock = threading.Lock()
analysis_persist_lock = threading.Lock()
worker: "WebRecordingWorker | None" = None
active_session_id: int | None = None
browser_frame_lock = threading.Lock()
browser_frame: Image.Image | None = None
browser_frame_at = 0.0
browser_capture_enabled = False


def _record_capture_audit(period: str | None, status: str, message: str) -> None:
    try:
        with get_session() as db:
            exists = db.scalar(select(CaptureAudit.id).where(CaptureAudit.period == period, CaptureAudit.status == status, CaptureAudit.message == message).limit(1))
            if not exists:
                db.add(CaptureAudit(period=period, status=status, message=message, source_status="OCR", created_at=utc_now()))
    except Exception:
        logger.exception("Could not save capture audit event")


def parse_table_rows(image):
    """Read complete period/number/size rows; colors are deliberately ignored."""
    configure_tesseract()
    data = pytesseract.image_to_data(image, config="--psm 6", output_type=pytesseract.Output.DICT)
    lines: dict[tuple[int, int, int], list[str]] = {}
    confs: dict[tuple[int, int, int], list[float]] = {}
    for i, raw in enumerate(data.get("text", [])):
        word = (raw or "").strip()
        if not word:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)
        try:
            confs.setdefault(key, []).append(float(data["conf"][i]))
        except (ValueError, TypeError):
            pass
    parsed = []
    raw_lines = []
    for key, words in lines.items():
        line = " ".join(words)
        raw_lines.append(re.sub(r"\b(Green|Red|Pink|Violet|Purple)\b", "", line, flags=re.I).strip())
        period_match = re.search(r"\b\d{10,24}\b", line)
        if not period_match:
            continue
        tail = line[period_match.end():]
        number_match = re.search(r"\b([0-9])\b", tail)
        size_match = re.search(r"\b(Big|Small)\b", tail, re.I)
        if not (number_match and size_match):
            continue
        parsed.append({"period": period_match.group(), "number": int(number_match.group(1)), "size": size_match.group(1).title(), "confidence": sum(confs.get(key, [0])) / max(1, len(confs.get(key, [])))})
    return parsed, "\n".join(raw_lines)


def _sync_pattern_identifiers(db, captures: list[TableCapture]) -> None:
    """Persist sequence counts without joining rows across invalid or missing periods."""
    global pattern_index_latest_id
    counts: dict[str, dict[str, int]] = {}
    valid = [row for row in captures if row_is_valid(row)]
    for start in range(len(valid)):
        for length in range(1, min(5, len(valid) - start) + 1):
            sequence = valid[start:start + length]
            if any(period_delta(sequence[j].period, sequence[j + 1].period) != 1 for j in range(len(sequence) - 1)):
                continue
            pattern = " ".join(row.size for row in sequence)
            entry = counts.setdefault(pattern, {"occurrences": 0, "Big": 0, "Small": 0})
            entry["occurrences"] += 1
            next_index = start + length
            if next_index < len(valid) and period_delta(valid[next_index - 1].period, valid[next_index].period) == 1:
                entry[valid[next_index].size] += 1

    existing = {row.pattern: row for row in db.scalars(select(PatternIdentifier))}
    for pattern in set(existing) - set(counts):
        db.delete(existing[pattern])
    for pattern, values in counts.items():
        row = existing.get(pattern)
        if row is None:
            row = PatternIdentifier(pattern=pattern)
            db.add(row)
        row.occurrences = values["occurrences"]
        row.next_big_count = values["Big"]
        row.next_small_count = values["Small"]
    pattern_index_latest_id = captures[-1].id if captures else None


def image_to_data_url(image) -> str:
    output = io.BytesIO()
    image.save(output, format="PNG")
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def latest_browser_frame() -> Image.Image | None:
    """Return a recent frame supplied by the browser tab-sharing stream."""
    with browser_frame_lock:
        if not browser_capture_enabled or browser_frame is None or time.monotonic() - browser_frame_at > 2.0:
            return None
        return browser_frame.copy()


def capture_region(region: ScreenRegion) -> Image.Image | None:
    """Capture the configured area from the active browser stream or desktop."""
    frame = latest_browser_frame()
    if frame is not None:
        return frame.crop(region.bbox)
    with browser_frame_lock:
        if browser_capture_enabled:
            return None
    return ScreenCapture().capture(region)


def limit_ocr_image(image: Image.Image, max_pixels: int = 1_200_000) -> Image.Image:
    """Keep one-off OCR tests responsive if the selected region is too large."""
    pixels = image.width * image.height
    if pixels <= max_pixels:
        return image
    factor = (max_pixels / pixels) ** 0.5
    new_size = (max(1, int(image.width * factor)), max(1, int(image.height * factor)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


class WebRecordingWorker(threading.Thread):
    def __init__(self, session_id: int) -> None:
        super().__init__(daemon=True)
        self.session_id = session_id
        self.capture = ScreenCapture()
        self.reader = MultiplierReader()
        self.history = HistoryDetector()

    def run(self) -> None:
        duplicate = DuplicateDetector()
        while not stop_event.is_set():
            try:
                with get_session() as db:
                    settings = db.get(AppSettings, 1) or AppSettings(id=1)
                    region = ScreenRegion(settings.screen_x, settings.screen_y, settings.screen_width, settings.screen_height)
                    mode = settings.ocr_mode
                    duplicate.minimum_interval = float(settings.minimum_interval)
                    confidence_threshold = float(settings.ocr_confidence_threshold)
                    threshold = Decimal(settings.loss_threshold)
                    interval = float(settings.capture_interval)
                    allow_low = bool(settings.allow_low_confidence)

                image = capture_region(region)
                if image is None:
                    events.put(WorkerEvent(kind="warning", message="Waiting for a frame from the shared browser tab"))
                    time.sleep(interval)
                    continue
                ocr = self._read(mode, image)
                if ocr.multiplier is None:
                    events.put(WorkerEvent(kind="warning", message="No multiplier detected", raw_text=ocr.raw_text, confidence=ocr.confidence))
                    time.sleep(interval)
                    continue
                if ocr.confidence < confidence_threshold and not allow_low:
                    events.put(
                        WorkerEvent(
                            kind="warning",
                            message="Low confidence OCR ignored",
                            multiplier=str(ocr.multiplier),
                            raw_text=ocr.raw_text,
                            confidence=ocr.confidence,
                        )
                    )
                    time.sleep(interval)
                    continue
                if duplicate.is_duplicate(ocr.multiplier):
                    events.put(WorkerEvent(kind="duplicate", message="Duplicate ignored", multiplier=str(ocr.multiplier), confidence=ocr.confidence))
                    time.sleep(interval)
                    continue
                saved = self._save_result(ocr, threshold, mode)
                events.put(
                    WorkerEvent(
                        kind="result",
                        message="Saved" if saved else "Database duplicate ignored",
                        multiplier=str(ocr.multiplier),
                        confidence=ocr.confidence,
                        raw_text=ocr.raw_text,
                        saved=saved,
                    )
                )
                time.sleep(interval)
            except Exception as exc:
                logger.exception("Web recorder error")
                events.put(WorkerEvent(kind="error", message=str(exc)))
            time.sleep(1.5)


class TableCaptureWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)

    def run(self):
        while not stop_event.is_set():
            try:
                with get_session() as db:
                    settings = db.get(AppSettings, 1) or AppSettings(id=1)
                    region = ScreenRegion(settings.screen_x, settings.screen_y, settings.screen_width, settings.screen_height)
                    interval = max(0.25, float(settings.capture_interval))
                image = capture_region(region)
                if image is None:
                    events.put(WorkerEvent(kind="waiting", message="Waiting for selected screen"))
                else:
                    # Match Fast Test OCR's resize path so live reads use the same scale.
                    image = limit_ocr_image(image)
                    rows, raw = parse_table_rows(image)
                    if not rows:
                        events.put(WorkerEvent(kind="waiting", message="Waiting for a complete readable row", raw_text=raw))
                    # Use the period value for ordering; OCR block order is not stable.
                    for item in sorted(rows, key=lambda row: int(row["period"]) if str(row.get("period", "")).isdigit() else -1):
                        period = str(item.get("period", ""))
                        number = item.get("number")
                        if not period.isdigit() or not 10 <= len(period) <= 24 or not isinstance(number, int) or not 0 <= number <= 9:
                            message = f"Invalid result rejected for period {period or '?'}"
                            _record_capture_audit(period or None, "INVALID RESULT", message)
                            events.put(WorkerEvent(kind="warning", message=message))
                            continue
                        calculated_size = classify_number(number)
                        if item.get("size") != calculated_size:
                            message = f"OCR number/size mismatch rejected for period {period}"
                            _record_capture_audit(period, "INVALID RESULT", message)
                            events.put(WorkerEvent(kind="warning", message=message))
                            continue
                        with get_session() as db:
                            if db.scalar(select(TableCapture.id).where(TableCapture.period == period)):
                                duplicate_audit = db.scalar(select(CaptureAudit.id).where(CaptureAudit.period == period, CaptureAudit.status == "DUPLICATE").limit(1))
                                if not duplicate_audit:
                                    db.add(CaptureAudit(period=period, status="DUPLICATE", message="Duplicate period ignored", source_status="OCR", created_at=utc_now()))
                                events.put(WorkerEvent(kind="duplicate", message="Duplicate period ignored", period=period))
                                continue
                            captures = list(db.scalars(select(TableCapture).order_by(TableCapture.period, TableCapture.id)))
                            period_value = int(period)
                            earlier = [row for row in captures if len(row.period) == len(period) and int(row.period) < period_value]
                            later = [row for row in captures if len(row.period) == len(period) and int(row.period) > period_value]
                            previous = earlier[-1] if earlier else None
                            following = later[0] if later else None
                            status = "Valid"
                            delta = period_delta(previous.period, period) if previous else None
                            fills_gap = bool(following and previous and delta == 1 and period_delta(period, following.period) > 1)
                            if following and not fills_gap:
                                message = f"Out-of-order period rejected: {period} does not fill a known gap"
                                abnormal_audit = db.scalar(select(CaptureAudit.id).where(CaptureAudit.period == period, CaptureAudit.status == "ABNORMAL JUMP").limit(1))
                                if not abnormal_audit:
                                    db.add(CaptureAudit(period=period, status="ABNORMAL JUMP", message=message, source_status="OCR", created_at=utc_now()))
                                events.put(WorkerEvent(kind="warning", message=message, period=period))
                                continue
                            if previous and delta > 1:
                                status = "Missing period"

                            if previous:
                                pending = db.scalar(select(WingoAnalysis).where(WingoAnalysis.analysis_period == previous.period, WingoAnalysis.evaluation_mode == "LIVE", WingoAnalysis.actual_period.is_(None)))
                                if pending:
                                    quality_issue = db.scalar(select(CaptureAudit.id).where(CaptureAudit.created_at > pending.created_at, CaptureAudit.status.in_(["INVALID RESULT", "ABNORMAL JUMP"])).limit(1))
                                    pending.actual_period = period
                                    pending.actual_number = number
                                    pending.actual_size = calculated_size
                                    pending.outcome = "SKIPPED_MISSING_PERIOD" if status == "Missing period" else "SKIPPED_DATA_QUALITY" if quality_issue else "NO SIGNAL" if pending.signal == "NO CLEAR SIGNAL" else "WIN" if pending.signal == calculated_size.upper() else "LOSS"
                                    pending.resolved_at = utc_now()
                                elif fills_gap and following:
                                    skipped = db.scalar(select(WingoAnalysis).where(
                                        WingoAnalysis.analysis_period == previous.period,
                                        WingoAnalysis.evaluation_mode == "LIVE",
                                        WingoAnalysis.actual_period == following.period,
                                        WingoAnalysis.outcome == "SKIPPED_MISSING_PERIOD",
                                    ))
                                    if skipped:
                                        skipped.actual_period = period
                                        skipped.actual_number = number
                                        skipped.actual_size = calculated_size
                                        skipped.outcome = "NO SIGNAL" if skipped.signal == "NO CLEAR SIGNAL" else "WIN" if skipped.signal == calculated_size.upper() else "LOSS"
                                        skipped.resolved_at = utc_now()

                            db.add(TableCapture(
                                period=period,
                                number=number,
                                size=calculated_size,
                                colors="",
                                captured_at=utc_now(),
                                confidence=Decimal(str(round(item["confidence"], 2))),
                                raw_text=raw,
                                previous_period=previous.period if previous else None,
                                previous_number=previous.number if previous else None,
                                data_status=status,
                                source_status="OCR",
                            ))
                            try:
                                db.flush()
                                saved_rows = list(db.scalars(select(TableCapture).order_by(TableCapture.period, TableCapture.id)))
                                prior_row = None
                                for saved_row in saved_rows:
                                    if prior_row and len(saved_row.period) == len(prior_row.period):
                                        saved_row.previous_period = prior_row.period
                                        saved_row.previous_number = prior_row.number
                                        saved_row.data_status = "Valid" if period_delta(prior_row.period, saved_row.period) == 1 else "Missing period"
                                    elif not prior_row:
                                        saved_row.previous_period = None
                                        saved_row.previous_number = None
                                        saved_row.data_status = "Valid"
                                    prior_row = saved_row
                                db.flush()
                                _sync_pattern_identifiers(db, saved_rows)
                                message = "Captured result" if status == "Valid" else "Captured result; missing period detected"
                                events.put(WorkerEvent(kind="result", message=message, period=period, raw_text=raw, confidence=item["confidence"], saved=True))
                            except IntegrityError:
                                db.rollback()
                time.sleep(interval)
            except Exception as exc:
                logger.exception("Table capture OCR error")
                events.put(WorkerEvent(kind="error", message=str(exc)))
                time.sleep(1.0)

    def _read(self, mode: str, image) -> OCRResult:
        if mode == "center":
            return self.reader.read_center(image, fast=True)
        history_ocr = self.reader.read_history(image, fast=True)
        new_value = self.history.detect_new(history_ocr.all_multipliers)
        if new_value is not None:
            return OCRResult(new_value, history_ocr.raw_text, history_ocr.confidence, history_ocr.all_multipliers)
        if mode == "both":
            return self.reader.read_center(image, fast=True)
        return OCRResult(None, history_ocr.raw_text, history_ocr.confidence, history_ocr.all_multipliers)

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
            fingerprint = hashlib.sha256(f"{self.session_id}:{result_number}:{ocr.multiplier}:{source}".encode()).hexdigest()
            row = GameResult(
                session_id=self.session_id,
                result_number=result_number,
                multiplier=ocr.multiplier,
                result_type=result_type,
                loss_streak_before=loss_before + (1 if result_type == "LOSS" else 0),
                win_streak_before=win_before + (1 if result_type == "WIN" else 0),
                timestamp=utc_now(),
                source=f"web-{source}",
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
            summary = summarize_results(prior + [row], threshold)
            session_model.total_results = summary.total
            session_model.total_wins = summary.wins
            session_model.total_losses = summary.losses
            session_model.maximum_loss_streak = summary.max_loss_streak
            session_model.maximum_win_streak = summary.max_win_streak
            return True


@app.on_event("startup")
def startup() -> None:
    logging.basicConfig(level=logging.INFO)
    initialize_database()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML


@app.get("/api/screenshot")
def screenshot() -> dict:
    image = latest_browser_frame()
    if image is None:
        image = ScreenCapture().capture(ScreenRegion(0, 0, 16000, 9000))
    return {"image": image_to_data_url(image), "width": image.width, "height": image.height}


@app.post("/api/browser-frame")
def receive_browser_frame(payload: BrowserFramePayload) -> dict:
    """Accept a frame from Chrome's getDisplayMedia tab-capture stream."""
    global browser_frame, browser_frame_at, browser_capture_enabled
    try:
        _, encoded = payload.image.split(",", 1)
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > 12_000_000:
            raise ValueError("Frame is too large")
        with Image.open(io.BytesIO(raw)) as decoded:
            decoded.load()
            frame = decoded.convert("RGB")
    except (ValueError, OSError, TypeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid browser capture frame.") from exc

    with browser_frame_lock:
        browser_frame = frame
        browser_frame_at = time.monotonic()
        browser_capture_enabled = True
    return {"ok": True, "width": frame.width, "height": frame.height}


@app.post("/api/browser-capture/stop")
def stop_browser_capture() -> dict:
    global browser_frame, browser_frame_at, browser_capture_enabled
    with browser_frame_lock:
        browser_capture_enabled = False
        browser_frame = None
        browser_frame_at = 0.0
    return {"ok": True}


@app.get("/api/settings")
def get_settings() -> dict:
    with get_session() as db:
        settings = db.get(AppSettings, 1) or AppSettings(id=1)
        db.merge(settings)
        return {
            "screen_x": settings.screen_x,
            "screen_y": settings.screen_y,
            "screen_width": settings.screen_width,
            "screen_height": settings.screen_height,
            "ocr_mode": settings.ocr_mode,
            "minimum_interval": float(settings.minimum_interval),
            "ocr_confidence_threshold": float(settings.ocr_confidence_threshold),
            "loss_threshold": float(settings.loss_threshold),
            "capture_interval": float(settings.capture_interval),
        }


@app.post("/api/region")
def save_region(payload: RegionPayload) -> dict:
    if payload.width < 5 or payload.height < 5:
        raise HTTPException(status_code=400, detail="Region is too small.")
    with get_session() as db:
        settings = db.get(AppSettings, 1) or AppSettings(id=1)
        settings.screen_x = payload.x
        settings.screen_y = payload.y
        settings.screen_width = payload.width
        settings.screen_height = payload.height
        settings.ocr_mode = "history"
        settings.capture_interval = Decimal("0.40")
        settings.ocr_confidence_threshold = Decimal("65.00")
        db.merge(settings)
    return {"ok": True}


@app.post("/api/settings")
def save_settings(payload: SettingsPayload) -> dict:
    with get_session() as db:
        settings = db.get(AppSettings, 1) or AppSettings(id=1)
        settings.ocr_mode = payload.ocr_mode
        settings.minimum_interval = Decimal(str(payload.minimum_interval))
        settings.ocr_confidence_threshold = Decimal(str(payload.ocr_confidence_threshold))
        settings.loss_threshold = Decimal(str(payload.loss_threshold))
        settings.capture_interval = Decimal(str(payload.capture_interval))
        db.merge(settings)
    return {"ok": True}


@app.post("/api/test-ocr")
def test_ocr() -> dict:
    with get_session() as db:
        settings = db.get(AppSettings, 1) or AppSettings(id=1)
        region = ScreenRegion(settings.screen_x, settings.screen_y, settings.screen_width, settings.screen_height)
        mode = settings.ocr_mode
    image = capture_region(region)
    if image is None:
        raise HTTPException(status_code=503, detail="Waiting for a frame from the shared browser tab.")
    if image.width * image.height > 8_000_000:
        raise HTTPException(
            status_code=413,
            detail=(
                "Selected region is too large for fast OCR. Use Auto Select Left Strip, "
                "then reduce W/H so only the multiplier history is inside the green box."
            ),
        )
    image = limit_ocr_image(image)
    started_at = time.monotonic()
    rows, raw = parse_table_rows(image)
    elapsed = time.monotonic() - started_at
    if elapsed > 20:
        raise HTTPException(
            status_code=504,
            detail="OCR test took too long. Shrink the green box so it covers only the latest history numbers.",
        )
    return {
        "rows": rows,
        "raw_text": raw,
        "confidence": round(rows[0]["confidence"], 2) if rows else 0,
        "preview": image_to_data_url(image),
        "elapsed": round(elapsed, 2),
    }


@app.post("/api/start")
def start() -> dict:
    global active_session_id, worker
    if worker and worker.is_alive():
        if not stop_event.is_set():
            return {"ok": True, "session_id": active_session_id}
        worker.join(timeout=2)
        if worker.is_alive():
            raise HTTPException(status_code=409, detail="Capture is still stopping. Try again in a moment.")
    stop_event.clear()
    worker = TableCaptureWorker()
    worker.start()
    return {"ok": True}


@app.post("/api/stop")
def stop() -> dict:
    global active_session_id
    stop_event.set()
    active_session_id = None
    return {"ok": True}


@app.get("/api/status")
def status() -> dict:
    latest_events = []
    while True:
        try:
            latest_events.append(events.get_nowait().model_dump())
        except queue.Empty:
            break
    running = bool(worker and worker.is_alive() and not stop_event.is_set())
    return {"recording": running, "status": "CAPTURING" if running else "STOPPED", "session_id": active_session_id, "events": latest_events}


def _persist_wingo_analyses(db, analysis: dict) -> None:
    known = {row.analysis_period for row in db.scalars(select(WingoAnalysis))}
    for record in analysis["backtest_records"]:
        if record["analysis_period"] in known:
            continue
        db.add(WingoAnalysis(
            analysis_period=record["analysis_period"],
            actual_period=record["actual_period"],
            big_score=Decimal(str(record["big_score"])),
            small_score=Decimal(str(record["small_score"])),
            signal=record["signal"],
            signal_strength=record["signal_strength"],
            sample_size=record["sample_size"],
            data_quality=record["data_quality"],
            backtest_accuracy=Decimal(str(record["backtest_accuracy"])) if record["backtest_accuracy"] is not None else None,
            actual_number=record["actual_number"],
            actual_size=record["actual_size"],
            outcome=record["outcome"],
            evaluation_mode="BACKTEST",
        ))
        known.add(record["analysis_period"])

    period = analysis.get("period")
    if period and period not in known:
        db.add(WingoAnalysis(
            analysis_period=period,
            big_score=Decimal(str(analysis["big_score"])),
            small_score=Decimal(str(analysis["small_score"])),
            signal=analysis["signal"],
            signal_strength=analysis["signal_strength"],
            sample_size=analysis["sample_size"],
            data_quality=analysis["data_quality"],
            backtest_accuracy=Decimal(str(analysis["backtest"]["accuracy"])),
            evaluation_mode="LIVE",
        ))


@app.get("/api/results")
def results(limit: int = 120) -> dict:
    # Keep the full read/compute/write transaction serialized. Locking only the
    # insert still allows a waiting request to keep a stale MySQL snapshot.
    with analysis_persist_lock:
        return _build_results(limit)


def _build_results(limit: int) -> dict:
    global pattern_index_latest_id, analysis_cache_latest_id, analysis_cache
    with get_session() as db:
        rows = list(db.scalars(select(TableCapture).order_by(TableCapture.period, TableCapture.id)))
        pattern_rows = list(db.scalars(select(PatternIdentifier)))
        if rows and (pattern_index_latest_id != rows[-1].id or not pattern_rows):
            _sync_pattern_identifiers(db, rows)
            db.flush()
            pattern_rows = list(db.scalars(select(PatternIdentifier)))

        latest_id = (rows[-1].id, len(rows)) if rows else (0, 0)
        with analysis_cache_lock:
            if analysis_cache is None or latest_id != analysis_cache_latest_id:
                analysis_cache = analyze_history(rows)
                db.flush()
                analysis_cache_latest_id = latest_id
            analysis = analysis_cache

        latest_capture_time = rows[-1].captured_at if rows else None
        audit_query = select(CaptureAudit).order_by(CaptureAudit.created_at.desc(), CaptureAudit.id.desc())
        audit_query = audit_query.where(CaptureAudit.created_at > latest_capture_time) if latest_capture_time else audit_query
        audits = list(db.scalars(audit_query.limit(100)))
        active_faults = [audit for audit in audits if audit.status in {"INVALID RESULT", "ABNORMAL JUMP"}]
        if active_faults:
            analysis["data_quality"] = "LOW"
            analysis["signal"] = "NO CLEAR SIGNAL"
            analysis["signal_status"] = "DATA QUALITY LOW"
            analysis["signal_strength"] = "VERY LOW"
            analysis["historical_signal_strength"] = "LOW"
            analysis["data_quality_flags"] = list(dict.fromkeys(analysis["data_quality_flags"] + [f"capture validation event: {audit.status.lower()}" for audit in active_faults]))
        _persist_wingo_analyses(db, analysis)
        db.flush()

        live_rows = list(db.scalars(select(WingoAnalysis).where(WingoAnalysis.evaluation_mode == "LIVE").order_by(WingoAnalysis.created_at, WingoAnalysis.id)))
        live_signals = [row for row in live_rows if row.signal in {"BIG", "SMALL"}]
        resolved = [row for row in live_signals if row.outcome in {"WIN", "LOSS"}]
        finished = [row for row in live_signals if row.outcome is not None]
        wins = sum(row.outcome == "WIN" for row in resolved)
        losses = len(resolved) - wins
        low_ocr = sum(row.confidence is not None and float(row.confidence) < 65 for row in rows)
        confidences = [float(row.confidence) for row in rows if row.confidence is not None]
        analysis_view = {key: value for key, value in analysis.items() if key != "backtest_records"}
        return {
            "summary": {"total": len(rows), "valid": sum(row_is_valid(row) for row in rows), "low_ocr": low_ocr, "mean_ocr": round(sum(confidences) / len(confidences), 1) if confidences else 0.0},
            "analysis": analysis_view,
            "backtest": analysis["backtest"],
            "live_kpis": {"signals": len(live_signals), "pending": len(live_signals) - len(finished), "skipped": sum(row.outcome == "SKIPPED_MISSING_PERIOD" for row in finished), "wins": wins, "losses": losses, "win_rate": round(wins * 100 / len(resolved), 1) if resolved else None, "confidence_interval": _wilson_interval(wins, len(resolved))},
            "windows": analysis["windows"],
            "transitions": analysis["transition"],
            "number_frequency": analysis["number_frequency"],
            "sequence_patterns": analysis["sequence_patterns"],
            "patterns": [
                {"pattern": row.pattern, "occurrences": row.occurrences, "next_big": row.next_big_count, "next_small": row.next_small_count, "matched": row.next_big_count + row.next_small_count,
                 "agreement": round(max(row.next_big_count, row.next_small_count) * 100 / max(1, row.next_big_count + row.next_small_count), 1)}
                for row in sorted(pattern_rows, key=lambda item: (-item.occurrences, item.pattern))
            ],
            "analysis_records": [
                {"period": row.analysis_period, "actual_period": row.actual_period or "Pending", "signal": row.signal, "actual": row.actual_size or "Pending", "outcome": row.outcome or "Pending", "strength": row.signal_strength, "sample": row.sample_size, "quality": row.data_quality, "mode": row.evaluation_mode}
                for row in db.scalars(select(WingoAnalysis).order_by(WingoAnalysis.created_at.desc(), WingoAnalysis.id.desc()).limit(100))
            ],
            "audit_events": [{"period": row.period or "—", "status": row.status, "message": row.message, "time": row.created_at.astimezone().strftime("%Y-%m-%d %H:%M:%S")} for row in audits],
            "results": [
                {"id": row.id, "period": row.period, "number": row.number, "size": row.size, "time": row.captured_at.astimezone().strftime("%H:%M:%S") if row.captured_at else "", "confidence": float(row.confidence or 0), "data_status": row.data_status or "Waiting for validation", "previous_period": row.previous_period, "previous_number": row.previous_number, "source_status": row.source_status or "Unknown"}
                for row in reversed(rows[-limit:])
            ],
        }


def _wilson_interval(wins: int, total: int, z: float = 1.96) -> list[float] | None:
    if total <= 0:
        return None
    rate = wins / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * ((rate * (1 - rate) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    return [round(max(0, center - margin) * 100, 1), round(min(1, center + margin) * 100, 1)]


def _analytics_values() -> tuple[list[object], float]:
    """Single repository boundary for all read-only analytics endpoints."""
    with get_session() as db:
        settings = db.get(AppSettings, 1) or AppSettings(id=1)
        rows = list(db.scalars(select(GameResult).order_by(GameResult.timestamp, GameResult.id)))
        return [row.multiplier for row in rows], float(settings.loss_threshold)


@app.get("/api/analytics/overview")
def analytics_overview() -> dict:
    values, threshold = _analytics_values()
    return overview(values, threshold)


@app.get("/api/analytics/loss-streaks")
def analytics_loss_streaks() -> dict:
    values, threshold = _analytics_values()
    return {"threshold": threshold, "records": loss_streaks(values, threshold)}


@app.get("/api/analytics/patterns")
def analytics_patterns() -> dict:
    values, threshold = _analytics_values()
    return {"threshold": threshold, "pattern_length": 3, "records": patterns(values, threshold)}


@app.get("/api/analytics/walk-forward")
def analytics_walk_forward() -> dict:
    values, threshold = _analytics_values()
    return walk_forward(values, threshold)


@app.get("/api/analytics/recent")
def analytics_recent() -> dict:
    values, threshold = _analytics_values()
    return {"threshold": threshold, "records": recent_comparison(values, threshold)}


@app.get("/api/agent")
def agent_view(limit: int = 120) -> dict:
    """Analyzer agent: settle old signals, learn, issue the next signal, and return UI data."""
    with analysis_persist_lock:
        with get_session() as db:
            rows = list(db.scalars(select(TableCapture).order_by(TableCapture.period, TableCapture.id)))
            latest_time = rows[-1].captured_at if rows else None
            fault_query = select(CaptureAudit.id).where(CaptureAudit.status.in_(["INVALID RESULT", "ABNORMAL JUMP"]))
            if latest_time:
                fault_query = fault_query.where(CaptureAudit.created_at > latest_time)
            has_fault = bool(rows) and db.scalar(fault_query.limit(1)) is not None
            state = agent_service.tick(db, rows, quality_ok=not has_fault, quality_note="Capture validation problem: signals paused until the next clean result")
            snap = agent_service.snapshot(db, rows, limit)
            return {
                "next_period": state["target_period"],
                "decision": state["decision"],
                "strategies": [{"name": s["name"], "version": s["version"], "enabled": s["enabled"], "min_matches": s["min_matches"], "min_agreement": s["min_agreement"]} for s in state["strategies"]],
                "total_results": len(rows),
                **snap,
            }


@app.post("/api/agent/reset")
def agent_reset() -> dict:
    with analysis_persist_lock:
        with get_session() as db:
            agent_service.reset_agent(db)
    return {"ok": True}


HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WinGo Pattern Analyzer</title>
  <style>
    :root { color-scheme: dark; font-family: Segoe UI, Arial, sans-serif; background:#11151c; color:#eef2f8; }
    body { margin:0; background:#11151c; }
    header { height:58px; display:flex; align-items:center; justify-content:space-between; padding:0 18px; background:#191f2a; border-bottom:1px solid #2d3545; position:sticky; top:0; z-index:5; }
    h1 { margin:0; font-size:18px; letter-spacing:.08em; }
    button, select, input { background:#263244; border:1px solid #3a4659; color:#f8fafc; border-radius:6px; padding:9px 11px; }
    button { cursor:pointer; font-weight:650; }
    button.primary { background:#f5b814; color:#101319; border-color:#f5b814; }
    main { display:grid; grid-template-columns: 360px 1fr; gap:14px; padding:14px; align-items:start; }
    section { background:#171d27; border:1px solid #2d3545; border-radius:8px; padding:12px; }
    .controls { display:grid; gap:8px; }
    .row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
    .grid { display:grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap:10px; margin-bottom:14px; }
    .card { background:#202838; border:1px solid #344055; border-radius:8px; padding:12px; }
    .label { font-size:11px; color:#9fb0c7; text-transform:uppercase; }
    .value { font-size:24px; font-weight:750; margin-top:4px; }
    .muted { color:#9fb0c7; font-size:13px; }
    #screenTools { display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:8px; }
    #screenWrap { position:relative; overflow:auto; max-height:560px; border:1px solid #344055; background:#080a0e; cursor:crosshair; user-select:none; }
    #screen { max-width:none; display:block; }
    #screen.fit { width:100%; height:auto; }
    #selection { position:absolute; border:2px solid #39ff88; background:rgba(57,255,136,.16); pointer-events:none; display:none; }
    .regionFields { display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap:8px; }
    .regionFields label { display:grid; gap:4px; font-size:11px; color:#9fb0c7; text-transform:uppercase; }
    .regionFields input { width:100%; box-sizing:border-box; }
    table { width:100%; border-collapse:collapse; font-size:13px; }
    th, td { padding:8px; border-bottom:1px solid #2a3342; text-align:left; white-space:nowrap; }
    th { position:sticky; top:0; background:#202838; z-index:1; }
    .tableWrap { max-height:420px; overflow:auto; border:1px solid #2d3545; border-radius:8px; }
    .analysis { max-height:300px; overflow:auto; }
    .notice { border-left:4px solid #f5b814; padding:10px; background:#202838; color:#dce7f7; }
    #log { min-height:42px; max-height:120px; overflow:auto; font-family:Consolas, monospace; color:#bfd2ee; }
    .good { color:#5eea8a; } .bad { color:#ff6b6b; } .warn { color:#ffd166; }
    .signalPanel { grid-column:1 / -1; border-color:#53657e; background:linear-gradient(135deg,#1b2635,#171d27 68%); }
    .signalPanel h2 { margin-top:2px; }
    .signalHero { display:flex; justify-content:space-between; align-items:center; gap:18px; margin:12px 0; padding:18px; border:1px solid #465772; border-radius:10px; background:#111823; }
    .targetPeriod { font:750 clamp(22px,4vw,38px) Consolas,monospace; margin-top:5px; }
    .mainSignal { font-size:clamp(26px,4vw,40px); font-weight:800; text-align:right; }
    .bigSignal { color:#ff7676; } .smallSignal { color:#60b8ff; } .noSignal { color:#ffd166; }
    .agentGrid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; margin:12px 0; }
    .agentCard { background:#202838; border:1px solid #344055; border-radius:8px; padding:12px; }
    .agentCard strong { display:block; margin:5px 0; }
    @media (max-width:900px) { main { grid-template-columns:1fr; } .signalHero { align-items:flex-start; flex-direction:column; } .mainSignal { text-align:left; } }
    .patternReadout { padding:10px 12px; margin:10px 0; background:#111823; border:1px solid #344055; border-radius:7px; overflow-wrap:anywhere; }
    .tableWrap { max-height:280px; }
  
    .full { grid-column:1 / -1; }
    .kpis { grid-template-columns:repeat(7,minmax(100px,1fr)); margin:0; }
    @media (max-width:1100px) { .kpis { grid-template-columns:repeat(2,1fr); } }
    .why { white-space:normal; min-width:320px; color:#bfd2ee; font-size:12px; }
    .chip { display:inline-block; margin:0 6px 6px 0; padding:4px 9px; border:1px solid #3a4659; border-radius:999px; font-size:12px; background:#202838; }
    .chip.off { opacity:.45; text-decoration:line-through; }
    .lesson { border-left:3px solid #3a4659; padding:6px 10px; margin:8px 0; background:#111823; border-radius:0 6px 6px 0; font-size:13px; }
    .lesson.WIN { border-color:#5eea8a; } .lesson.LOSS { border-color:#ff6b6b; } .lesson.UPGRADE { border-color:#f5b814; } .lesson.DISABLE { border-color:#ff9f43; }
    .lesson .tag { font-size:10px; font-weight:800; letter-spacing:.08em; }
    .idea { color:#ffd166; margin-top:3px; }
    #lessons { max-height:420px; overflow:auto; }
    .tableWrap { max-height:520px; }
</style>
</head>
<body>
<header>
  <h1>WINGO PATTERN ANALYZER</h1>
  <div><span id="status" class="warn">● STOPPED</span></div>
</header>
<main>
  <section class="signalPanel">
    <div class="row" style="justify-content:space-between"><h2>Next Period Signal</h2><button id="soundButton" onclick="toggleSignalSound()">Enable Signal Sound</button></div>
    <div class="signalHero">
      <div><div class="label">NEXT PERIOD</div><div id="nextTarget" class="targetPeriod">Waiting for captured results</div></div>
      <div><div class="label">WILL BE</div><div id="mainSignal" class="mainSignal noSignal">NO SIGNAL</div></div>
    </div>
    <div id="signalWhy" class="patternReadout">Waiting for captured results.</div>
    <div id="signalEdge" class="muted"></div>
    <p class="muted">The agent compares the current scenario with every earlier record and signals only when enough past cases agree. Outcomes of a random game cannot be guaranteed; check the “vs 50% chance” card before trusting any signal.</p>
  </section>
  <div class="grid kpis full" id="kpis"></div>
  <section>
    <h2>Automatic table capture</h2>
    <p class="muted">Choose the browser tab that shows the results table, then select the table rows.</p>
    <div class="controls">
      <button class="primary" onclick="chooseBrowserTab()">Choose Game Browser Tab</button>
      <button onclick="loadScreenshot()">Refresh Selected Tab Preview</button>
      <button onclick="stopBrowserCapture()">Stop Tab Capture</button>
      <div id="screenTools">
        <button onclick="setPreviewFit(true)">Fit Preview</button>
        <button onclick="setPreviewFit(false)">Actual Size</button>
        <button class="primary" onclick="selectLeftStrip()">Auto Select Left Strip</button>
        <button onclick="clearSelection()">Clear Selection</button>
      </div>
      <div id="screenWrap"><img id="screen"><div id="selection"></div></div>
      <div class="regionFields">
        <label>X <input id="regionX" type="number" min="0" step="1"></label>
        <label>Y <input id="regionY" type="number" min="0" step="1"></label>
        <label>W <input id="regionW" type="number" min="1" step="1"></label>
        <label>H <input id="regionH" type="number" min="1" step="1"></label>
      </div>
      <button class="primary" onclick="saveRegion()">Save Selected Table Region</button>
      <div class="row"><label class="muted">OCR retry interval (seconds) <input id="captureInterval" type="number" min="0.25" step="0.1" value="0.4"></label></div>
      <button onclick="saveSettings()">Save Capture Settings</button>
      <button onclick="testOcr()">Fast Test OCR</button>
      <button class="primary" onclick="startRecording()">Start Capture</button>
      <button onclick="stopRecording()">Stop Capture</button>
      <div id="ocrPreview"></div>
      <div id="log"></div>
    </div>
  </section>
  <section>
    <div class="row" style="justify-content:space-between"><h2>Agent learning log</h2><button onclick="resetAgent()">Reset agent</button></div>
    <p class="muted">Every settled signal is explained here. After every 10 settled signals the agent reviews its strategies and only keeps a change if it also works on newer data it was not tuned on.</p>
    <div id="strategyChips"></div>
    <div id="lessons"></div>
  </section>
  <section class="full">
    <div class="row" style="justify-content:space-between"><h2>Period records</h2><span id="recordCount" class="muted"></span></div>
    <div class="tableWrap"><table><thead><tr><th>Period</th><th>Number</th><th>Big / Small</th><th>Signal</th><th>Result</th><th>Why</th></tr></thead><tbody id="records"></tbody></table></div>
  </section>
</main>
<script>
let shot = {w:0,h:0}, drag = null, selected = null, previewFit = true;
const img = document.getElementById('screen'), sel = document.getElementById('selection'), wrap = document.getElementById('screenWrap');
const regionX = document.getElementById('regionX'), regionY = document.getElementById('regionY');
const regionW = document.getElementById('regionW'), regionH = document.getElementById('regionH');
const regionInputs = ['X','Y','W','H'].map(id => document.getElementById('region' + id));
const captureVideo = document.createElement('video'), captureCanvas = document.createElement('canvas');
let captureStream = null, captureTimer = null, frameUploadInProgress = false;
let signalAudio = null, signalSoundEnabled = false, lastSeenSignal = null, firstSignalCheck = true;
captureVideo.muted = true; captureVideo.playsInline = true;
function log(msg, cls='') { const el=document.getElementById('log'); el.innerHTML = `<div class="${cls}">${new Date().toLocaleTimeString()} ${msg}</div>` + el.innerHTML; }
async function toggleSignalSound(){
  signalSoundEnabled=!signalSoundEnabled;
  const button=document.getElementById('soundButton');
  button.textContent=signalSoundEnabled?'Signal Sound: ON':'Enable Signal Sound';
  if(signalSoundEnabled){
    try { signalAudio=signalAudio||new (window.AudioContext||window.webkitAudioContext)(); await signalAudio.resume(); log('Signal sound enabled','good'); }
    catch (_) { signalSoundEnabled=false; button.textContent='Enable Signal Sound'; log('Audio is unavailable in this browser','warn'); }
  } else log('Signal sound disabled','warn');
}
function playSignalPop(){
  if(!signalSoundEnabled||!signalAudio) return;
  const now=signalAudio.currentTime, oscillator=signalAudio.createOscillator(), gain=signalAudio.createGain();
  oscillator.type='sine'; oscillator.frequency.setValueAtTime(880,now); oscillator.frequency.exponentialRampToValueAtTime(1320,now+0.07);
  gain.gain.setValueAtTime(0.0001,now); gain.gain.exponentialRampToValueAtTime(0.22,now+0.012); gain.gain.exponentialRampToValueAtTime(0.0001,now+0.16);
  oscillator.connect(gain); gain.connect(signalAudio.destination); oscillator.start(now); oscillator.stop(now+0.17);
}
function resetSelection(){ selected=null; drag=null; sel.style.display='none'; regionInputs.forEach(input=>input.value=''); }
function clearSelection(){ resetSelection(); log('Selection cleared.', 'warn'); }
function displayScale(){
  if(!shot.w || !img.getBoundingClientRect().width) return {x:1,y:1};
  const b=img.getBoundingClientRect();
  return {x:b.width/shot.w, y:b.height/shot.h};
}
function syncRegionInputs(){
  if(!selected) return;
  regionX.value=selected.x; regionY.value=selected.y; regionW.value=selected.width; regionH.value=selected.height;
}
function drawSelection(){
  if(!selected || !shot.w || !shot.h){ sel.style.display='none'; return; }
  const scale=displayScale();
  Object.assign(sel.style,{
    display:'block',
    left:(selected.x*scale.x)+'px',
    top:(selected.y*scale.y)+'px',
    width:(selected.width*scale.x)+'px',
    height:(selected.height*scale.y)+'px'
  });
}
function setSelected(region, announce=true){
  if(!shot.w || !shot.h) return log('Load the selected tab preview first.', 'warn');
  const x=Math.max(0, Math.min(shot.w-1, Math.round(region.x)));
  const y=Math.max(0, Math.min(shot.h-1, Math.round(region.y)));
  const width=Math.max(1, Math.min(shot.w-x, Math.round(region.width)));
  const height=Math.max(1, Math.min(shot.h-y, Math.round(region.height)));
  selected={x,y,width,height}; syncRegionInputs(); drawSelection();
  if(announce) log(`Selected x=${x}, y=${y}, w=${width}, h=${height}`, 'good');
}
function setPreviewFit(fit){
  previewFit=fit; img.classList.toggle('fit', fit); requestAnimationFrame(drawSelection);
}
function selectLeftStrip(){
  setSelected({x:0, y:0, width:Math.round(shot.w*0.26), height:shot.h});
  log('Left strip preset selected. Adjust X/Y/W/H if the green box needs tuning.', 'good');
}
function drawCapturedTab(){
  if(!captureVideo.videoWidth || !captureVideo.videoHeight) return null;
  captureCanvas.width=captureVideo.videoWidth; captureCanvas.height=captureVideo.videoHeight;
  captureCanvas.getContext('2d').drawImage(captureVideo,0,0);
  return captureCanvas.toDataURL('image/jpeg',0.88);
}
async function uploadBrowserFrame(){
  if(!captureStream || frameUploadInProgress) return;
  const frame=drawCapturedTab(); if(!frame) return;
  frameUploadInProgress=true;
  try { await fetch('/api/browser-frame',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image:frame})}); }
  catch (_) { /* The next frame will retry if the local server is briefly busy. */ }
  finally { frameUploadInProgress=false; }
}
async function chooseBrowserTab(){
  if(!navigator.mediaDevices?.getDisplayMedia){ log('This browser does not support tab capture. Use a current Chrome version.', 'bad'); return; }
  await stopBrowserCapture(false);
  try {
    captureStream=await navigator.mediaDevices.getDisplayMedia({
      video:{displaySurface:'browser'}, audio:false, preferCurrentTab:false,
      selfBrowserSurface:'exclude', surfaceSwitching:'include'
    });
    captureVideo.srcObject=captureStream;
    await new Promise(resolve => captureVideo.onloadedmetadata=resolve);
    await captureVideo.play();
    captureStream.getVideoTracks()[0].addEventListener('ended',()=>stopBrowserCapture(true));
    resetSelection(); await loadScreenshot(); await uploadBrowserFrame();
    captureTimer=setInterval(uploadBrowserFrame,400);
    log('Game tab connected. You may return to this tracker tab; recording uses the selected game tab.', 'good');
  } catch (err) { log('Tab capture was cancelled or could not start.', 'warn'); }
}
async function loadScreenshot(){
  const frame=drawCapturedTab();
  if(!frame) { log('Choose the game browser tab first.', 'warn'); return; }
  img.src=frame; shot={w:captureCanvas.width,h:captureCanvas.height};
  img.classList.toggle('fit', previewFit);
  await uploadBrowserFrame();
  log(`Selected-tab preview loaded ${shot.w}x${shot.h}`);
  requestAnimationFrame(drawSelection);
}
async function stopBrowserCapture(fromTrack=false){
  if(captureTimer){ clearInterval(captureTimer); captureTimer=null; }
  const oldStream=captureStream; captureStream=null; captureVideo.srcObject=null;
  if(oldStream && !fromTrack) oldStream.getTracks().forEach(track=>track.stop());
  try { await fetch('/api/browser-capture/stop',{method:'POST'}); } catch (_) {}
  if(oldStream) log('Tab capture stopped.', 'warn');
}
function point(e){
  const b=img.getBoundingClientRect();
  const displayX=Math.max(0, Math.min(b.width, e.clientX-b.left));
  const displayY=Math.max(0, Math.min(b.height, e.clientY-b.top));
  return {x:Math.round(displayX*(shot.w/b.width)), y:Math.round(displayY*(shot.h/b.height))};
}
img.addEventListener('mousedown', e => { e.preventDefault(); drag = point(e); sel.style.display='block'; });
window.addEventListener('mousemove', e => {
  if(!drag) return; const p=point(e);
  setSelected({x:Math.min(drag.x,p.x), y:Math.min(drag.y,p.y), width:Math.abs(p.x-drag.x), height:Math.abs(p.y-drag.y)}, false);
});
window.addEventListener('mouseup', e => {
  if(!drag) return;
  const start=drag, p=point(e); drag=null;
  setSelected({x:Math.min(start.x,p.x), y:Math.min(start.y,p.y), width:Math.abs(p.x-start.x), height:Math.abs(p.y-start.y)}, false);
  if(selected) log(`Selected x=${selected.x}, y=${selected.y}, w=${selected.width}, h=${selected.height}`, 'good');
});
regionInputs.forEach(input => input.addEventListener('change', () => {
  setSelected({x:Number(regionX.value||0), y:Number(regionY.value||0), width:Number(regionW.value||1), height:Number(regionH.value||1)});
}));
window.addEventListener('resize', drawSelection);
async function saveRegion(){
  if(!selected) return log('Select a region first', 'warn');
  const r = await fetch('/api/region',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(selected)});
  log(r.ok ? 'History region saved' : 'Could not save region', r.ok ? 'good':'bad');
}
async function saveSettings(){
  const payload = {ocr_mode:'history', minimum_interval:1.0, ocr_confidence_threshold:65, loss_threshold:1.5, capture_interval:Number(captureInterval.value)};
  await fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}); log('Settings saved','good');
}
async function testOcr(){
  log('Running fast OCR test...');
  const r = await fetch('/api/test-ocr',{method:'POST'}); const d = await r.json();
  document.getElementById('ocrPreview').innerHTML = `<p>Complete rows found: <b>${d.rows?.length || 0}</b> Confidence: ${d.confidence}%</p><p class="muted">Raw: ${d.raw_text || '-'}</p><img src="${d.preview}" style="max-width:100%;border:1px solid #344055">`;
  log(`OCR found ${d.rows?.length || 0} complete table rows`, d.rows?.length ? 'good':'warn');
}
async function startRecording(){ await fetch('/api/start',{method:'POST'}); log('Recording started','good'); refresh(); }
async function stopRecording(){ await fetch('/api/stop',{method:'POST'}); log('Recording stopped','warn'); refresh(); }
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
async function refresh(){
  const s=await (await fetch('/api/status')).json();
  const status=s.events.some(e=>e.kind==='error')?'OCR ERROR':s.events.some(e=>e.kind==='warning')?'INVALID / ABNORMAL':s.events.some(e=>e.kind==='duplicate')?'DUPLICATE':s.recording?(s.events.some(e=>e.kind==='result')?'CAPTURING':'WAITING'):'STOPPED';
  const st=document.getElementById('status'); st.textContent=`● ${status}`;
  st.className=status==='OCR ERROR'?'bad':['INVALID / ABNORMAL','DUPLICATE','STOPPED'].includes(status)?'warn':'good';
  for(const ev of s.events) log(`${ev.kind}: ${ev.message} ${ev.period||''}`,ev.kind==='error'?'bad':ev.kind==='warning'?'warn':'');
  const d=await (await fetch('/api/agent')).json(), dec=d.decision, k=d.kpis;
  document.getElementById('nextTarget').textContent=d.next_period||'Waiting for captured results';
  const sig=document.getElementById('mainSignal'); sig.textContent=dec.signal;
  sig.className=`mainSignal ${dec.signal==='BIG'?'bigSignal':dec.signal==='SMALL'?'smallSignal':'noSignal'}`;
  const directional=['BIG','SMALL'].includes(dec.signal);
  document.getElementById('signalWhy').innerHTML=directional
    ? `<b>${esc(dec.strategy)}</b> v${dec.version}: scenario “${esc(dec.context)}” appeared <b>${dec.matches}</b> times before and <b>${dec.agreement}%</b> were followed by <b>${dec.signal}</b>.`
    : esc(dec.status);
  const edge=dec.edge?` Track record of this strategy: ${dec.wf_hits}/${dec.wf_tested} correct in replay — <b>${esc(dec.edge)}</b>.`:'';
  document.getElementById('signalEdge').innerHTML=directional?edge:'';
  const key=directional&&d.next_period?`${d.next_period}:${dec.signal}`:null;
  if(firstSignalCheck){lastSeenSignal=key;firstSignalCheck=false;}
  else if(key&&key!==lastSeenSignal){lastSeenSignal=key;playSignalPop();log(`New signal for ${d.next_period}: ${dec.signal}`,'good');}
  const rate=k.win_rate==null?'—':`${k.win_rate}%`, range=k.interval?`${k.interval[0]}–${k.interval[1]}%`:'—';
  document.getElementById('kpis').innerHTML=[
    ['SIGNALS ISSUED',k.signals,''],['WINS',k.wins,'good'],['LOSSES',k.losses,'bad'],['WIN RATE',rate,''],
    ['95% RANGE',range,''],['PENDING',k.pending,''],['VS 50% CHANCE',k.verdict,k.verdict==='EDGE FOUND'?'good':k.verdict==='WORSE THAN CHANCE'?'bad':'warn']
  ].map(([l,v,c])=>`<div class="card"><div class="label">${l}</div><div class="value ${c}" style="${String(v).length>8?'font-size:15px;margin-top:10px':''}">${esc(v)}</div></div>`).join('');
  document.getElementById('strategyChips').innerHTML=d.strategies.map(x=>`<span class="chip ${x.enabled?'':'off'}">${esc(x.name)} v${x.version} · ≥${x.min_matches} matches · ≥${x.min_agreement}%${x.enabled?'':' · OFF'}</span>`).join('');
  document.getElementById('lessons').innerHTML=d.lessons.map(l=>`<div class="lesson ${l.kind}"><span class="tag">${l.kind}</span> <span class="muted">${esc(l.time)}</span><div>${esc(l.text)}</div>${l.idea?`<div class="idea">Upgrade idea: ${esc(l.idea)}</div>`:''}</div>`).join('')||'<p class="muted">No lessons yet. Every settled signal will be explained here.</p>';
  document.getElementById('recordCount').textContent=`${d.total_results} captured results`;
  document.getElementById('records').innerHTML=d.records.map(r=>`<tr><td>${esc(r.period)}</td><td>${r.number}</td><td class="${r.size==='BIG'?'bigSignal':'smallSignal'}">${r.size}</td><td>${esc(r.signal)}</td><td class="${r.outcome==='WIN'?'good':r.outcome==='LOSS'?'bad':'muted'}">${esc(r.outcome)}</td><td class="why">${esc(r.reason)}</td></tr>`).join('')||'<tr><td colspan="6">Waiting for captured results.</td></tr>';
}
async function resetAgent(){ if(!confirm('Erase all agent signals, lessons and strategy upgrades and restart from defaults?')) return; await fetch('/api/agent/reset',{method:'POST'}); log('Agent reset to default strategies','warn'); refresh(); }
refresh(); setInterval(refresh, 1500);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web_app:app", host="127.0.0.1", port=8000, reload=False)
