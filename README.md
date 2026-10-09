# Table Capture Predictor

Automatic period, number, and size table capture

WinGo Historical Pattern Analyzer is a local Windows application that reads completed result rows from a user-selected screen region. It records the period, number, Big/Small classification, capture time, previous result, data status, and OCR source status. The analyzer derives Big/Small from the number, validates OCR classification and period order, and never fills missing results or fabricates a future period ID.

The dashboard reports several historical windows, number frequencies, streaks, alternation, repetitions, transitions, sequence continuations, transparent weighted tendency scores, conflict status, and a no-signal state. It stores a per-period analysis archive and reports strict chronological walk-forward accuracy, baselines, and accuracy by signal strength/window/condition. Results are descriptive historical evidence and do not guarantee future outcomes.

It does not click, bet, cash out, log in, collect credentials, inspect cookies, bypass security, or manipulate the game. Predictions are estimates from prior captured rows and cannot guarantee the next result.

## Features

- Local browser dashboard with screen selection, capture controls, status, result history, and analysis archive.
- User-selected screen capture region that works with different window sizes and monitor layouts.
- Table row OCR with duplicate checks, number-derived size validation, missing-period detection, and retries for incomplete rows.
- Historical analysis across windows of 5, 10, 20, 30, 50, and 100 results, including transition and sequence sample sizes.
- Conflict-aware signal strength and `NO CLEAR SIGNAL` behavior; no gambler's-fallacy or overdue-number claims.
- Walk-forward backtest and per-period analysis history. Low-quality capture data disables directional signals.
- MySQL storage through SQLAlchemy ORM and PyMySQL.

## Install

1. Install Python 3.11 or newer.
2. Install MySQL 8 or newer and make sure the server is running.
3. Install Tesseract OCR for Windows. The app auto-detects the default install path. If needed, set `TESSERACT_CMD` in `.env`.
4. Create or choose a MySQL user. The app can create the `multiplier_tracker` database if the user has permission.
5. Copy `.env.example` to `.env` and fill in your database password:

```env
DB_HOST=localhost
DB_PORT=3306
DB_NAME=multiplier_tracker
DB_USER=root
DB_PASSWORD=YOUR_PASSWORD
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

6. Install dependencies:

```powershell
cd F:\Predictor\multiplier_tracker
python -m pip install -r requirements.txt
```

7. Initialize the database:

```powershell
python -m database.migrations
```

8. Start the application:

```powershell
python app.py
```

Or start the browser-based localhost dashboard:

```powershell
python web_app.py
```

Then open:

```text
http://127.0.0.1:8000
```

## Workflow

1. Open the screen or browser tab containing the results table.
2. Run `python app.py` and visit `http://127.0.0.1:8000`.
3. Choose the browser tab or refresh the screen preview.
4. Drag over the table rows, then save the selected region.
5. Use `Fast Test OCR` to check that complete rows are recognized.
6. Click `Start Capture`. New periods are added once, as soon as all fields are readable.
7. Click `Stop Capture` to stop monitoring.

For browser tab capture, choose the browser tab that displays the WinGo result table, select the table region, save it, verify the complete rows with Fast Test OCR, and start capture. The analysis dashboard displays data quality and abstains when evidence is weak or capture continuity is broken.

## Database

Tables are created automatically:

- `sessions`
- `game_results` (legacy multiplier data)
- `table_captures`
- `pattern_identifiers`
- `wingo_analysis`
- `prediction_signals` (legacy signal data)
- `settings`

Existing capture databases receive additive metadata columns for previous-period references and validation/source status. Historical results are not replaced. The analyzer does not synthesize records for missing periods.

## Historical Analysis

Scores, signal strength, and backtest results describe captured historical data. They are not guaranteed probabilities for future rounds. `NO CLEAR SIGNAL` is expected when the sample is small, indicators conflict, OCR confidence is low, or a period gap is detected.

## Tests

```powershell
cd F:\Predictor\multiplier_tracker
pytest
```

The tests cover classification, OCR parsing, complete streak counting, duplicate detection, new-result detection, and basic ORM insertion.
