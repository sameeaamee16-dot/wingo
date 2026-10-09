from __future__ import annotations

from pathlib import Path

from statistics.analytics import results_to_dataframe


def export_excel(rows: list[object], path: str | Path) -> Path:
    output = Path(path)
    df = results_to_dataframe(rows)
    df.to_excel(output, index=False)
    return output
