from __future__ import annotations

import logging
from pathlib import Path

from config import config
from database.migrations import initialize_database


def configure_logging() -> None:
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[
            logging.FileHandler(config.logs_dir / "multiplier_tracker.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def main() -> None:
    configure_logging()
    initialize_database()
    import uvicorn

    uvicorn.run("web_app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
