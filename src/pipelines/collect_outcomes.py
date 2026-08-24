"""The outcomes run: revisit scored papers at t+7, t+30 and t+90.

This table is the asset. It is the one thing here that cannot be regenerated, because
it can only be accumulated forwards in time.
"""

from __future__ import annotations

from ..outcomes import calibration, collect
from ..store import db
from ..util import dates
from ..util import logging as log
from ..util.config import db_path
from ..util.config import load as load_config
from ..util.http import Fetcher

LOG = log.get("pipeline.outcomes")


def run(
    *,
    config: dict | None = None,
    today: str | None = None,
    limit: int | None = None,
    offline: bool = False,
) -> dict:
    config = config or load_config()
    today = today or dates.today_str()
    conn = db.connect(db_path(config))
    run_id = db.start_run(conn, "outcomes", today)
    fetcher = Fetcher(offline=offline, cache_ttl_seconds=3600.0)
    try:
        stats = collect.collect_due(conn, fetcher, config, today=today, limit=limit)
        labelled = calibration.compute_labels(conn, config)
        summary = calibration.summarise(conn)
        stats.update({"labelled": labelled, "calibration": summary})
        db.finish_run(conn, run_id, "ok", stats)
        return stats
    except Exception as exc:  # noqa: BLE001
        db.finish_run(conn, run_id, "failed", {}, notes=str(exc)[:500])
        raise
    finally:
        fetcher.close()
        db.close(conn)
