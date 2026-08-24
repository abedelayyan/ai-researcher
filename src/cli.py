"""Command line entry points.

    python -m src.cli daily            ingest, score, log, render
    python -m src.cli outcomes         collect t+7 / t+30 / t+90 observations
    python -m src.cli calibration      print precision at k over the prediction log
    python -m src.cli render           re-render a digest from stored data
    python -m src.cli spend            model spend so far
"""

from __future__ import annotations

import argparse
import json
import sys

from .outcomes import calibration as calib
from .pipelines import collect_outcomes, daily
from .render import digest as render_digest
from .score import rank
from .store import db
from .util import dates
from .util import logging as log
from .util.config import db_path
from .util.config import load as load_config

LOG = log.get("cli")


def cmd_daily(args: argparse.Namespace) -> int:
    result = daily.run(
        run_date=args.date,
        lookback_hours=args.lookback_hours,
        limit=args.limit,
        offline=args.offline,
        skip_ingest=args.skip_ingest,
    )
    print(json.dumps({
        "run_id": result.run_id, "run_date": result.run_date, "ingested": result.ingested,
        "extracted": result.extracted, "scored": result.scored,
        "shortlisted": result.shortlisted, "spend": result.spend,
        "digest": str(result.paths["markdown"]),
    }, indent=2))
    return 0


def cmd_outcomes(args: argparse.Namespace) -> int:
    stats = collect_outcomes.run(today=args.date, limit=args.limit, offline=args.offline)
    print(json.dumps(stats, indent=2))
    return 0


def cmd_calibration(args: argparse.Namespace) -> int:
    conn = db.connect(db_path(load_config()))
    print(json.dumps(calib.summarise(conn, k=args.k), indent=2))
    rows = calib.missed_winners(conn)
    if rows:
        print("\nMissed winners, the expensive errors:")
        for row in rows:
            print(f"  {row['arxiv_id']}  score {row['score']:.2f}  outcome {row['composite']:.2f}  {row['title'][:70]}")
    db.close(conn)
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    config = load_config()
    date = args.date or dates.today_str()
    path = db_path(config)
    conn = db.connect(path)
    scoped = db.open_feature_scoped(path)
    ids = [r[0] for r in conn.execute(
        "SELECT arxiv_id FROM paper_scores WHERE run_date = ?", (date,)
    )]
    if not ids:
        ids = [r[0] for r in conn.execute(
            "SELECT arxiv_id FROM papers WHERE announced_date = ?", (date,)
        )]
    buckets = rank.rank_ids(scoped, ids, date, config, persist_log=False)
    summaries = {
        r["arxiv_id"]: r["summary"] or ""
        for r in conn.execute(
            "SELECT arxiv_id, summary FROM paper_scores WHERE run_date = ?", (date,)
        )
    }
    context = render_digest.build_context(
        date=date, buckets=buckets, summaries=summaries, scored_count=len(ids),
        config=config, calibration=calib.summarise(conn),
        lab_posts=render_digest.lab_posts_for(conn, date), spend={}, run_id="rerender",
    )
    paths = render_digest.write(context, config)
    print(paths["markdown"])
    db.close(scoped)
    db.close(conn)
    return 0


def cmd_spend(args: argparse.Namespace) -> int:
    conn = db.connect(db_path(load_config()))
    rows = conn.execute(
        "SELECT date(ts) AS day, count(*) AS calls, sum(input_tokens) AS tin,"
        " sum(output_tokens) AS tout, sum(cost_usd) AS cost FROM llm_calls"
        " GROUP BY day ORDER BY day DESC LIMIT ?", (args.days,)
    ).fetchall()
    for row in rows:
        print(f"{row['day']}  {row['calls']:4d} calls  {row['tin']:8d} in  {row['tout']:7d} out"
              f"  ${row['cost'] or 0:.4f}")
    db.close(conn)
    return 0


def cmd_init_db(args: argparse.Namespace) -> int:
    conn = db.connect(db_path(load_config()))
    print(f"schema ready at {db_path(load_config())}")
    db.close(conn)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="signal-zero", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_daily = sub.add_parser("daily", help="ingest, score and render today's digest")
    p_daily.add_argument("--date", help="run date, defaults to today (UTC)")
    p_daily.add_argument("--lookback-hours", type=float)
    p_daily.add_argument("--limit", type=int, help="cap papers scored, for a quick test run")
    p_daily.add_argument("--offline", action="store_true", help="cache only, no outbound calls")
    p_daily.add_argument("--skip-ingest", action="store_true")
    p_daily.set_defaults(func=cmd_daily)

    p_out = sub.add_parser("outcomes", help="collect due observations")
    p_out.add_argument("--date", help="treat this as today")
    p_out.add_argument("--limit", type=int)
    p_out.add_argument("--offline", action="store_true")
    p_out.set_defaults(func=cmd_outcomes)

    p_cal = sub.add_parser("calibration", help="precision at k over the prediction log")
    p_cal.add_argument("--k", type=int, default=10)
    p_cal.set_defaults(func=cmd_calibration)

    p_render = sub.add_parser("render", help="re-render a digest from stored data")
    p_render.add_argument("--date")
    p_render.set_defaults(func=cmd_render)

    p_spend = sub.add_parser("spend", help="model spend per day")
    p_spend.add_argument("--days", type=int, default=14)
    p_spend.set_defaults(func=cmd_spend)

    p_init = sub.add_parser("init-db", help="create the database and apply migrations")
    p_init.set_defaults(func=cmd_init_db)
    return parser


def main(argv: list[str] | None = None) -> int:
    log.setup()
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
