"""Date handling. arXiv announces around 20:00 ET, so a run's t=0 day is not
always the calendar day the runner happens to wake up on."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

ISO_DATE = "%Y-%m-%d"


def utcnow() -> datetime:
    return datetime.now(UTC)


def today_str() -> str:
    return utcnow().strftime(ISO_DATE)


def to_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(value[:10], ISO_DATE).date()


def date_str(value: str | date | datetime) -> str:
    return to_date(value).strftime(ISO_DATE)


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed = datetime.strptime(text[:19], "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def arxiv_stamp(moment: datetime) -> str:
    """arXiv's submittedDate filter format, YYYYMMDDHHMM, in UTC."""
    return moment.astimezone(UTC).strftime("%Y%m%d%H%M")


def days_between(earlier: str | date, later: str | date) -> int:
    return (to_date(later) - to_date(earlier)).days


def shift(value: str | date, days: int) -> str:
    return (to_date(value) + timedelta(days=days)).strftime(ISO_DATE)


def lookback_window(
    last_run_iso: str | None,
    *,
    now: datetime | None = None,
    default_hours: float = 30.0,
    max_hours: float = 168.0,
) -> tuple[datetime, datetime]:
    """Work out the ingest window.

    A missed day widens the window rather than losing the papers, which is what makes a
    scheduled run safe to miss. Capped so a long outage cannot ask arXiv for a month in
    one go.
    """
    end = now or utcnow()
    start = end - timedelta(hours=default_hours)
    last_run = parse_iso(last_run_iso) if last_run_iso else None
    if last_run:
        # Overlap by an hour. Ingest is idempotent, so a repeat costs nothing.
        candidate = last_run - timedelta(hours=1)
        start = min(start, candidate)
    floor = end - timedelta(hours=max_hours)
    return max(start, floor), end
