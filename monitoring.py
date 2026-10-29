import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Optional

# =====================================================
# 1. DATABASE CONFIGURATION
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent
DB_PATH = PROJECT_DIR / "monitoring.db"

# Serializes writes within this Python process. SQLite also coordinates
# access between separate connections/processes.
_write_lock = Lock()

VALID_STATUSES = {"success", "failure", "blocked"}


def _utc_now() -> str:
    """Return a timezone-aware UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@contextmanager
def _connect():
    """Open a SQLite connection and ensure transactions are closed safely."""
    connection = sqlite3.connect(
        str(DB_PATH),
        timeout=30,
        isolation_level="DEFERRED",
    )
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA foreign_keys = ON")
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


# =====================================================
# 2. INITIALIZE DATABASE
# =====================================================

def init_db() -> None:
    """Create monitoring tables and indexes if they do not already exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with _write_lock:
        with _connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS request_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc TEXT NOT NULL,
                    status TEXT NOT NULL
                        CHECK (status IN ('success', 'failure', 'blocked')),
                    latency_seconds REAL NOT NULL
                        CHECK (latency_seconds >= 0),
                    error_category TEXT,
                    endpoint TEXT NOT NULL DEFAULT '/ask',
                    request_id TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_request_metrics_timestamp
                ON request_metrics(timestamp_utc)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_request_metrics_status
                ON request_metrics(status)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_request_metrics_error_category
                ON request_metrics(error_category)
                """
            )


# Initialize on import so existing main.py imports continue to work.
init_db()


# =====================================================
# 3. RECORD REQUESTS
# =====================================================

def record_request(
    status: str,
    latency_seconds: float,
    error_category: Optional[str] = None,
    endpoint: str = "/ask",
    request_id: Optional[str] = None,
) -> None:
    """
    Persist one completed request.

    Compatible with the existing call:
        record_request(status, elapsed_seconds)

    status must be 'success', 'failure', or 'blocked'.
    Avoid storing employee question text or other personal information.
    """
    normalized_status = str(status).strip().lower()
    if normalized_status not in VALID_STATUSES:
        # Unknown status should not silently become a success/failure count.
        raise ValueError(
            f"Invalid status {status!r}. Expected one of: "
            f"{', '.join(sorted(VALID_STATUSES))}."
        )

    try:
        latency = float(latency_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("latency_seconds must be a non-negative number.") from exc

    if latency < 0:
        raise ValueError("latency_seconds must be non-negative.")

    category = (
        str(error_category).strip()[:100]
        if error_category is not None and str(error_category).strip()
        else None
    )
    endpoint_value = str(endpoint or "/ask").strip()[:100]
    request_id_value = (
        str(request_id).strip()[:100]
        if request_id is not None and str(request_id).strip()
        else None
    )

    with _write_lock:
        with _connect() as connection:
            connection.execute(
                """
                INSERT INTO request_metrics (
                    timestamp_utc, status, latency_seconds,
                    error_category, endpoint, request_id
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    _utc_now(),
                    normalized_status,
                    latency,
                    category,
                    endpoint_value,
                    request_id_value,
                ),
            )


# =====================================================
# 4. METRICS SNAPSHOT (EXISTING API COMPATIBLE)
# =====================================================

def get_metrics() -> dict:
    """
    Return all-time metrics in a shape compatible with the existing
    FastAPI /metrics endpoint.
    """
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT
                COUNT(*) AS total_requests,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END)
                    AS successful_requests,
                SUM(CASE WHEN status = 'failure' THEN 1 ELSE 0 END)
                    AS failed_requests,
                SUM(CASE WHEN status = 'blocked' THEN 1 ELSE 0 END)
                    AS blocked_requests,
                COALESCE(AVG(latency_seconds), 0) AS average_latency_seconds,
                COALESCE(SUM(latency_seconds), 0) AS total_latency_seconds
            FROM request_metrics
            """
        ).fetchone()

        categories = connection.execute(
            """
            SELECT COALESCE(error_category, 'uncategorized') AS category,
                   COUNT(*) AS count
            FROM request_metrics
            WHERE status = 'failure' AND error_category IS NOT NULL
            GROUP BY error_category
            ORDER BY count DESC, category ASC
            """
        ).fetchall()

    return {
        "total_requests": int(row["total_requests"] or 0),
        "successful_requests": int(row["successful_requests"] or 0),
        "failed_requests": int(row["failed_requests"] or 0),
        "blocked_requests": int(row["blocked_requests"] or 0),
        "average_latency_seconds": round(
            float(row["average_latency_seconds"] or 0), 3
        ),
        "total_latency_seconds": round(
            float(row["total_latency_seconds"] or 0), 3
        ),
        "error_categories": {
            item["category"]: int(item["count"]) for item in categories
        },
        "storage": "sqlite",
    }


# =====================================================
# 5. REQUEST HISTORY FOR ADMIN DASHBOARD
# =====================================================

def get_request_history(limit: int = 100, offset: int = 0) -> list[dict]:
    """
    Return recent request records for an admin dashboard.
    Does not include question text or generated answers.
    """
    safe_limit = max(1, min(int(limit), 1000))
    safe_offset = max(0, int(offset))

    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT id, timestamp_utc, status, latency_seconds,
                   error_category, endpoint, request_id
            FROM request_metrics
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (safe_limit, safe_offset),
        ).fetchall()

    return [
        {
            "id": int(row["id"]),
            "timestamp_utc": row["timestamp_utc"],
            "status": row["status"],
            "latency_seconds": round(float(row["latency_seconds"]), 3),
            "error_category": row["error_category"],
            "endpoint": row["endpoint"],
            "request_id": row["request_id"],
        }
        for row in rows
    ]


# =====================================================
# 6. LATENCY TREND FOR CHARTS
# =====================================================

def get_latency_trend(limit: int = 100) -> list[dict]:
    """Return recent request latency points in chronological order."""
    safe_limit = max(1, min(int(limit), 1000))

    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT timestamp_utc, latency_seconds, status
            FROM (
                SELECT id, timestamp_utc, latency_seconds, status
                FROM request_metrics
                ORDER BY id DESC
                LIMIT ?
            )
            ORDER BY id ASC
            """,
            (safe_limit,),
        ).fetchall()

    return [
        {
            "timestamp_utc": row["timestamp_utc"],
            "latency_seconds": round(float(row["latency_seconds"]), 3),
            "status": row["status"],
        }
        for row in rows
    ]


# =====================================================
# 7. DAILY COUNTS
# =====================================================

def get_daily_metrics(days: int = 30) -> list[dict]:
    """Return daily counts and average latency for the last N days."""
    safe_days = max(1, min(int(days), 365))

    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT
                substr(timestamp_utc, 1, 10) AS date,
                COUNT(*) AS total_requests,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END)
                    AS successful_requests,
                SUM(CASE WHEN status = 'failure' THEN 1 ELSE 0 END)
                    AS failed_requests,
                SUM(CASE WHEN status = 'blocked' THEN 1 ELSE 0 END)
                    AS blocked_requests,
                AVG(latency_seconds) AS average_latency_seconds
            FROM request_metrics
            WHERE timestamp_utc >= date('now', ?)
            GROUP BY substr(timestamp_utc, 1, 10)
            ORDER BY date ASC
            """,
            (f"-{safe_days - 1} days",),
        ).fetchall()

    return [
        {
            "date": row["date"],
            "total_requests": int(row["total_requests"] or 0),
            "successful_requests": int(row["successful_requests"] or 0),
            "failed_requests": int(row["failed_requests"] or 0),
            "blocked_requests": int(row["blocked_requests"] or 0),
            "average_latency_seconds": round(
                float(row["average_latency_seconds"] or 0), 3
            ),
        }
        for row in rows
    ]