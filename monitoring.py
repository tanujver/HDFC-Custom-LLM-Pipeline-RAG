import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Optional


PROJECT_DIR = Path(__file__).resolve().parent
DB_PATH = PROJECT_DIR / "monitoring.db"

_write_lock = Lock()

VALID_STATUSES = {"success", "failure", "blocked"}
VALID_FEEDBACK = {"helpful", "not_helpful"}


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA foreign_keys = ON")

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with _write_lock:
        with _connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS request_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc TEXT NOT NULL,
                    status TEXT NOT NULL
                        CHECK(status IN ('success', 'failure', 'blocked')),
                    latency_seconds REAL NOT NULL,
                    error_category TEXT,
                    endpoint TEXT,
                    request_id TEXT,
                    predicted_intent TEXT,
                    feedback TEXT
                        CHECK(
                            feedback IS NULL OR
                            feedback IN ('helpful', 'not_helpful')
                        )
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_request_timestamp
                ON request_metrics(timestamp_utc)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_request_status
                ON request_metrics(status)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_request_intent
                ON request_metrics(predicted_intent)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_request_id
                ON request_metrics(request_id)
            """)


def record_request(
    status: str,
    latency_seconds: float,
    error_category: Optional[str] = None,
    endpoint: str = "/ask",
    request_id: Optional[str] = None,
    predicted_intent: Optional[str] = None
):
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status: {status}")

    with _write_lock:
        with _connect() as conn:
            conn.execute("""
                INSERT INTO request_metrics (
                    timestamp_utc,
                    status,
                    latency_seconds,
                    error_category,
                    endpoint,
                    request_id,
                    predicted_intent
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                _utc_now(),
                status,
                float(latency_seconds),
                error_category,
                endpoint,
                request_id,
                predicted_intent
            ))


def record_feedback(request_id: str, feedback: str) -> bool:
    if feedback not in VALID_FEEDBACK:
        raise ValueError("Feedback must be helpful or not_helpful")

    if not request_id:
        return False

    with _write_lock:
        with _connect() as conn:
            cursor = conn.execute("""
                UPDATE request_metrics
                SET feedback = ?
                WHERE request_id = ?
            """, (feedback, request_id))

            return cursor.rowcount > 0


def get_metrics():
    with _connect() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*) AS total_requests,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END)
                    AS successful_requests,
                SUM(CASE WHEN status = 'failure' THEN 1 ELSE 0 END)
                    AS failed_requests,
                SUM(CASE WHEN status = 'blocked' THEN 1 ELSE 0 END)
                    AS blocked_requests,
                AVG(latency_seconds) AS avg_latency_seconds,
                SUM(latency_seconds) AS total_latency_seconds
            FROM request_metrics
        """).fetchone()

        errors = conn.execute("""
            SELECT error_category, COUNT(*) AS count
            FROM request_metrics
            WHERE error_category IS NOT NULL
            GROUP BY error_category
        """).fetchall()

        return {
            "total_requests": row["total_requests"] or 0,
            "successful_requests": row["successful_requests"] or 0,
            "failed_requests": row["failed_requests"] or 0,
            "blocked_requests": row["blocked_requests"] or 0,
            "avg_latency_seconds": row["avg_latency_seconds"] or 0,
            "total_latency_seconds": row["total_latency_seconds"] or 0,
            "error_categories": {
                item["error_category"]: item["count"]
                for item in errors
            },
            "storage": "sqlite"
        }


def get_request_history(limit=100, offset=0):
    with _connect() as conn:
        rows = conn.execute("""
            SELECT
                timestamp_utc,
                status,
                latency_seconds,
                error_category,
                endpoint,
                request_id,
                predicted_intent,
                feedback
            FROM request_metrics
            ORDER BY id DESC
            LIMIT ? OFFSET ?
        """, (limit, offset)).fetchall()

        return [dict(row) for row in rows]


def get_latency_trend(limit=100):
    with _connect() as conn:
        rows = conn.execute("""
            SELECT timestamp_utc, latency_seconds, status
            FROM (
                SELECT timestamp_utc, latency_seconds, status, id
                FROM request_metrics
                ORDER BY id DESC
                LIMIT ?
            )
            ORDER BY id ASC
        """, (limit,)).fetchall()

        return [dict(row) for row in rows]


def get_daily_metrics(days=30):
    with _connect() as conn:
        rows = conn.execute("""
            SELECT
                DATE(timestamp_utc) AS date,
                COUNT(*) AS total_requests,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END)
                    AS successful_requests,
                SUM(CASE WHEN status = 'failure' THEN 1 ELSE 0 END)
                    AS failed_requests,
                SUM(CASE WHEN status = 'blocked' THEN 1 ELSE 0 END)
                    AS blocked_requests,
                AVG(latency_seconds) AS avg_latency_seconds
            FROM request_metrics
            WHERE timestamp_utc >= DATETIME('now', ?)
            GROUP BY DATE(timestamp_utc)
            ORDER BY date ASC
        """, (f"-{days} days",)).fetchall()

        return [dict(row) for row in rows]


def get_intent_metrics():
    with _connect() as conn:
        rows = conn.execute("""
            SELECT
                predicted_intent AS intent,
                COUNT(*) AS count
            FROM request_metrics
            WHERE predicted_intent IS NOT NULL
            GROUP BY predicted_intent
            ORDER BY count DESC
        """).fetchall()

        return [dict(row) for row in rows]


def get_feedback_metrics():
    with _connect() as conn:
        row = conn.execute("""
            SELECT
                SUM(CASE WHEN feedback = 'helpful' THEN 1 ELSE 0 END)
                    AS helpful,
                SUM(CASE WHEN feedback = 'not_helpful' THEN 1 ELSE 0 END)
                    AS not_helpful,
                COUNT(feedback) AS total_feedback
            FROM request_metrics
        """).fetchone()

        helpful = row["helpful"] or 0
        not_helpful = row["not_helpful"] or 0
        total = row["total_feedback"] or 0

        return {
            "helpful": helpful,
            "not_helpful": not_helpful,
            "total_feedback": total,
            "positive_percentage": (
                round((helpful / total) * 100, 2)
                if total else 0
            )
        }


init_db()