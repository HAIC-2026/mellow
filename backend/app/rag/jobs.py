"""The index_jobs table: one row per document, written by saves and by the worker."""

import threading
from datetime import datetime, timezone
from typing import Any

from app.db import connect

# Set when a job is queued, so the worker in this process starts at once instead of
# waiting for its next poll.
wake = threading.Event()


def enqueue(doc_ids: list[str]) -> None:
    """Queue documents for indexing. A newer generation makes any running job stale."""
    now = _now()
    with connect() as conn:
        conn.executemany(
            "INSERT INTO index_jobs (doc_id, generation, stage, status, attempts, error, updated_at) "
            "VALUES (?, 1, 'embed', 'pending', 0, NULL, ?) "
            "ON CONFLICT (doc_id) DO UPDATE SET generation = generation + 1, stage = 'embed', "
            "status = 'pending', attempts = 0, error = NULL, updated_at = excluded.updated_at",
            [(doc_id, now) for doc_id in doc_ids],
        )
    wake.set()


def enqueue_graph(doc_ids: list[str]) -> None:
    """Queue stage extract only, for documents whose chunks are already current."""
    now = _now()
    with connect() as conn:
        conn.executemany(
            "UPDATE index_jobs SET generation = generation + 1, stage = 'extract', status = 'pending', "
            "attempts = 0, error = NULL, updated_at = ? WHERE doc_id = ?",
            [(now, doc_id) for doc_id in doc_ids],
        )
    wake.set()


def is_current(conn: Any, doc_id: str, generation: int) -> bool:
    """A stage may write only while its generation is still the job's generation."""
    row = conn.execute("SELECT generation FROM index_jobs WHERE doc_id = ?", (doc_id,)).fetchone()
    return row is not None and row["generation"] == generation


def claim() -> dict[str, Any] | None:
    """Mark the oldest pending job running and return it, or None if there is none.

    The conditional UPDATE makes the claim atomic, so two processes never run one job.
    """
    with connect() as conn:
        while True:
            row = conn.execute(
                "SELECT doc_id, generation, stage FROM index_jobs WHERE status = 'pending' "
                "ORDER BY updated_at LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            cur = conn.execute(
                "UPDATE index_jobs SET status = 'running', attempts = attempts + 1, updated_at = ? "
                "WHERE doc_id = ? AND generation = ? AND status = 'pending'",
                (_now(), row["doc_id"], row["generation"]),
            )
            if cur.rowcount == 1:
                conn.commit()
                return dict(row)


def finish(doc_id: str, generation: int, status: str, error: str | None = None) -> None:
    # No-op when the document was saved again meanwhile: the newer job keeps its state.
    with connect() as conn:
        conn.execute(
            "UPDATE index_jobs SET status = ?, error = ?, updated_at = ? WHERE doc_id = ? AND generation = ?",
            (status, error, _now(), doc_id, generation),
        )


def reset_running() -> None:
    # Jobs left running by a crash or restart start over.
    with connect() as conn:
        conn.execute("UPDATE index_jobs SET status = 'pending' WHERE status = 'running'")


def status_of(doc_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT j.stage, j.status, j.error, j.updated_at, d.indexed_at, d.graph_indexed_at FROM index_jobs j "
            "JOIN documents d ON d.id = j.doc_id WHERE j.doc_id = ?",
            (doc_id,),
        ).fetchone()
    return dict(row) if row else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
