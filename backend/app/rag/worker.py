"""Background indexing: one thread per process takes jobs from index_jobs."""

import logging
import threading

from app.db import connect
from app.rag import jobs
from app.rag.indexer import index_document
from app.routers.documents import _read_pages

log = logging.getLogger(__name__)

# Fallback poll, for jobs queued by another process (e.g. python -m app.rag.reindex).
POLL_SECONDS = 5


def run_job(job: dict) -> None:
    doc_id, generation = job["doc_id"], job["generation"]
    try:
        with connect() as conn:
            row = conn.execute(
                "SELECT final_md_path FROM documents WHERE id = ? AND status = 'saved'", (doc_id,)
            ).fetchone()
        if row is None:
            raise RuntimeError("document is not saved")
        if index_document(doc_id, _read_pages(row["final_md_path"]), generation):
            jobs.finish(doc_id, generation, "done")
    except Exception as exc:
        log.exception("indexing failed for %s (stage %s)", doc_id, job["stage"])
        jobs.finish(doc_id, generation, "failed", str(exc))


def run_until_idle() -> None:
    while (job := jobs.claim()) is not None:
        run_job(job)


class Worker:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="index-worker", daemon=True)

    def start(self) -> None:
        jobs.reset_running()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        jobs.wake.set()
        # A job in progress finishes on its own; the daemon thread does not block exit.
        self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.is_set():
            jobs.wake.clear()
            try:
                run_until_idle()
            except Exception:
                log.exception("index worker error")
            jobs.wake.wait(POLL_SECONDS)
