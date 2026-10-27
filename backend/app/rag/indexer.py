import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np

from app.db import connect
from app.llm import embed
from app.rag.spans import Chunk, group_chunks, split_spans

log = logging.getLogger(__name__)


def index_document(doc_id: str, pages: list[dict[str, Any]]) -> str | None:
    """Rebuild the retrieval index for one document. Returns a warning if embedding failed."""
    entries: list[tuple[dict[str, Any], Chunk]] = []
    for page in pages:
        for chunk in group_chunks(split_spans(page["markdown"])):
            entries.append((page, chunk))

    # Embed before touching the DB. If it fails, keyword search still works off FTS.
    warning = None
    vectors: list[bytes | None] = [None] * len(entries)
    if entries:
        try:
            raw = embed([chunk.text for _, chunk in entries], "RETRIEVAL_DOCUMENT")
            vectors = [np.asarray(v, dtype=np.float32).tobytes() for v in raw]
        except Exception as exc:
            log.exception("embedding failed for %s", doc_id)
            warning = f"indexed for keyword search only; embedding failed: {exc}"

    with connect() as conn:
        _delete(conn, doc_id)
        for (page, chunk), vector in zip(entries, vectors):
            cur = conn.execute(
                "INSERT INTO chunks (doc_id, page_number, page_title, heading_path, text, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (doc_id, page["number"], page["title"], chunk.heading_path, chunk.text, vector),
            )
            chunk_id = cur.lastrowid
            conn.execute("INSERT INTO chunks_fts (rowid, text) VALUES (?, ?)", (chunk_id, chunk.text))
            conn.executemany(
                "INSERT INTO spans (chunk_id, doc_id, page_number, kind, start, end, text, "
                "llm_text, heading_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (chunk_id, doc_id, page["number"], s.kind, s.start, s.end, s.text, s.llm_text, s.heading_path)
                    for s in chunk.spans
                ],
            )
        indexed_at = None if warning else datetime.now(timezone.utc).isoformat()
        conn.execute("UPDATE documents SET indexed_at = ? WHERE id = ?", (indexed_at, doc_id))
    return warning


def _delete(conn: Any, doc_id: str) -> None:
    conn.execute(
        "DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE doc_id = ?)", (doc_id,)
    )
    conn.execute("DELETE FROM spans WHERE doc_id = ?", (doc_id,))
    conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
