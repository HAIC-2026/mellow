import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np

from app.db import connect
from app.llm import embed, embed_model_key
from app.rag import jobs
from app.rag.graph_extract import delete_graph
from app.rag.spans import Chunk, group_chunks, split_spans

log = logging.getLogger(__name__)


def index_document(doc_id: str, pages: list[dict[str, Any]], generation: int) -> bool:
    """Rebuild the retrieval index for one document.

    Raises if embedding fails; the old index is then left as it was. Returns False without
    writing when the document was saved again while this ran (its job generation changed).
    """
    entries: list[tuple[dict[str, Any], Chunk]] = []
    for page in pages:
        for chunk in group_chunks(split_spans(page["markdown"])):
            entries.append((page, chunk))

    # Embed before touching the DB, so the old index stays searchable until the swap.
    model_key = embed_model_key()
    raw = embed([chunk.text for _, chunk in entries], "RETRIEVAL_DOCUMENT") if entries else []
    vectors = [np.asarray(v, dtype=np.float32).tobytes() for v in raw]

    with connect() as conn:
        # Take the write lock before the generation check, so no save can commit between
        # the check and the swap.
        conn.execute("BEGIN IMMEDIATE")
        if not jobs.is_current(conn, doc_id, generation):
            log.info("discarding stale index run for %s (generation %s)", doc_id, generation)
            return False
        _delete(conn, doc_id)
        for (page, chunk), vector in zip(entries, vectors):
            cur = conn.execute(
                "INSERT INTO chunks (doc_id, page_number, page_title, heading_path, text, embedding, embed_model) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (doc_id, page["number"], page["title"], chunk.heading_path, chunk.text, vector, model_key),
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
        # The graph is built from these spans, so it is stale until stage extract finishes.
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE documents SET indexed_at = ?, graph_indexed_at = NULL WHERE id = ?", (now, doc_id)
        )
        conn.execute(
            "UPDATE index_jobs SET stage = 'extract', status = 'pending', attempts = 0, error = NULL, "
            "updated_at = ? WHERE doc_id = ?",
            (now, doc_id),
        )
    return True


def _delete(conn: Any, doc_id: str) -> None:
    conn.execute(
        "DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE doc_id = ?)", (doc_id,)
    )
    conn.execute("DELETE FROM spans WHERE doc_id = ?", (doc_id,))
    conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
    delete_graph(conn, doc_id)
