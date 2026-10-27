import logging
import re
from typing import Any

import numpy as np

from app.db import connect
from app.llm import answer_with_citations, embed

log = logging.getLogger(__name__)

CANDIDATES = 20
RRF_K = 60
NOT_FOUND = "The saved documents do not cover this question."


def ask(question: str, top_k: int = 8) -> dict[str, Any]:
    with connect() as conn:
        ranked = _fuse([_keyword_search(conn, question), _vector_search(conn, question)])[:top_k]
        spans = _spans_for(conn, ranked)

    if not spans:
        return {"found": False, "answer_segments": [{"text": NOT_FOUND, "citations": []}], "citations": {}}

    passages = "\n".join(
        f"[{s['span_id']}] ({s['original_filename']} › {s['heading_path'] or s['page_title']}) {s['llm_text']}"
        for s in spans
    )
    answer = answer_with_citations(question, passages)

    # Only ids we sent count as citations; the LLM never supplies document names or quotes.
    by_id = {s["span_id"]: s for s in spans}
    segments = []
    cited: dict[int, dict[str, Any]] = {}
    for seg in answer.answer_segments:
        ids = [i for i in dict.fromkeys(seg.span_ids) if i in by_id]
        segments.append({"text": seg.text, "citations": ids})
        for i in ids:
            cited[i] = {k: v for k, v in by_id[i].items() if k != "llm_text"}

    found = answer.found and bool(cited)
    if not found:
        segments, cited = [{"text": NOT_FOUND, "citations": []}], {}
    return {"found": found, "answer_segments": segments, "citations": cited}


def _keyword_search(conn: Any, question: str) -> list[int]:
    tokens = [t for t in re.findall(r"\w+", question.lower()) if len(t) > 1]
    if not tokens:
        return []
    match = " OR ".join(f'"{t}"' for t in dict.fromkeys(tokens))
    rows = conn.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
        (match, CANDIDATES),
    ).fetchall()
    return [r["rowid"] for r in rows]


def _vector_search(conn: Any, question: str) -> list[int]:
    rows = conn.execute("SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL").fetchall()
    if not rows:
        return []
    try:
        query = np.asarray(embed([question], "RETRIEVAL_QUERY")[0], dtype=np.float32)
    except Exception:
        log.exception("query embedding failed; using keyword search only")
        return []
    # Skip vectors from a different embedding model until `python -m app.rag.reindex` is run.
    rows = [r for r in rows if len(r["embedding"]) == query.nbytes]
    if not rows:
        return []
    matrix = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    scores = matrix @ query / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(query) + 1e-9)
    order = np.argsort(-scores)[:CANDIDATES]
    return [rows[i]["id"] for i in order]


def _fuse(rankings: list[list[int]]) -> list[int]:
    # Reciprocal rank fusion: robust to BM25 and cosine scores being on different scales.
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
    return sorted(scores, key=lambda c: -scores[c])


def _spans_for(conn: Any, chunk_ids: list[int]) -> list[dict[str, Any]]:
    if not chunk_ids:
        return []
    marks = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        "SELECT s.id AS span_id, s.chunk_id, s.doc_id, d.original_filename, s.page_number, "
        "c.page_title, s.heading_path, s.kind, s.start, s.end, s.text, s.llm_text "
        "FROM spans s JOIN chunks c ON c.id = s.chunk_id JOIN documents d ON d.id = s.doc_id "
        f"WHERE s.chunk_id IN ({marks}) AND d.status = 'saved' ORDER BY s.start",
        chunk_ids,
    ).fetchall()
    rank = {c: i for i, c in enumerate(chunk_ids)}
    spans = [dict(r) for r in rows]
    spans.sort(key=lambda s: (rank[s["chunk_id"]], s["start"]))
    return spans
