import re
from typing import Any

import numpy as np

from app.db import connect
from app.llm import answer_with_citations, embed, embed_model_key
from app.rag import graph_query

CANDIDATES = 20
RRF_K = 60
NOT_FOUND = "The saved documents do not cover this question."
# Question words that only add noise matches to the keyword search.
STOPWORDS = frozenset(
    "a an and are as at be by can could do does for from had has have how in is its of on or "
    "should that the their there these this those to was we were what when where which who whom "
    "why will with would you your".split()
)


def ask(question: str, top_k: int = 8, debug: bool = False) -> dict[str, Any]:
    analysis = graph_query.analyse(question)
    targets = graph_query.resolve_targets(analysis.entity, question) if analysis.kind == "causes_of" else []

    with connect() as conn:
        keyword = _keyword_search(conn, question)
        vector = _vector_search(conn, question)
        ranked = _fuse([keyword, vector])[:top_k]
        spans = _spans_for(conn, ranked)
        debug_chunks = _debug_chunks(conn, ranked, keyword, vector) if debug else None
        graph = graph_query.build(conn, analysis.kind, targets)
        graph_debug = None
        if debug:
            target_info = graph_query.debug_targets(conn, targets)
            graph_debug = graph.debug(target_info) if graph else {
                "target": None, "targets": [], "question_kind": analysis.kind,
                "assertions": [], "paths": [], "conflicts": [],
            }

    # Graph passages first: they are the chains the answer has to state.
    if graph is not None:
        seen = {s["span_id"] for s in graph.spans}
        spans = graph.spans + [s for s in spans if s["span_id"] not in seen]

    def with_debug(result: dict[str, Any], passages: str | None, raw: str | None) -> dict[str, Any]:
        if debug:
            result["debug"] = {"chunks": debug_chunks, "passages": passages, "raw_response": raw, "graph": graph_debug}
        return result

    if not spans:
        not_found = {"found": False, "answer_segments": [{"text": NOT_FOUND, "citations": []}], "citations": {}}
        return with_debug(not_found, None, None)

    passages = "\n".join(
        f"[{s['span_id']}] ({s['original_filename']} › {s['heading_path'] or s['page_title']}) {s['llm_text']}"
        for s in spans
    )
    answer, raw = answer_with_citations(question, passages, graph.rules() if graph else "")

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
    return with_debug({"found": found, "answer_segments": segments, "citations": cited}, passages, raw)


def _debug_chunks(conn: Any, ranked: list[int], keyword: list[int], vector: list[int]) -> list[dict[str, Any]]:
    # The chunks whose spans were sent to the LLM, in fused order, with their rank in each search.
    if not ranked:
        return []
    marks = ",".join("?" * len(ranked))
    rows = conn.execute(
        "SELECT c.id AS chunk_id, c.doc_id, d.original_filename, c.page_number, c.page_title, "
        "c.heading_path, c.text FROM chunks c JOIN documents d ON d.id = c.doc_id "
        f"WHERE c.id IN ({marks}) AND d.status = 'saved'",
        ranked,
    ).fetchall()
    by_id = {r["chunk_id"]: dict(r) for r in rows}
    chunks = []
    for fused_rank, chunk_id in enumerate(ranked, start=1):
        if chunk_id not in by_id:
            continue
        chunk = by_id[chunk_id]
        chunk["fused_rank"] = fused_rank
        chunk["keyword_rank"] = keyword.index(chunk_id) + 1 if chunk_id in keyword else None
        chunk["vector_rank"] = vector.index(chunk_id) + 1 if chunk_id in vector else None
        chunks.append(chunk)
    return chunks


def _keyword_search(conn: Any, question: str) -> list[int]:
    tokens = [t for t in re.findall(r"\w+", question.lower()) if len(t) > 1]
    # Fall back to all tokens when the question is only stopwords.
    tokens = [t for t in tokens if t not in STOPWORDS] or tokens
    if not tokens:
        return []
    match = " OR ".join(f'"{t}"' for t in dict.fromkeys(tokens))
    rows = conn.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
        (match, CANDIDATES),
    ).fetchall()
    return [r["rowid"] for r in rows]


def _vector_search(conn: Any, question: str) -> list[int]:
    # Only vectors from the current model and size are comparable. Others are ignored
    # until `python -m app.rag.reindex` is run.
    rows = conn.execute(
        "SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL AND embed_model = ?", (embed_model_key(),)
    ).fetchall()
    if not rows:
        return []
    # No keyword-only fallback: if Ollama is down after retries, the question fails.
    query = np.asarray(embed([question], "RETRIEVAL_QUERY")[0], dtype=np.float32)
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
