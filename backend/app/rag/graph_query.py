"""Graph answers, for the question side of /api/ask (docs/graph-rag-v2.md, section 6).

Only "what caused X" questions use the graph. The question names one entity. Its causal links
are followed backwards to their causes, the best chains are kept, and their spans are passed to
the answer model with the v1 passages.
"""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel

from app.db import connect
from app.llm import embed, embed_model_key, generate_json
from app.rag.graph_resolve import RESOLVE_MAX_CANDIDATES, RESOLVE_MIN_COSINE, SameAnswer
from app.rag.names import normalize

GRAPH_MAX_DEPTH = 3
GRAPH_MAX_BRANCH = 5
GRAPH_MAX_PATHS = 3

ANALYSIS_PROMPT = """Read this question about a set of documents.

Question: {question}

kind: "causes_of" when the question asks what caused, led to, or explains one named thing,
for example "What caused C?". Use "other" for everything else, including questions about
what a named thing caused.
entity: the one thing the question is about, as the question names it. Empty when none.
"""

QUESTION_SAME_PROMPT = """A question names a thing. Decide whether the name in the question refers to a known entity.

Question: {question}
Name in the question: "{q_name}"

Known entity: "{b_name}" (type: {b_type})
Passage for the known entity: {b_sample}

Answer "same" when the name in the question is this entity. A shorter name the question uses for the
same thing counts: "traffic jams" is the same as "heavy traffic jams" when the passage is about that
kind of traffic jam. Answer "different" if the name is a part, a cause, an effect, or a related but
separate thing, such as a map of the thing or a different thing that shares a word. Answer "unsure"
if the passage does not show it.
"""


class QuestionAnalysis(BaseModel):
    kind: Literal["causes_of", "other"]
    entity: str


@dataclass(frozen=True)
class Link:
    """All assertions between one src and dst with one type."""

    src: int
    dst: int
    type: str
    assertions: tuple[dict[str, Any], ...]

    @property
    def score(self) -> tuple[bool, int]:
        # Explicit beats inferred; then more supporting documents.
        return (any(a["explicit"] for a in self.assertions), len({a["doc_id"] for a in self.assertions}))


@dataclass(frozen=True)
class Path:
    """Causal links from the target backwards: links[0] points at the target."""

    links: tuple[Link, ...]

    @property
    def score(self) -> tuple[bool, int]:
        # A path is as strong as its weakest link.
        return min(link.score for link in self.links)

    @property
    def assertion_ids(self) -> list[list[int]]:
        return [[a["id"] for a in link.assertions] for link in self.links]


@dataclass
class GraphContext:
    question_kind: str
    targets: list[int]
    names: dict[int, str]
    paths: list[Path]
    conflicts: list[Link]
    spans: list[dict[str, Any]]
    assertion_spans: dict[int, list[int]]

    def rules(self) -> str:
        """Instructions for the answer prompt, built from the kept chains. Empty when there are none."""
        if not self.paths:
            return ""
        lines = ["Causal chains found in the graph (their passages are included below):"]
        for path in self.paths:
            lines.append(f"- {self._chain(path)}: " + ", ".join(
                f'"{self.names[link.src]} → {self.names[link.dst]}" (spans '
                + ", ".join(str(i) for a in link.assertions for i in self.assertion_spans[a["id"]]) + ")"
                for link in reversed(path.links)
            ))
        for link in self.conflicts:
            lines.append(
                f'- Disagreement: "{self.names[link.src]} → {self.names[link.dst]}" is also stated as not '
                f"caused (spans " + ", ".join(str(i) for a in link.assertions for i in self.assertion_spans[a["id"]]) + ")."
            )
        lines.append(
            "When the question asks for a cause, state the full chain, such as A → B → C. Cite the span "
            "for every link. If a link has no passage, say so; do not infer it. Report a disagreement "
            "and cite both sides."
        )
        return "\n".join(lines)

    def _chain(self, path: Path) -> str:
        nodes = [path.links[-1].src] + [link.dst for link in reversed(path.links)]
        return " → ".join(self.names[n] for n in nodes)

    def debug(self, target_names: dict[int, dict[str, Any]]) -> dict[str, Any]:
        assertions = {}
        for path in self.paths:
            for link in path.links:
                assertions.update({a["id"]: a for a in link.assertions})
        for link in self.conflicts:
            assertions.update({a["id"]: a for a in link.assertions})
        return {
            "target": target_names.get(self.targets[0]) if self.targets else None,
            "targets": self.targets,
            "question_kind": self.question_kind,
            "assertions": [
                {
                    "id": a["id"], "src": a["src"], "dst": a["dst"],
                    "src_name": self.names[a["src"]], "dst_name": self.names[a["dst"]],
                    "type": a["type"], "predicate": a["predicate"], "polarity": a["polarity"],
                    "explicit": a["explicit"], "doc_id": a["doc_id"],
                    "span_ids": self.assertion_spans[a["id"]],
                }
                for a in assertions.values()
            ],
            "paths": [
                {"links": path.assertion_ids, "score": list(path.score), "chain": self._chain(path)}
                for path in self.paths
            ],
            "conflicts": [
                {"src": link.src, "dst": link.dst, "type": link.type,
                 "assertion_ids": [a["id"] for a in link.assertions]}
                for link in self.conflicts
            ],
        }


def analyse(question: str) -> QuestionAnalysis:
    """One model call. Runs without a database connection."""
    return generate_json(ANALYSIS_PROMPT.format(question=question), QuestionAnalysis, think=False)


def resolve_targets(name: str, question: str) -> list[int]:
    """Entity ids the question names, best first. Exact name or alias first, then embeddings plus a confirm."""
    if not name.strip():
        return []
    with connect() as conn:
        ids = _exact_targets(conn, name)
    if ids:
        return ids
    vector = _unit(np.asarray(embed([name], "RETRIEVAL_DOCUMENT")[0], dtype=np.float32))
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, canonical_name, type, embedding FROM entities WHERE embedding IS NOT NULL AND embed_model = ?",
            (embed_model_key(),),
        ).fetchall()
    scored = []
    for r in rows:
        score = float(_unit(np.frombuffer(r["embedding"], dtype=np.float32)) @ vector)
        if score >= RESOLVE_MIN_COSINE:
            scored.append((score, r))
    scored.sort(key=lambda t: -t[0])
    for _, r in scored[:RESOLVE_MAX_CANDIDATES]:
        with connect() as conn:
            sample = conn.execute(
                "SELECT s.llm_text FROM entity_mentions m JOIN spans s ON s.id = m.span_id "
                "WHERE m.entity_id = ? ORDER BY m.id LIMIT 1",
                (r["id"],),
            ).fetchone()
        prompt = QUESTION_SAME_PROMPT.format(
            question=question, q_name=name, b_name=r["canonical_name"], b_type=r["type"],
            b_sample=sample["llm_text"] if sample else "(none)",
        )
        decision = generate_json(prompt, SameAnswer, think=False).decision
        if decision == "same":
            return [r["id"]]
        if decision == "unsure":
            break
    return []


def build(conn: Any, kind: str, targets: list[int]) -> GraphContext | None:
    """Traverse, rank, find conflicts and collect passages. None when the graph does not apply."""
    if kind != "causes_of" or not targets:
        return None
    causal = [r["name"] for r in conn.execute(
        "SELECT name FROM relation_types WHERE class = 'causal' AND status = 'active'"
    )]
    if not causal:
        return None
    candidates = []
    for target in targets:
        candidates.extend(_maximal_paths(conn, target, causal))
    paths = _rank(candidates)
    conflicts = _conflicts(conn, paths)
    if not paths and not conflicts:
        return GraphContext(kind, targets, {}, [], [], [], {})

    links = [link for path in paths for link in path.links] + conflicts
    all_assertions = [a for link in links for a in link.assertions]
    span_rows, assertion_spans = _span_rows(conn, [a["id"] for a in all_assertions])
    entity_ids = {link.src for link in links} | {link.dst for link in links}
    names = _names(conn, entity_ids)
    return GraphContext(kind, targets, names, paths, conflicts, span_rows, assertion_spans)


def debug_targets(conn: Any, ids: list[int]) -> dict[int, dict[str, Any]]:
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    return {
        r["id"]: {"id": r["id"], "canonical_name": r["canonical_name"], "type": r["type"]}
        for r in conn.execute(f"SELECT id, canonical_name, type FROM entities WHERE id IN ({marks})", ids)
    }


def _exact_targets(conn: Any, name: str) -> list[int]:
    key = normalize(name)
    if not key:
        return []
    mentions = {r["entity_id"]: r["n"] for r in conn.execute(
        "SELECT entity_id, COUNT(*) AS n FROM entity_mentions GROUP BY entity_id"
    )}
    by_alias = {r["entity_id"] for r in conn.execute("SELECT entity_id FROM entity_aliases WHERE alias = ?", (key,))}
    matches = [
        r["id"] for r in conn.execute("SELECT id, canonical_name FROM entities")
        if normalize(r["canonical_name"]) == key or r["id"] in by_alias
    ]
    return sorted(matches, key=lambda i: (-mentions.get(i, 0), i))


def _incoming(conn: Any, node: int, causal: list[str]) -> list[Link]:
    marks = ",".join("?" * len(causal))
    rows = conn.execute(
        "SELECT a.id, a.src_entity, a.dst_entity, a.type, a.predicate, a.polarity, a.explicit, a.doc_id, "
        "d.original_filename FROM relation_assertions a JOIN documents d ON d.id = a.doc_id "
        f"WHERE d.status = 'saved' AND a.dst_entity = ? AND a.polarity = 'positive' AND a.type IN ({marks})",
        [node, *causal],
    ).fetchall()
    grouped: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault((r["src_entity"], r["dst_entity"], r["type"]), []).append({
            "id": r["id"], "src": r["src_entity"], "dst": r["dst_entity"], "type": r["type"],
            "predicate": r["predicate"], "polarity": r["polarity"], "explicit": bool(r["explicit"]),
            "doc_id": r["doc_id"], "doc_name": r["original_filename"],
        })
    links = [Link(src, dst, type_, tuple(items)) for (src, dst, type_), items in grouped.items()]
    # Branch cut after ranking by link score, so the strongest causes survive it.
    links.sort(key=lambda link: (link.score, -link.src), reverse=True)
    return links[:GRAPH_MAX_BRANCH]


def _maximal_paths(conn: Any, target: int, causal: list[str]) -> list[Path]:
    """Paths that stop: no further cause, or the depth limit. Prefixes are not returned."""
    found: list[Path] = []

    def walk(node: int, chain: list[Link], visited: set[int]) -> None:
        if len(chain) == GRAPH_MAX_DEPTH:
            found.append(Path(tuple(chain)))
            return
        links = [link for link in _incoming(conn, node, causal) if link.src not in visited]
        if not links:
            if chain:
                found.append(Path(tuple(chain)))
            return
        for link in links:
            walk(link.src, chain + [link], visited | {link.src})

    walk(target, [], {target})
    return found


def _rank(paths: list[Path]) -> list[Path]:
    # Longer paths first on equal score. Equal paths keep the order of their lowest assertion id.
    by_id = sorted(paths, key=lambda p: min(a["id"] for link in p.links for a in link.assertions))
    return sorted(by_id, key=lambda p: (p.score, len(p.links)), reverse=True)[:GRAPH_MAX_PATHS]


def _conflicts(conn: Any, paths: list[Path]) -> list[Link]:
    found = []
    seen: set[tuple[int, int, str]] = set()
    for path in paths:
        for link in path.links:
            key = (link.src, link.dst, link.type)
            if key in seen:
                continue
            seen.add(key)
            rows = conn.execute(
                "SELECT a.id, a.src_entity, a.dst_entity, a.type, a.predicate, a.polarity, a.explicit, a.doc_id, "
                "d.original_filename FROM relation_assertions a JOIN documents d ON d.id = a.doc_id "
                "WHERE d.status = 'saved' AND a.src_entity = ? AND a.dst_entity = ? AND a.type = ? "
                "AND a.polarity = 'negative' ORDER BY a.id",
                key,
            ).fetchall()
            if rows:
                items = tuple({
                    "id": r["id"], "src": r["src_entity"], "dst": r["dst_entity"], "type": r["type"],
                    "predicate": r["predicate"], "polarity": r["polarity"], "explicit": bool(r["explicit"]),
                    "doc_id": r["doc_id"], "doc_name": r["original_filename"],
                } for r in rows)
                found.append(Link(link.src, link.dst, link.type, items))
    return found


def _span_rows(conn: Any, assertion_ids: list[int]) -> tuple[list[dict[str, Any]], dict[int, list[int]]]:
    """Passages in the order the assertions were kept, without duplicates, plus span ids per assertion."""
    if not assertion_ids:
        return [], {}
    marks = ",".join("?" * len(assertion_ids))
    rows = conn.execute(
        "SELECT asp.assertion_id, s.id AS span_id, s.chunk_id, s.doc_id, d.original_filename, s.page_number, "
        "c.page_title, s.heading_path, s.kind, s.start, s.end, s.text, s.llm_text "
        "FROM assertion_spans asp JOIN spans s ON s.id = asp.span_id JOIN chunks c ON c.id = s.chunk_id "
        f"JOIN documents d ON d.id = s.doc_id WHERE asp.assertion_id IN ({marks}) AND d.status = 'saved'",
        assertion_ids,
    ).fetchall()
    by_assertion: dict[int, list[int]] = {}
    spans: dict[int, dict[str, Any]] = {}
    for r in rows:
        by_assertion.setdefault(r["assertion_id"], []).append(r["span_id"])
        spans.setdefault(r["span_id"], {k: r[k] for k in r.keys() if k != "assertion_id"})
    ordered = []
    for aid in assertion_ids:
        for sid in sorted(by_assertion.get(aid, [])):
            if sid in spans:  # popped once added, so a span shared by two assertions appears once
                ordered.append(spans.pop(sid))
    return ordered, {aid: sorted(ids) for aid, ids in by_assertion.items()}


def _names(conn: Any, ids: set[int]) -> dict[int, str]:
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    return {r["id"]: r["canonical_name"] for r in conn.execute(
        f"SELECT id, canonical_name FROM entities WHERE id IN ({marks})", list(ids)
    )}


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector
