"""Review list: python -m app.rag.review

Lists what needs a person's decision (docs/graph-rag-v2.md, 1.3 and 1.4):
- wordings stored as type "other", grouped by predicate, with the proposed types recorded so far;
- two-way causal pairs (A -> B and B -> A), which usually mean a direction error.

Read only. Promote, map or reject proposals by hand, then run python -m app.rag.reindex --graph.
"""

import argparse
from collections import defaultdict
from typing import Any

from app.db import connect
from app.rag.names import normalize


def other_wordings(conn: Any, min_docs: int) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    rows = conn.execute(
        "SELECT a.predicate, a.doc_id, d.original_filename FROM relation_assertions a "
        "JOIN documents d ON d.id = a.doc_id WHERE d.status = 'saved' AND a.type = 'other'"
    )
    for r in rows:
        key = normalize(r["predicate"])
        g = groups.setdefault(key, {"wording": key, "assertions": 0, "docs": set(), "example": r["predicate"]})
        g["assertions"] += 1
        g["docs"].add(r["original_filename"])
    result = [
        {**g, "docs": len(g["docs"])} for g in groups.values() if len(g["docs"]) >= min_docs
    ]
    return sorted(result, key=lambda g: (-g["docs"], -g["assertions"], g["wording"]))


def proposed_types(conn: Any) -> list[tuple[str, str | None]]:
    return [(r["name"], r["inverse_of"]) for r in conn.execute(
        "SELECT name, inverse_of FROM relation_types WHERE status = 'proposed' ORDER BY name"
    )]


def two_way_pairs(conn: Any) -> list[dict[str, Any]]:
    causal = [r["name"] for r in conn.execute("SELECT name FROM relation_types WHERE class = 'causal' AND status = 'active'")]
    if not causal:
        return []
    marks = ",".join("?" * len(causal))
    by_link: dict[tuple[int, int, str], dict[str, Any]] = {}
    rows = conn.execute(
        "SELECT a.src_entity, a.dst_entity, a.type, a.predicate, a.doc_id, d.original_filename FROM relation_assertions a "
        f"JOIN documents d ON d.id = a.doc_id WHERE d.status = 'saved' AND a.polarity = 'positive' AND a.type IN ({marks})",
        causal,
    )
    for r in rows:
        link = by_link.setdefault((r["src_entity"], r["dst_entity"], r["type"]), {"predicates": set(), "docs": set()})
        link["predicates"].add(r["predicate"])
        link["docs"].add(r["original_filename"])
    names = {r["id"]: r["canonical_name"] for r in conn.execute("SELECT id, canonical_name FROM entities")}
    pairs = []
    for (src, dst, type_), link in by_link.items():
        back = by_link.get((dst, src, type_))
        if back is None or src > dst:
            continue  # report each pair once
        pairs.append({
            "a": names.get(src, str(src)), "b": names.get(dst, str(dst)), "type": type_,
            "forward": sorted(link["predicates"]), "backward": sorted(back["predicates"]),
            "docs": sorted(link["docs"] | back["docs"]),
        })
    return sorted(pairs, key=lambda p: (p["a"], p["b"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min-docs", type=int, default=1, help="only wordings seen in at least this many documents")
    args = parser.parse_args(argv)
    with connect() as conn:
        proposals = proposed_types(conn)
        wordings = other_wordings(conn, args.min_docs)
        pairs = two_way_pairs(conn)

    print(f"Proposed relation types ({len(proposals)}), recorded by extraction, not yet decided:")
    for name, inverse in proposals:
        print(f"  {name}" + (f"  (inverse of {inverse})" if inverse else ""))
    print(f"\nWordings stored as 'other' ({len(wordings)}), by documents then count:")
    for g in wordings:
        print(f"  {g['docs']} doc(s), {g['assertions']} assertion(s): {g['wording']!r}  e.g. {g['example']!r}")
    print(f"\nTwo-way causal pairs ({len(pairs)}), check the direction:")
    for p in pairs:
        print(f"  {p['a']} <-> {p['b']} [{p['type']}]: {p['forward']} / {p['backward']}  docs: {', '.join(p['docs'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
