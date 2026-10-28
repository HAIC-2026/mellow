import re
import unittest
import zlib
from unittest import mock

import numpy as np

from app import db
from app.rag import graph_extract, jobs, worker
from app.rag.graph_extract import ExtractedEntity, ExtractedGraph, ExtractedRelation
from app.rag.graph_resolve import SameAnswer
from app.routers import documents
from tests.test_indexing import TempDbTest, fake_embed

DIMS = 16
# Known names get fixed vectors. Other names get a stable random direction, so they are unrelated.
KNOWN = {
    "flood": [1, 0, 0],
    "flood maps": [0.8, 0.6, 0],  # cosine 0.80 with "flood": a candidate, and not the same thing
    "Flooding": [0.95, 0.3, 0],  # cosine 0.95 with "flood": a variant
    "Department of Taxation": [0, 1, 0],
    "Department of Taxation (DOTAX)": [0, 0.99, 0.1],  # cosine 0.99: the same thing
}


def vector(text: str) -> list[float]:
    if text in KNOWN:
        return KNOWN[text] + [0] * (DIMS - len(KNOWN[text]))
    rng = np.random.default_rng(zlib.crc32(text.encode()))
    v = rng.normal(size=DIMS)
    return (v / np.linalg.norm(v)).tolist()


class ResolveCase(TempDbTest):
    def setUp(self) -> None:
        super().setUp()
        self.decisions: dict[frozenset[str], str] = {}
        self.asked: list[frozenset[str]] = []
        for target, kwargs in [
            ("app.rag.graph_resolve.embed", {"side_effect": lambda texts, task: [vector(t) for t in texts]}),
            ("app.rag.graph_resolve.generate_json", {"side_effect": self.decide}),
        ]:
            patcher = mock.patch(target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)

    def decide(self, prompt: str, model_cls: type, think: bool | None = None) -> SameAnswer:
        a = re.search(r'Name A: "(.*)" \(type', prompt).group(1)
        b = re.search(r'Name B: "(.*)" \(type', prompt).group(1)
        pair = frozenset((a, b))
        self.asked.append(pair)
        return SameAnswer(decision=self.decisions.get(pair, "different"))

    def add_doc(self, doc_id: str, sentences: list[str], entities: list[tuple[str, str, str]],
                relations: list[tuple[str, str, str]] = ()) -> None:
        """Save and extract a document. Each entity is (name, type, fragment of the sentence that states it)."""
        self.save_doc(doc_id)
        markdown = "# Notes\n\n" + "\n\n".join(sentences) + "\n"
        documents._write_pages(f"markdown/{doc_id}/final.json", [{"number": 1, "title": "Sheet 1", "markdown": markdown}])
        with mock.patch("app.rag.indexer.embed", side_effect=fake_embed):
            jobs.enqueue([doc_id])
            worker.run_until_idle()  # embed stage; extraction is empty here
        self.extract(doc_id, entities, relations)

    def extract(self, doc_id: str, entities: list[tuple[str, str, str]],
                relations: list[tuple[str, str, str]] = ()) -> None:
        """Run stage extract on the document's current chunks."""
        graph = ExtractedGraph(
            entities=[ExtractedEntity(name=n, type=t, span_ids=[self.span_for(doc_id, f)]) for n, t, f in entities],
            relations=[
                ExtractedRelation(
                    src=s, dst=d, type="causes", predicate="caused", polarity="positive", explicit=True,
                    span_ids=[self.span_for(doc_id, f)],
                )
                for s, d, f in relations
            ],
        )
        with mock.patch("app.rag.graph_extract.generate_json", return_value=graph):
            jobs.enqueue_graph([doc_id])
            worker.run_until_idle()
        self.assertEqual(jobs.status_of(doc_id)["status"], "done")

    def span_for(self, doc_id: str, fragment: str) -> int:
        with db.connect() as conn:
            row = conn.execute(
                "SELECT id FROM spans WHERE doc_id = ? AND llm_text LIKE ? ORDER BY id", (doc_id, f"%{fragment}%")
            ).fetchone()
        return row["id"]

    def entities(self) -> list[tuple[str, str]]:
        with db.connect() as conn:
            return [tuple(r) for r in conn.execute("SELECT canonical_name, type FROM entities ORDER BY id")]

    def aliases(self, name: str) -> set[str]:
        with db.connect() as conn:
            rows = conn.execute(
                "SELECT a.alias FROM entity_aliases a JOIN entities e ON e.id = a.entity_id WHERE e.canonical_name = ?",
                (name,),
            )
            return {r["alias"] for r in rows}


class ResolveTest(ResolveCase):
    def test_same_name_and_type_merges_without_model(self) -> None:
        self.add_doc("a", ["The flood closed the road."], [("flood", "Event", "flood closed")])
        self.add_doc("b", ["A flood was reported again."], [("flood", "event", "flood was reported")])
        self.assertEqual(self.entities(), [("flood", "Event")])
        self.assertEqual(self.asked, [])

    def test_related_names_stay_separate_when_model_says_different(self) -> None:
        # The phase check: "flood" and "flood maps" are close in embedding space but not the same.
        self.add_doc("a", ["The flood closed the road."], [("flood", "Event", "flood closed")])
        self.add_doc("b", ["Flood maps were updated."], [("flood maps", "Map", "maps were updated")])
        self.assertEqual(self.asked, [frozenset(("flood maps", "flood"))])
        self.assertEqual(sorted(self.entities()), [("flood", "Event"), ("flood maps", "Map")])

    def test_variant_merges_when_model_says_same(self) -> None:
        self.decisions[frozenset(("Department of Taxation (DOTAX)", "Department of Taxation"))] = "same"
        self.add_doc("a", ["The Department of Taxation issued it."], [("Department of Taxation", "Organization", "issued it")])
        self.add_doc("b", ["The Department of Taxation (DOTAX) reports."], [("Department of Taxation (DOTAX)", "organization", "reports")])
        self.assertEqual(self.entities(), [("Department of Taxation", "Organization")])

    def test_unsure_creates_a_new_entity(self) -> None:
        self.decisions[frozenset(("flood maps", "flood"))] = "unsure"
        self.add_doc("a", ["The flood closed the road."], [("flood", "Event", "flood closed")])
        self.add_doc("b", ["Flood maps were updated."], [("flood maps", "Map", "maps were updated")])
        self.assertEqual(len(self.entities()), 2)

    def test_names_in_one_document_can_merge(self) -> None:
        self.decisions[frozenset(("Flooding", "flood"))] = "same"
        self.add_doc(
            "a",
            ["The flood closed the road.", "Flooding closed the bridge."],
            [("flood", "Event", "flood closed"), ("Flooding", "Event", "bridge")],
        )
        self.assertEqual(self.entities(), [("flood", "Event")])

    def test_same_name_different_type_asks_model(self) -> None:
        self.decisions[frozenset(("flood", "flood"))] = "same"
        self.add_doc("a", ["The flood closed the road."], [("flood", "Event", "flood closed")])
        self.add_doc("b", ["The flood map is new."], [("flood", "Map", "map is new")])
        self.assertEqual(self.asked, [frozenset(("flood",))])
        self.assertEqual(self.entities(), [("flood", "Event")])

    def test_resave_keeps_names_and_merges(self) -> None:
        self.decisions[frozenset(("Flooding", "flood"))] = "same"
        self.add_doc("a", ["The flood closed the road."], [("flood", "Event", "flood closed")])
        self.add_doc("b", ["Flooding closed the bridge."], [("Flooding", "Event", "bridge")])
        before = self.entities()
        self.extract("b", [("Flooding", "Event", "bridge")])
        self.assertEqual(sorted(self.entities()), sorted(before))
        self.assertEqual(len(before), 1)

    def test_entity_changed_after_planning_is_not_reused(self) -> None:
        # Another worker may change an entity between planning and the write. The write must
        # then create a new entity rather than attach mentions to the changed one.
        self.decisions[frozenset(("Flooding", "flood"))] = "same"
        self.add_doc("a", ["The flood closed the road."], [("flood", "Event", "flood closed")])
        self.add_doc("b", ["Flooding closed the bridge."], [])
        real_resolve = graph_extract.resolve_names

        def resolve_then_rename(*args, **kwargs):
            resolution = real_resolve(*args, **kwargs)
            with db.connect() as conn:
                conn.execute("UPDATE entities SET canonical_name = 'renamed elsewhere'")
            return resolution

        with mock.patch("app.rag.graph_extract.resolve_names", side_effect=resolve_then_rename):
            self.extract("b", [("Flooding", "Event", "bridge")])
        self.assertEqual(sorted(self.entities()), [("Flooding", "Event"), ("renamed elsewhere", "Event")])
        with db.connect() as conn:
            mentions = {r["surface_form"] for r in conn.execute("SELECT surface_form FROM entity_mentions WHERE doc_id = 'b'")}
        self.assertEqual(mentions, {"Flooding"})

    def test_deleting_a_document_removes_its_aliases(self) -> None:
        self.decisions[frozenset(("Flooding", "flood"))] = "same"
        self.add_doc("a", ["The flood closed the road."], [("flood", "Event", "flood closed")])
        self.add_doc("b", ["Flooding closed the bridge."], [("Flooding", "Event", "bridge")])
        self.assertEqual(self.aliases("flood"), {"flood", "flooding"})
        self.extract("b", [])  # b is re-extracted without its mentions
        self.assertEqual(self.aliases("flood"), {"flood"})


if __name__ == "__main__":
    unittest.main()
