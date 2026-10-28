import unittest

from fastapi import HTTPException

from app.routers import graph
from tests.test_graph_query import GraphFixture


class GraphApiTest(GraphFixture):
    def setUp(self) -> None:
        super().setUp()
        self.doc("d1")
        self.doc("d2")
        self.link("A", "B", "d1")
        self.link("B", "C", "d1")
        self.link("C", "D", "d2")
        self.link("B", "C", "d2", polarity="negative")

    def test_depth_limits_the_neighbourhood(self) -> None:
        body = graph.graph_around(entity=self.entities["B"], depth=1, max_nodes=150)
        names = {n["name"] for n in body["nodes"]}
        self.assertEqual(names, {"A", "B", "C"})
        self.assertFalse(body["truncated"])

    def test_both_directions_are_followed(self) -> None:
        body = graph.graph_around(entity=self.entities["B"], depth=2, max_nodes=150)
        self.assertEqual({n["name"] for n in body["nodes"]}, {"A", "B", "C", "D"})

    def test_max_nodes_truncates(self) -> None:
        body = graph.graph_around(entity=self.entities["B"], depth=3, max_nodes=2)
        self.assertEqual(len(body["nodes"]), 2)
        self.assertTrue(body["truncated"])

    def test_unsaved_documents_are_excluded(self) -> None:
        self.doc("draft", status="parsed")
        self.link("X", "B", "draft")
        body = graph.graph_around(entity=self.entities["B"], depth=1, max_nodes=150)
        self.assertNotIn("X", {n["name"] for n in body["nodes"]})

    def test_conflict_and_review_markers(self) -> None:
        self.link("D", "C", "d1")  # C -> D and D -> C: a two-way causal pair
        body = graph.graph_around(entity=self.entities["C"], depth=1, max_nodes=150)
        edges = {(e["src"], e["dst"]): e for e in body["edges"]}
        c, d = self.entities["C"], self.entities["D"]
        self.assertTrue(edges[(c, d)]["review"])
        self.assertTrue(edges[(d, c)]["review"])
        b = self.entities["B"]
        self.assertTrue(edges[(b, c)]["conflict"])
        self.assertFalse(edges[(c, d)]["conflict"])

    def test_assertions_carry_span_detail_for_the_source_viewer(self) -> None:
        body = graph.graph_around(entity=self.entities["A"], depth=1, max_nodes=150)
        assertion = body["edges"][0]["assertions"][0]
        span = assertion["spans"][0]
        self.assertEqual(span["span_id"], assertion["span_ids"][0])
        for key in ("doc_id", "original_filename", "page_number", "page_title", "heading_path", "kind", "start", "end", "text"):
            self.assertIn(key, span)

    def test_unknown_entity_is_404(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            graph.graph_around(entity=999999)
        self.assertEqual(ctx.exception.status_code, 404)

    def test_search_matches_name_and_alias(self) -> None:
        results = graph.find_entities(q="  b ")
        self.assertEqual([r["name"] for r in results], ["B"])
        self.assertEqual(graph.find_entities(q=""), [])


if __name__ == "__main__":
    unittest.main()
