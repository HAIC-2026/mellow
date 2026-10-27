import json
import unittest

from app.config import DATA_DIR
from app.rag.spans import split_spans

SENTENCE_CASES = [
    (
        "The budget is $1.5 M this year. Weight is 10.5 kg per unit.",
        ["The budget is $1.5 M this year.", "Weight is 10.5 kg per unit."],
    ),
    (
        "Upgrade to v2.1 before Friday. See Section 3.A for details.",
        ["Upgrade to v2.1 before Friday.", "See Section 3.A for details."],
    ),
    (
        "Read [Release Notes 1.0](http://example.com/v1.0/notes.html) first. Then deploy.",
        ["Read [Release Notes 1.0](http://example.com/v1.0/notes.html) first.", "Then deploy."],
    ),
    (
        "Steps: 1. Open the file. 2. Save it. 3. Close it.",
        ["Steps: 1. Open the file.", "2. Save it.", "3. Close it."],
    ),
    (
        "The U.S. Department of Agriculture reported it. Approx. 5 farms closed.",
        ["The U.S. Department of Agriculture reported it.", "Approx. 5 farms closed."],
    ),
    (
        "See Fig. 3 for the chart. Meeting at 3 p.m. Monday.",
        ["See Fig. 3 for the chart.", "Meeting at 3 p.m. Monday."],
    ),
    ("Funding was $1.5M. The rest came later.", ["Funding was $1.5M.", "The rest came later."]),
    ("It ended in 2022. 60,378 acres were harvested.", ["It ended in 2022.", "60,378 acres were harvested."]),
    ("There were 12. The rest left.", ["There were 12.", "The rest left."]),
    ("Nobody will enroll. 2026 Use Case Page 1 of 6", ["Nobody will enroll.", "2026 Use Case Page 1 of 6"]),
    ("Contractor enters number. 2)", ["Contractor enters number.", "2)"]),
    ("Contact R. Malia Taum-Deenik. Call Dr. Smith.", ["Contact R. Malia Taum-Deenik.", "Call Dr. Smith."]),
]


class SentenceSplitTest(unittest.TestCase):
    def test_cases(self) -> None:
        for text, expected in SENTENCE_CASES:
            with self.subTest(text=text):
                self.assertEqual([s.text for s in split_spans(text)], expected)

    def test_offsets_match_source_on_stored_documents(self) -> None:
        files = sorted((DATA_DIR / "markdown").glob("*/*.json"))
        if not files:
            self.skipTest("no stored documents")
        for path in files:
            for page in json.loads(path.read_text(encoding="utf-8")):
                markdown = page["markdown"]
                for span in split_spans(markdown):
                    with self.subTest(file=str(path), span=span.text[:40]):
                        self.assertEqual(markdown[span.start : span.end], span.text)
                        self.assertNotIn("\n\n", span.text)


if __name__ == "__main__":
    unittest.main()
