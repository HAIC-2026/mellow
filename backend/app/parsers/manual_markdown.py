from pathlib import Path

from app.parsers.base import Page


class ManualMarkdownParser:
    # Placeholder: no source file to read. Content is typed or pasted in the UI and stored via PUT /api/documents/{id}.
    name = "manual-markdown"
    version = "0.1"

    def parse(self, path: Path) -> list[Page]:
        return [Page(number=1, title="Untitled", markdown="")]
