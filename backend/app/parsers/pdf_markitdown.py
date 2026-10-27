from pathlib import Path

import markitdown
from markitdown import MarkItDown

from app.parsers.base import Page


class MarkItDownPdfParser:
    name = "markitdown"
    version = markitdown.__version__

    def __init__(self) -> None:
        self._converter = MarkItDown()

    def parse(self, path: Path) -> list[Page]:
        # markitdown returns one markdown string for the whole PDF; it has no page boundaries.
        text = self._converter.convert(str(path)).text_content
        return [Page(number=1, title=path.stem, markdown=text)]
