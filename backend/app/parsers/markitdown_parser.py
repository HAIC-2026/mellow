from importlib.metadata import version
from pathlib import Path

from markitdown import MarkItDown, StreamInfo


class MarkItDownParser:
    name = "markitdown"
    version = version("markitdown")

    def parse(self, path: Path) -> str:
        result = MarkItDown().convert(str(path), stream_info=StreamInfo(extension=path.suffix))
        return result.text_content
