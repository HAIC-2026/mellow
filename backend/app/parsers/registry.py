from app.parsers.base import Parser
from app.parsers.markitdown_parser import MarkItDownParser

PARSERS: dict[str, Parser] = {".xlsx": MarkItDownParser()}


def get_parser(extension: str) -> Parser | None:
    return PARSERS.get(extension.lower())
