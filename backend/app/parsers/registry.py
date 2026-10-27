from app.parsers.base import Parser
from app.parsers.excel_pandas import ExcelPandasParser
from app.parsers.manual_markdown import ManualMarkdownParser
from app.parsers.pdf_markitdown import MarkItDownPdfParser

PARSERS: dict[str, Parser] = {
    ".xlsx": ExcelPandasParser(),
    ".pdf": MarkItDownPdfParser(),
}


def get_parser(extension: str) -> Parser | None:
    return PARSERS.get(extension.lower())


# Not in PARSERS: it has no file extension, so it must not appear in /api/parsers or the upload accept list.
MANUAL_PARSER = ManualMarkdownParser()
