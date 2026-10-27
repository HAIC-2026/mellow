from app.parsers.base import Parser
from app.parsers.excel_pandas import ExcelPandasParser
from app.parsers.pdf_markitdown import MarkItDownPdfParser

PARSERS: dict[str, Parser] = {
    ".xlsx": ExcelPandasParser(),
    ".pdf": MarkItDownPdfParser(),
}


def get_parser(extension: str) -> Parser | None:
    return PARSERS.get(extension.lower())
