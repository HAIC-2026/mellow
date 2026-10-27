from app.parsers.base import Parser
from app.parsers.excel_pandas import ExcelPandasParser

PARSERS: dict[str, Parser] = {".xlsx": ExcelPandasParser()}


def get_parser(extension: str) -> Parser | None:
    return PARSERS.get(extension.lower())
