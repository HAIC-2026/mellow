from pathlib import Path
from typing import Any

import pandas as pd

from app.parsers.base import Page


def _cell(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _render_sheet(name: str, df: pd.DataFrame) -> str:
    headers = [_cell(c) for c in df.columns]
    lines = [f"## {name}", ""]
    if headers:
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(_cell(v) for v in row) + " |")
    return "\n".join(lines) + "\n\n"


class ExcelPandasParser:
    name = "pandas"
    version = pd.__version__

    def parse(self, path: Path) -> list[Page]:
        sheets = pd.read_excel(path, sheet_name=None, dtype=object)
        return [
            Page(number=i, title=str(name), markdown=_render_sheet(name, df))
            for i, (name, df) in enumerate(sheets.items(), start=1)
        ]
