from pathlib import Path
from typing import Protocol


class Parser(Protocol):
    name: str
    version: str

    def parse(self, path: Path) -> str: ...
