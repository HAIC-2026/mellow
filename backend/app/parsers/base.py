from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class Page:
    number: int
    title: str
    markdown: str


class Parser(Protocol):
    name: str
    version: str

    def parse(self, path: Path) -> list[Page]: ...
