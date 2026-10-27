import re
from dataclasses import dataclass, field

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+•▪◦]|\d+[.)])\s+")
# PDF text is hard-wrapped; "decision-\nmaking" should read as "decision-making".
HYPHEN_WRAP_RE = re.compile(r"-\s*\n\s*(?=[a-z])")
# Sentence end: terminal punctuation, optional closing quotes/brackets, whitespace, then a capital, digit or opener.
SENTENCE_END_RE = re.compile(r"[.!?][\"'”’)\]]*(?=\s+[\"'“‘(\[A-Z0-9])")
# Sentence-splitting rules lean toward not splitting: an under-split only widens a highlight,
# an over-split yields citable fragments like "2." or "The U.S.".
# Titles that are followed by a capitalized name. Other abbreviations are caught by the rules below.
ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs"}
# "U.S.", "p.m.", "e.g.", "i.e."
DOTTED_ABBREVIATION_RE = re.compile(r"^(?:[A-Za-z]\.)+[A-Za-z]$")
# "Fig. 3", "Approx. 5", "Sec. 4", "No. 5", "Jan. 5"
MAX_ABBREVIATION_BEFORE_DIGIT = 6
SHORT_NUMBER_RE = re.compile(r"\d{1,3}(?!\d)")
NEXT_LIST_MARKER_RE = re.compile(r"\d{1,2}[.)](?:\s|$)")
# Inline numbered list marker: "Steps: 1. Open ... 2. Save ..."
LIST_MARKER_RE = re.compile(r"^\d{1,2}$")
MAX_CHUNK_CHARS = 1500


@dataclass
class Span:
    kind: str  # heading | sentence | table_row
    start: int  # offsets into the original page markdown: page[start:end] == text
    end: int
    text: str
    llm_text: str  # what the LLM sees; table rows are prefixed with their column headers
    heading_path: str


@dataclass
class Chunk:
    heading_path: str
    spans: list[Span] = field(default_factory=list)

    @property
    def text(self) -> str:
        body = "\n".join(s.llm_text for s in self.spans)
        return f"{self.heading_path}\n{body}" if self.heading_path else body


def split_spans(markdown: str) -> list[Span]:
    spans: list[Span] = []
    headings: list[tuple[int, str]] = []
    table_headers: list[str] | None = None
    block_start: int | None = None
    block_end = 0

    def heading_path() -> str:
        return " › ".join(title for _, title in headings)

    def flush_block() -> None:
        nonlocal block_start
        if block_start is not None:
            spans.extend(_split_sentences(markdown, block_start, block_end, heading_path()))
            block_start = None

    lines = markdown.splitlines(keepends=True)
    offset = 0
    for i, line in enumerate(lines):
        line_start, offset = offset, offset + len(line)
        content = line.rstrip("\r\n")
        line_end = line_start + len(content)
        stripped = content.strip()

        if not stripped:
            flush_block()
            table_headers = None
            continue

        heading = HEADING_RE.match(content)
        if heading:
            flush_block()
            table_headers = None
            level, title = len(heading.group(1)), heading.group(2)
            headings = [h for h in headings if h[0] < level] + [(level, title)]
            spans.append(Span("heading", *_trim(markdown, line_start, line_end), title, heading_path()))
            continue

        if stripped.startswith("|"):
            flush_block()
            if TABLE_SEPARATOR_RE.match(content):
                continue
            cells = _table_cells(stripped)
            next_line = lines[i + 1] if i + 1 < len(lines) else ""
            if table_headers is None and TABLE_SEPARATOR_RE.match(next_line):
                table_headers = cells
                llm_text = " | ".join(cells)
            elif table_headers:
                llm_text = "; ".join(
                    f"{h}: {c}" if h else c for h, c in zip(table_headers, cells) if c
                )
            else:
                llm_text = " | ".join(cells)
            start, end, text = _trim(markdown, line_start, line_end)
            spans.append(Span("table_row", start, end, text, llm_text, heading_path()))
            continue

        table_headers = None
        if LIST_ITEM_RE.match(content):
            flush_block()
        if block_start is None:
            block_start = line_start
        block_end = line_end

    flush_block()
    return spans


def group_chunks(spans: list[Span]) -> list[Chunk]:
    chunks: list[Chunk] = []
    current: Chunk | None = None
    size = 0
    for span in spans:
        new_section = current is None or span.kind == "heading" or span.heading_path != current.heading_path
        if new_section or size + len(span.llm_text) > MAX_CHUNK_CHARS:
            current = Chunk(span.heading_path)
            chunks.append(current)
            size = 0
        current.spans.append(span)
        size += len(span.llm_text) + 1
    return chunks


def _split_sentences(markdown: str, start: int, end: int, heading_path: str) -> list[Span]:
    block = markdown[start:end]
    cuts = [0]
    for match in SENTENCE_END_RE.finditer(block):
        if not _is_sentence_end(block, match, cuts[-1]):
            continue
        cuts.append(match.end())
    cuts.append(len(block))

    spans = []
    for a, b in zip(cuts, cuts[1:]):
        s, e, text = _trim(markdown, start + a, start + b)
        if text:
            flat = " ".join(HYPHEN_WRAP_RE.sub("-", text).split())
            spans.append(Span("sentence", s, e, text, flat, heading_path))
    return spans


def _is_sentence_end(block: str, match: re.Match[str], last_cut: int) -> bool:
    if block[match.start()] != ".":
        return True
    word = re.search(r"([\w.]+)\.$", block[: match.start() + 1])
    if word is None:
        return True
    token = word.group(1)
    rest = block[match.end() :].lstrip()

    if token.lower() in ABBREVIATIONS or (len(token) == 1 and token.isalpha()):
        return False
    if DOTTED_ABBREVIATION_RE.match(token):
        return False
    # Short numbers only, so "enroll. 2026 ..." still splits. A following list marker ("2. ") always splits.
    if (
        token.isalpha()
        and len(token) <= MAX_ABBREVIATION_BEFORE_DIGIT
        and SHORT_NUMBER_RE.match(rest)
        and not NEXT_LIST_MARKER_RE.match(rest)
    ):
        return False
    if LIST_MARKER_RE.match(token):
        # A list marker sits at block start, after ":" or ";", or right after a sentence end.
        before = block[last_cut : word.start(1)].strip()
        if not before or before.endswith((":", ";")):
            return False
    return True


def _table_cells(row: str) -> list[str]:
    inner = row.strip()
    if inner.startswith("|"):
        inner = inner[1:]
    if inner.endswith("|") and not inner.endswith("\\|"):
        inner = inner[:-1]
    cells = re.split(r"(?<!\\)\|", inner)
    return [c.strip().replace("\\|", "|").replace("<br>", " ") for c in cells]


def _trim(markdown: str, start: int, end: int) -> tuple[int, int, str]:
    while start < end and markdown[start].isspace():
        start += 1
    while end > start and markdown[end - 1].isspace():
        end -= 1
    return start, end, markdown[start:end]
