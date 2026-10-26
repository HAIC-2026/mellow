import os

from google import genai

DEFAULT_MODEL = "gemini-2.5-flash"

PROMPT_TEMPLATE = """You edit one sheet of a Markdown document converted from an Excel workbook.

Apply the user's instruction to the sheet below. Keep the "## " sheet heading and all
unchanged content exactly as it is. Output only the complete revised Markdown for this
sheet, with no code fences and no commentary.

Instruction:
{instruction}

Sheet:
{sheet}
"""


def edit_sheet(sheet_markdown: str, instruction: str) -> str:
    client = genai.Client()
    response = client.models.generate_content(
        model=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL),
        contents=PROMPT_TEMPLATE.format(instruction=instruction, sheet=sheet_markdown),
    )
    text = response.text or ""
    return _strip_code_fence(text)


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        return "\n".join(lines) + "\n"
    return text
