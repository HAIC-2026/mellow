import re


def normalize(name: str) -> str:
    """Lowercase, replace punctuation with spaces, collapse whitespace.

    Entity names are compared in this form, in extraction and in entity resolution.
    """
    return " ".join(re.sub(r"[^\w\s]", " ", name.lower()).split())
