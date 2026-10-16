"""Turn raw answer text into what a reader sees. No app imports."""

import re

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
# same patterns as AnswerWithInlineCitation.replace_citation_with_link
_MARKER = re.compile(r"【([\d,\s]+)】")
_ALT_MARKER = re.compile(r"\[(\d+)\]")
_CITATION_LIST = "CITATION LIST"


def strip_think(text: str) -> str:
    text = _THINK.sub("", text or "")
    if "</think>" in text:  # opening tag omitted by some models
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


def _marker_pattern(text: str) -> re.Pattern:
    return _MARKER if _MARKER.search(text) else _ALT_MARKER


def inline_markers(text: str) -> list[int]:
    """Distinct citation numbers in first-seen order."""
    numbers: list[int] = []
    for match in _marker_pattern(text).finditer(text or ""):
        for part in match.group(1).split(","):
            part = part.strip()
            if part.isdigit() and int(part) not in numbers:
                numbers.append(int(part))
    return numbers


def clean_answer(text: str, mode: str) -> str:
    text = strip_think(text)
    if mode == "inline":
        text = text.split(_CITATION_LIST, 1)[0]
        text = _marker_pattern(text).sub("", text)
    return text.strip()
