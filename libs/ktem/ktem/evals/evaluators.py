"""Evaluators for ``langsmith.evaluation.evaluate``.

Each is a plain function of ``outputs`` / ``reference_outputs`` (langsmith maps
arguments by name) returning ``{"key", "score", "comment"}``. A score of
``None`` means "not applicable" and is left out of the means.

Must not import ``ktem.app`` or theflow settings.
"""

import os
import re
from typing import Callable, Iterable

from .cleaning import strip_think

Judge = Callable[[str], str]

CORRECTNESS_LABELS = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}
GROUNDEDNESS_LABELS = {
    "grounded": 1.0,
    "partially_grounded": 0.5,
    "ungrounded": 0.0,
}

CORRECTNESS_PROMPT = """You are grading an answer to a question against a reference answer.

Question: {question}
Reference answer: {reference}
Answer to grade: {answer}

Judge only the facts, not style or length. Is the answer correct, partially correct, or incorrect compared to the reference?

Reply with exactly two lines:
LABEL: <correct | partially_correct | incorrect>
REASON: <one sentence>"""  # noqa: E501

GROUNDEDNESS_PROMPT = """You are checking whether an answer is supported by the context it was written from.

Context:
{evidence}

Answer:
{answer}

Is every claim in the answer supported by the context (grounded), only some claims (partially_grounded), or are the main claims unsupported (ungrounded)?

Reply with exactly two lines:
LABEL: <grounded | partially_grounded | ungrounded>
REASON: <one sentence>"""  # noqa: E501

_NO_OUTPUTS = "target failed; no outputs"


def _result(key: str, score: float | None, comment: str) -> dict:
    # LangSmith rejects scores with more than 4 decimal places (HTTP 422)
    if score is not None:
        score = round(score, 4)
    return {"key": key, "score": score, "comment": comment}


def parse_label(text: str, labels: Iterable[str]) -> str | None:
    """Find a judge label in free text, tolerating case, spaces and hyphens."""
    labels = list(labels)
    norm = re.sub(r"partially[\s-]+", "partially_", (text or "").lower())
    explicit = re.search(r"label:\s*([a-z_]+)", norm)
    if explicit and explicit.group(1) in labels:
        return explicit.group(1)
    best: tuple[int, str] | None = None
    for label in labels:
        match = re.search(rf"(?<![a-z_]){label}(?![a-z_])", norm)
        if match and (best is None or match.start() < best[0]):
            best = (match.start(), label)
    return best[1] if best else None


def _judge(key: str, judge: Judge, prompt: str, labels: dict[str, float]) -> dict:
    try:
        reply = judge(prompt)
    except Exception as exc:
        return _result(key, None, f"judge error: {exc}")
    label = parse_label(strip_think(reply), labels)
    if label is None:
        return _result(key, None, f"unparseable judge reply: {reply}")
    return _result(key, labels[label], reply)


def make_correctness(judge: Judge) -> Callable[[dict, dict, dict], dict]:
    def correctness(inputs, outputs, reference_outputs):
        if not outputs or "answer" not in outputs:
            return _result("correctness", None, _NO_OUTPUTS)
        if not outputs["answer"].strip():
            return _result("correctness", 0.0, "empty answer")
        prompt = CORRECTNESS_PROMPT.format(
            question=inputs.get("question", ""),
            reference=reference_outputs.get("reference_answer", ""),
            answer=outputs["answer"],
        )
        return _judge("correctness", judge, prompt, CORRECTNESS_LABELS)

    return correctness


def make_groundedness(judge: Judge) -> Callable[[dict], dict]:
    def groundedness(outputs):
        if not outputs or "answer" not in outputs:
            return _result("groundedness", None, _NO_OUTPUTS)
        if not outputs.get("evidence"):
            return _result("groundedness", None, "no evidence retrieved")
        if not outputs["answer"].strip():
            return _result("groundedness", None, "empty answer")
        prompt = GROUNDEDNESS_PROMPT.format(
            evidence=outputs["evidence"], answer=outputs["answer"]
        )
        return _judge("groundedness", judge, prompt, GROUNDEDNESS_LABELS)

    return groundedness


def _is_thumbnail(context: dict) -> bool:
    # VectorRetrieval relabels raw page thumbnails as type "image"; their text
    # stays "Page thumbnail". Image docs linked to a chunk carry the chunk text.
    return (
        context.get("type") == "thumbnail"
        or (context.get("text") or "").strip() == "Page thumbnail"
    )


def _first_hit_rank(outputs: dict, reference_outputs: dict) -> int | None:
    """1-based rank of the first non-thumbnail context matching the source."""
    target = os.path.basename(reference_outputs["source_file"]).lower()
    page = reference_outputs.get("page")
    rank = 0
    for context in outputs["contexts"]:
        if _is_thumbnail(context):
            continue
        rank += 1
        name = os.path.basename(context.get("file_name") or "").lower()
        if name != target:
            continue
        if page not in (None, "") and str(context.get("page_label")) != str(page):
            continue
        return rank
    return None


def _retrieval_skip(key: str, outputs: dict, reference_outputs: dict) -> dict | None:
    if not reference_outputs.get("source_file"):
        return _result(key, None, "no source_file in the dataset row")
    if not outputs or "contexts" not in outputs:
        return _result(key, None, _NO_OUTPUTS)
    return None


def retrieval_hit(outputs, reference_outputs):
    skip = _retrieval_skip("retrieval_hit", outputs, reference_outputs)
    if skip:
        return skip
    rank = _first_hit_rank(outputs, reference_outputs)
    return _result(
        "retrieval_hit",
        1.0 if rank else 0.0,
        f"first hit at rank {rank}" if rank else "source not retrieved",
    )


def reciprocal_rank(outputs, reference_outputs):
    skip = _retrieval_skip("reciprocal_rank", outputs, reference_outputs)
    if skip:
        return skip
    rank = _first_hit_rank(outputs, reference_outputs)
    return _result(
        "reciprocal_rank",
        1.0 / rank if rank else 0.0,
        f"rank {rank}" if rank else "source not retrieved",
    )


def citation_validity(outputs):
    if not outputs or "citations" not in outputs:
        return _result("citation_validity", None, _NO_OUTPUTS)
    facts = outputs["citations"]
    if facts.get("mode") == "off":
        return _result("citation_validity", None, "citations off")
    if not facts.get("claimed"):
        return _result("citation_validity", None, "no citations claimed")
    return _result(
        "citation_validity",
        facts["matched"] / facts["claimed"],
        f"{facts['matched']}/{facts['claimed']} citations found in the contexts",
    )


def build_evaluators(judge: Judge) -> list[Callable]:
    return [
        make_correctness(judge),
        make_groundedness(judge),
        retrieval_hit,
        reciprocal_rank,
        citation_validity,
    ]
