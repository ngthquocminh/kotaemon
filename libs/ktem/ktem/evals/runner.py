"""Run one dataset question through the ``simple`` pipeline for ``evaluate``."""

import uuid
from typing import Callable, Sequence

from ktem.reasoning.simple import FullQAPipeline

from kotaemon.base import Document
from kotaemon.indices.qa.utils import find_start_end_phrase, find_text

from .cleaning import clean_answer, inline_markers


class EvalQAPipeline(FullQAPipeline):
    """``FullQAPipeline`` that keeps the docs ``retrieve()`` returned."""

    def retrieve(self, message, history):
        docs, infos = super().retrieve(message, history)
        # theflow passes "_" attributes straight through (theflow/base.py)
        self._eval_docs = docs
        return docs, infos


def _quote_matched(quote: str, docs: Sequence[Document]) -> bool:
    # same rule as AnswerWithContextPipeline.match_evidence_with_context
    for doc in docs:
        for start, end in find_text(quote, doc.text):
            if "|" not in doc.text[start:end]:
                return True
    return False


def citation_facts(mode: str, answer: Document, docs: Sequence[Document]) -> dict:
    """How many citations the answer claims and how many match a retrieved doc."""
    if mode == "highlight":
        citation = answer.metadata.get("citation")
        quotes = list(citation.evidences) if citation else []
        matched = sum(_quote_matched(q, docs) for q in quotes)
        return {"mode": mode, "claimed": len(quotes), "matched": matched}

    if mode == "inline":
        claimed = set(inline_markers(answer.text))
        found = set()
        for position, evidence in enumerate(answer.metadata.get("citation") or [], 1):
            number = evidence.idx if evidence.idx is not None else position
            for doc in docs:
                match, _ = find_start_end_phrase(
                    evidence.start_phrase, evidence.end_phrase, doc.text
                )
                if match is not None:
                    found.add(number)
                    break
        return {"mode": mode, "claimed": len(claimed), "matched": len(claimed & found)}

    return {"mode": mode, "claimed": 0, "matched": 0}


def make_target(
    build_pipeline: Callable[[], EvalQAPipeline], citation_mode: str
) -> Callable[[dict], dict]:
    def target(inputs: dict) -> dict:
        pipeline = build_pipeline()  # fresh per question, as the app does
        stream = pipeline.stream(
            inputs["question"], conv_id=f"eval-{uuid.uuid4()}", history=[]
        )
        answer = None
        try:
            while True:
                next(stream)
        except StopIteration as stop:
            answer = stop.value
        if answer is None:
            raise RuntimeError("pipeline returned no answer")

        docs = getattr(pipeline, "_eval_docs", [])
        evidence = pipeline.evidence_pipeline(docs).content[1]
        return {
            "answer": clean_answer(answer.text, citation_mode),
            "raw_answer": answer.text,
            "contexts": [
                {
                    "doc_id": doc.doc_id,
                    "file_name": doc.metadata.get("file_name"),
                    "page_label": doc.metadata.get("page_label"),
                    "type": doc.metadata.get("type", ""),
                    "text": doc.text,
                }
                for doc in docs
            ],
            "evidence": evidence,
            "citations": citation_facts(citation_mode, answer, docs),
        }

    return target
