"""Stub components for tracing and eval tests: no LLM, no index, no network."""

from typing import Any

from langsmith.run_helpers import get_current_run_tree

from kotaemon.base import BaseComponent, Document, RetrievedDocument

DOC_TEXT = "Sargon of Akkad founded the empire. It ruled Mesopotamia from Akkad."
ANSWER_TEXT = "Sargon of Akkad."
SEEN: dict = {}


def drain(gen):
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        return stop.value


class StubRetriever(BaseComponent):
    def run(self, text, **kwargs):
        SEEN["retriever"] = get_current_run_tree()
        return [
            RetrievedDocument(
                text=DOC_TEXT,
                score=0.9,
                metadata={"file_name": "Akkadian Empire.pdf", "page_label": "1"},
            )
        ]

    def generate_relevant_scores(self, query, documents):
        SEEN["scorer"] = get_current_run_tree()
        return documents


class StubAnswer(BaseComponent):
    answer_text: str = ANSWER_TEXT
    citation: Any = None

    def stream(self, question, evidence, **kwargs):
        SEEN["answer"] = get_current_run_tree()
        yield Document(channel="chat", content=self.answer_text)
        return Document(
            text=self.answer_text,
            metadata={
                "citation": self.citation,
                "mindmap": None,
                "citation_viz": False,
                "qa_score": None,
            },
        )

    def prepare_citations(self, answer, docs):
        return [], []

    def run(self, *args, **kwargs):
        raise NotImplementedError
