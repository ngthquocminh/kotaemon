"""Answer pipelines import ktem (citation_qa -> ktem.llms): run from the repo root."""

from langsmith.run_helpers import get_current_run_tree, tracing_context

from kotaemon.base import BaseComponent
from kotaemon.base.schema import LLMInterface
from kotaemon.indices.qa.citation_qa import AnswerWithContextPipeline
from kotaemon.indices.qa.citation_qa_inline import AnswerWithInlineCitation
from kotaemon.llms import ChatLLM

SEEN: dict = {}


class StubLLM(ChatLLM):
    def invoke(self, messages, **kwargs):
        return LLMInterface(content="FINAL ANSWER\nSargon of Akkad.")

    def stream(self, messages, **kwargs):
        yield LLMInterface(content="FINAL ANSWER\nSargon of Akkad.")


class StubCitation(BaseComponent):
    def run(self, context, question):
        SEEN["citation"] = get_current_run_tree()
        return None


class StubMindmap(BaseComponent):
    def run(self, context, question):
        SEEN["mindmap"] = get_current_run_tree()
        return None


def _drain(gen):
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        return stop.value


def test_citation_and_mindmap_threads_nest_under_answer_span():
    SEEN.clear()
    pipe = AnswerWithContextPipeline(
        llm=StubLLM(),
        citation_pipeline=StubCitation(),
        create_mindmap_pipeline=StubMindmap(),
        enable_citation=True,
        enable_mindmap=True,
    )
    with tracing_context(enabled="local"):
        answer = _drain(
            pipe.stream(question="who?", evidence="Sargon of Akkad founded it.")
        )
    assert answer.text == "FINAL ANSWER\nSargon of Akkad."
    assert SEEN["citation"] is not None
    assert SEEN["citation"].name == "answer_with_context"
    assert SEEN["mindmap"] is not None
    assert SEEN["mindmap"].name == "answer_with_context"
    assert SEEN["citation"].outputs == {"answer": "FINAL ANSWER\nSargon of Akkad."}


def test_inline_mindmap_thread_nests_under_inline_span():
    SEEN.clear()
    pipe = AnswerWithInlineCitation(
        llm=StubLLM(), create_mindmap_pipeline=StubMindmap(), enable_mindmap=True
    )
    with tracing_context(enabled="local"):
        _drain(pipe.stream(question="who?", evidence="Sargon of Akkad founded it."))
    assert SEEN["mindmap"].name == "answer_with_inline_citation"
