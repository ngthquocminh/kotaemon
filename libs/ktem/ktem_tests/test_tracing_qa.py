from ktem.reasoning.simple import FullQAPipeline
from langsmith.run_helpers import tracing_context

from ktem_tests.eval_stubs import (
    ANSWER_TEXT,
    DOC_TEXT,
    SEEN,
    StubAnswer,
    StubRetriever,
    drain,
)


def test_qa_pipeline_spans_nest():
    SEEN.clear()
    pipe = FullQAPipeline(retrievers=[StubRetriever()], answering_pipeline=StubAnswer())
    with tracing_context(enabled="local"):
        answer = drain(pipe.stream("who founded akkad?", conv_id="c1", history=[]))
    assert answer.text == ANSWER_TEXT
    root = SEEN["answer"]
    assert root.name == "kotaemon.qa"
    assert root.outputs == {"answer": ANSWER_TEXT}
    retrieve = SEEN["retriever"]
    assert retrieve.name == "retrieve" and retrieve.run_type == "retriever"
    assert retrieve.parent_run_id == root.id
    doc = retrieve.outputs["documents"][0]
    assert doc["page_content"] == DOC_TEXT
    assert doc["metadata"]["file_name"] == "Akkadian Empire.pdf"
    assert doc["metadata"]["page_label"] == "1"
    assert doc["metadata"]["score"] == 0.9
    assert set(retrieve.outputs) == {"documents"}  # HTML infos dropped
    assert SEEN["scorer"].id == root.id  # scorer thread kept the context


def test_qa_pipeline_untraced_still_answers():
    pipe = FullQAPipeline(retrievers=[StubRetriever()], answering_pipeline=StubAnswer())
    assert drain(pipe.stream("q", conv_id="c2", history=[])).text == ANSWER_TEXT
