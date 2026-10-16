import json

from ktem.evals.cleaning import clean_answer, inline_markers, strip_think
from ktem.evals.runner import EvalQAPipeline, citation_facts, make_target

from kotaemon.base import Document
from kotaemon.indices.qa.citation import CiteEvidence
from kotaemon.indices.qa.citation_qa_inline import InlineEvidence
from ktem_tests.eval_stubs import DOC_TEXT, StubAnswer, StubRetriever

DOCS = [Document(text=DOC_TEXT)]


def test_strip_think():
    assert strip_think("<think>a\nb</think>Answer") == "Answer"
    assert strip_think("reasoning</think> Answer") == "Answer"
    assert strip_think("Plain") == "Plain"


def test_inline_markers():
    assert inline_markers("A【1】 B【2, 3】 C【1】") == [1, 2, 3]
    assert inline_markers("A [1] B [2]") == [1, 2]
    assert inline_markers("A【1】 and [2]") == [1]


def test_clean_answer_modes():
    raw = "Sargon【1】 founded it【2】.\n\nCITATION LIST\n\n【1】 start_phrase: x"
    assert clean_answer(raw, "inline") == "Sargon founded it."
    assert clean_answer("<think>x</think>Hi", "off") == "Hi"
    assert clean_answer("See [1]", "highlight") == "See [1]"


def test_citation_facts_highlight():
    ans = Document(
        text="a",
        metadata={
            "citation": CiteEvidence(
                evidences=[
                    "Sargon of Akkad founded the empire",
                    "The comet returns every 76 years",
                ]
            )
        },
    )
    assert citation_facts("highlight", ans, DOCS) == {
        "mode": "highlight",
        "claimed": 2,
        "matched": 1,
    }
    none = Document(text="a", metadata={"citation": None})
    assert citation_facts("highlight", none, DOCS) == {
        "mode": "highlight",
        "claimed": 0,
        "matched": 0,
    }


def test_citation_facts_inline_missing_phrase():
    ans = Document(
        text="Sargon founded it【1】 and ruled【2】",
        metadata={
            "citation": [
                InlineEvidence(
                    idx=1,
                    start_phrase="Sargon of Akkad",
                    end_phrase="founded the empire",
                ),
                InlineEvidence(
                    idx=2, start_phrase="Halley comet orbit text", end_phrase=None
                ),
            ]
        },
    )
    assert citation_facts("inline", ans, DOCS) == {
        "mode": "inline",
        "claimed": 2,
        "matched": 1,
    }


def test_citation_facts_inline_idx_none_uses_position():
    ans = Document(
        text="X【1】",
        metadata={
            "citation": [
                InlineEvidence(
                    idx=None,
                    start_phrase="Sargon of Akkad",
                    end_phrase="founded the empire",
                )
            ]
        },
    )
    assert citation_facts("inline", ans, DOCS)["matched"] == 1


def test_citation_facts_off():
    assert citation_facts("off", Document(text="a"), DOCS) == {
        "mode": "off",
        "claimed": 0,
        "matched": 0,
    }


def test_target_returns_plain_json_and_builds_fresh_pipelines():
    built = []

    def build():
        p = EvalQAPipeline(
            retrievers=[StubRetriever()],
            answering_pipeline=StubAnswer(
                answer_text="<think>hm</think>Sargon of Akkad."
            ),
        )
        built.append(p)
        return p

    target = make_target(build, "highlight")
    out = target({"question": "Who founded Akkad?"})
    target({"question": "Again?"})
    assert len(built) == 2 and built[0] is not built[1]
    assert out["answer"] == "Sargon of Akkad."
    assert out["raw_answer"] == "<think>hm</think>Sargon of Akkad."
    assert out["contexts"] == [
        {
            "doc_id": out["contexts"][0]["doc_id"],
            "file_name": "Akkadian Empire.pdf",
            "page_label": "1",
            "type": "",
            "text": DOC_TEXT,
        }
    ]
    assert "Content from Akkadian Empire.pdf" in out["evidence"]
    assert out["citations"] == {"mode": "highlight", "claimed": 0, "matched": 0}
    json.dumps(out)


def test_eval_pipeline_keeps_simple_id():
    assert EvalQAPipeline.get_info()["id"] == "simple"
