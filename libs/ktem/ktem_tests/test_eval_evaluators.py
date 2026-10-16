import pytest
from ktem.evals.evaluators import (
    citation_validity,
    make_correctness,
    make_groundedness,
    parse_label,
    reciprocal_rank,
    retrieval_hit,
)

C = ["partially_correct", "incorrect", "correct"]
G = ["partially_grounded", "ungrounded", "grounded"]


def ctx(name, page="1", type_=""):
    return {
        "doc_id": "d",
        "file_name": name,
        "page_label": page,
        "type": type_,
        "text": "t",
    }


OUT = {
    "answer": "Sargon.",
    "evidence": "Sargon of Akkad founded it.",
    "contexts": [ctx("other.pdf"), ctx("dir/Halley's Comet.PDF", "3")],
    "citations": {"mode": "highlight", "claimed": 4, "matched": 3},
}
REF = {
    "reference_answer": "Sargon of Akkad.",
    "source_file": "halley's comet.pdf",
    "page": "3",
}


@pytest.mark.parametrize(
    "text, labels, want",
    [
        ("LABEL: Partially correct\nREASON: misses date", C, "partially_correct"),
        ("label: partially-correct", C, "partially_correct"),
        ("The answer is incorrect.", C, "incorrect"),
        ("Correct", C, "correct"),
        ("LABEL: ungrounded", G, "ungrounded"),
        ("I cannot tell", C, None),
    ],
)
def test_parse_label(text, labels, want):
    assert parse_label(text, labels) == want


def test_correctness_scores_label_and_prompt_has_inputs():
    prompts = []

    def judge(p):
        prompts.append(p)
        return "LABEL: partially_correct\nREASON: misses date"

    res = make_correctness(judge)({"question": "Who?"}, OUT, REF)
    assert res["key"] == "correctness" and res["score"] == 0.5
    assert "misses date" in res["comment"]
    assert "Who?" in prompts[0]
    assert "Sargon of Akkad." in prompts[0] and "Sargon." in prompts[0]


def test_correctness_empty_answer_scores_zero_without_judge():
    def judge(_):
        raise AssertionError("judge must not be called")

    res = make_correctness(judge)({"question": "q"}, {**OUT, "answer": ""}, REF)
    assert res["score"] == 0.0


def test_correctness_missing_outputs_is_none():
    assert (
        make_correctness(lambda p: "correct")({"question": "q"}, {}, REF)["score"]
        is None
    )


def test_correctness_unparseable_or_error_is_none():
    assert (
        make_correctness(lambda p: "no idea")({"question": "q"}, OUT, REF)["score"]
        is None
    )

    def boom(_):
        raise RuntimeError("boom")

    res = make_correctness(boom)({"question": "q"}, OUT, REF)
    assert res["score"] is None and "boom" in res["comment"]


def test_groundedness():
    assert make_groundedness(lambda p: "LABEL: grounded")(OUT)["score"] == 1.0
    assert (
        make_groundedness(lambda p: "grounded")({**OUT, "evidence": ""})["score"]
        is None
    )
    assert (
        make_groundedness(lambda p: "grounded")({**OUT, "answer": ""})["score"] is None
    )
    assert make_groundedness(lambda p: "grounded")({})["score"] is None


def test_retrieval_hit_file_and_page():
    assert retrieval_hit(OUT, REF)["score"] == 1.0
    assert retrieval_hit(OUT, {**REF, "page": "4"})["score"] == 0.0
    no_page = {"reference_answer": "x", "source_file": "halley's comet.pdf"}
    assert retrieval_hit(OUT, no_page)["score"] == 1.0
    assert retrieval_hit(OUT, {"reference_answer": "x"})["score"] is None
    assert retrieval_hit({}, REF)["score"] is None


def test_retrieval_ignores_thumbnails():
    out = {**OUT, "contexts": [ctx("Halley's Comet.pdf", "3", "thumbnail")]}
    assert retrieval_hit(out, REF)["score"] == 0.0
    assert reciprocal_rank(out, REF)["score"] == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(OUT, REF)["score"] == 0.5
    assert reciprocal_rank({**OUT, "contexts": [ctx("other.pdf")]}, REF)["score"] == 0.0
    assert reciprocal_rank(OUT, {"reference_answer": "x"})["score"] is None


def test_citation_validity():
    assert citation_validity(OUT)["score"] == 0.75
    off = {**OUT, "citations": {"mode": "off", "claimed": 0, "matched": 0}}
    assert citation_validity(off)["score"] is None
    none_claimed = {**OUT, "citations": {"mode": "inline", "claimed": 0, "matched": 0}}
    assert citation_validity(none_claimed)["score"] is None
    assert citation_validity({})["score"] is None


def test_retrieval_ignores_relabelled_page_thumbnails():
    # VectorRetrieval relabels raw page thumbnails as type "image"
    thumb = {**ctx("Halley's Comet.pdf", "3", "image"), "text": "Page thumbnail"}
    assert retrieval_hit({**OUT, "contexts": [thumb]}, REF)["score"] == 0.0
    # an image doc linked to a real chunk keeps the chunk text and still counts
    linked = {
        **ctx("Halley's Comet.pdf", "3", "image"),
        "text": "Its aphelion is 35 AU",
    }
    assert retrieval_hit({**OUT, "contexts": [linked]}, REF)["score"] == 1.0


def test_judge_think_block_is_ignored():
    reply = "<think>label: correct? no, the year differs</think>\nLABEL: incorrect"
    res = make_correctness(lambda p: reply)({"question": "q"}, OUT, REF)
    assert res["score"] == 0.0


def test_scores_fit_langsmith_precision():
    # LangSmith rejects feedback scores with more than 4 decimal places (HTTP 422)
    out = {**OUT, "contexts": [ctx("a.pdf"), ctx("b.pdf"), ctx("halley's comet.pdf", "3")]}
    assert reciprocal_rank(out, REF)["score"] == 0.3333
    thirds = {**OUT, "citations": {"mode": "highlight", "claimed": 3, "matched": 2}}
    assert citation_validity(thirds)["score"] == 0.6667
