import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from ktem.evals.dataset import (
    DatasetError,
    EvalExample,
    example_id,
    load_dataset,
    plan_sync,
    sync_dataset,
)

JSONL = (
    '{"question": "Who founded Akkad?", "reference_answer": "Sargon.", '
    '"source_file": "Akkadian Empire.pdf", "page": 1}\n'
    "\n"
    '{"question": "Exact speed of light?", "reference_answer": "299,792,458 m/s"}\n'
)


def test_load_jsonl(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(JSONL, encoding="utf-8")
    rows = load_dataset(p)
    assert rows[0] == EvalExample(
        "Who founded Akkad?", "Sargon.", "Akkadian Empire.pdf", "1"
    )
    assert rows[1].source_file is None and rows[1].page is None
    assert rows[1].outputs() == {"reference_answer": "299,792,458 m/s"}


def test_load_csv_with_bom_and_blank_optional(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text(
        "question,reference_answer,source_file,page\nQ1,A1,f.pdf,\nQ2,A2,,\n",
        encoding="utf-8-sig",
    )
    rows = load_dataset(p)
    assert rows[0] == EvalExample("Q1", "A1", "f.pdf", None)
    assert rows[1] == EvalExample("Q2", "A2", None, None)


@pytest.mark.parametrize(
    "body, msg",
    [
        ('{"question": " ", "reference_answer": "a"}\n', "line 1: empty question"),
        (
            '{"question": "q", "reference_answer": ""}\n',
            "line 1: empty reference_answer",
        ),
        (
            '{"question": "q", "reference_answer": "a"}\n'
            '{"question": " q ", "reference_answer": "b"}\n',
            "line 2: duplicate question",
        ),
    ],
)
def test_load_rejects_bad_rows(tmp_path, body, msg):
    p = tmp_path / "d.jsonl"
    p.write_text(body, encoding="utf-8")
    with pytest.raises(DatasetError, match=msg):
        load_dataset(p)


def test_load_rejects_unknown_suffix(tmp_path):
    p = tmp_path / "d.txt"
    p.write_text("x")
    with pytest.raises(DatasetError, match=r"\.csv or \.jsonl"):
        load_dataset(p)


def test_example_id_is_deterministic():
    assert example_id("ds", " Who? ") == example_id("ds", "Who?")
    assert example_id("ds", "Who?") != example_id("other", "Who?")
    assert isinstance(example_id("ds", "Who?"), uuid.UUID)


def _existing(ds, ex, outputs=None):
    return SimpleNamespace(
        id=example_id(ds, ex.question),
        inputs=ex.inputs(),
        outputs=outputs if outputs is not None else ex.outputs(),
    )


def test_plan_sync_updates_only_changed_outputs():
    a, b = EvalExample("A?", "a"), EvalExample("B?", "b")
    edited_b = EvalExample("B?", "b, edited")
    c = EvalExample("C?", "c")
    plan = plan_sync(
        "ds", [a, edited_b, c], [_existing("ds", a), _existing("ds", b)], prune=False
    )
    assert [ex for _, ex in plan.create] == [c]
    assert plan.update == [(example_id("ds", "B?"), edited_b)]
    assert plan.delete == []


def test_plan_sync_prune_deletes_missing_only_when_asked():
    a, gone = EvalExample("A?", "a"), EvalExample("Gone?", "g")
    existing = [_existing("ds", a), _existing("ds", gone)]
    kept = plan_sync("ds", [a], existing, prune=False)
    assert kept.delete == [] and kept.stale == 1
    pruned = plan_sync("ds", [a], existing, prune=True)
    assert pruned.delete == [example_id("ds", "Gone?")] and pruned.stale == 0


def test_sync_dataset_creates_dataset_and_examples():
    client = MagicMock()
    client.has_dataset.return_value = False
    client.create_dataset.return_value = SimpleNamespace(id="ds-id")
    ex = EvalExample("A?", "a", "f.pdf", "2")
    sync_dataset(client, "ds", [ex])
    client.create_dataset.assert_called_once()
    kwargs = client.create_examples.call_args.kwargs
    assert kwargs["inputs"] == [{"question": "A?"}]
    assert kwargs["outputs"] == [
        {"reference_answer": "a", "source_file": "f.pdf", "page": "2"}
    ]
    assert kwargs["ids"] == [example_id("ds", "A?")]
    assert kwargs["dataset_id"] == "ds-id"
    client.update_examples.assert_not_called()
    client.delete_example.assert_not_called()
