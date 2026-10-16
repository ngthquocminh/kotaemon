import inspect
from types import SimpleNamespace

import pytest
from ktem.evals import run as cli
from ktem.evals.bootstrap import BootstrapError, EvalContext
from ktem.evals.runner import EvalQAPipeline
from langsmith.evaluation import EvaluationResult

from ktem_tests.eval_stubs import StubAnswer, StubRetriever

ROWS = (
    '{"question": "Who founded Akkad?", "reference_answer": "Sargon of Akkad.", '
    '"source_file": "Akkadian Empire.pdf", "page": "1"}\n'
)


def fake_evaluate(calls, examples):
    def _evaluate(
        target,
        *,
        data,
        evaluators,
        experiment_prefix,
        max_concurrency,
        metadata,
        client,
    ):
        calls.update(
            data=data,
            prefix=experiment_prefix,
            max_concurrency=max_concurrency,
            metadata=metadata,
        )
        rows = []
        for ex in examples:
            outputs = target(ex.inputs())
            args = {
                "inputs": ex.inputs(),
                "outputs": outputs,
                "reference_outputs": ex.outputs(),
            }
            results = [
                EvaluationResult(
                    **ev(**{k: args[k] for k in inspect.signature(ev).parameters})
                )
                for ev in evaluators
            ]
            rows.append({"evaluation_results": {"results": results}})
        return SimpleNamespace(experiment_name="exp-1", rows=rows)

    return _evaluate


@pytest.fixture
def setup(tmp_path, monkeypatch, mocker):
    monkeypatch.setattr(cli, "export_langsmith_env", lambda *a, **k: None)
    monkeypatch.setenv("LANGSMITH_API_KEY", "test")
    path = tmp_path / "akkad.jsonl"
    path.write_text(ROWS, encoding="utf-8")
    ctx = EvalContext(
        build_pipeline=lambda: EvalQAPipeline(
            retrievers=[StubRetriever()], answering_pipeline=StubAnswer()
        ),
        judge=lambda prompt: "LABEL: correct\nREASON: ok",
        citation_mode="highlight",
        metadata={"index": "File Collection"},
    )
    boot = mocker.patch.object(cli, "bootstrap", return_value=ctx)
    client_cls = mocker.patch.object(cli, "Client")
    mocker.patch.object(
        cli,
        "sync_dataset",
        return_value=SimpleNamespace(create=[1], update=[], delete=[], stale=3),
    )
    calls: dict = {}
    mocker.patch.object(
        cli, "evaluate", side_effect=fake_evaluate(calls, cli.load_dataset(path))
    )
    mocker.patch.object(cli, "_rows", side_effect=lambda results: results.rows)
    return SimpleNamespace(
        path=path, ctx=ctx, boot=boot, client_cls=client_cls, calls=calls
    )


def test_main_happy_path(setup, capsys):
    assert cli.main(["--dataset", str(setup.path)]) == 0
    out = capsys.readouterr().out
    assert "dataset akkad: 1 created, 0 updated, 0 deleted" in out
    assert "3 examples in LangSmith are not in the file" in out
    assert "experiment: exp-1" in out
    assert "correctness" in out and "1.00" in out and "retrieval_hit" in out
    assert setup.calls == {
        "data": setup.client_cls.return_value.list_examples.return_value,
        "prefix": "akkad-highlight",
        "max_concurrency": 2,
        "metadata": {"index": "File Collection"},
    }


def test_main_evaluates_only_the_files_examples(setup):
    from ktem.evals.dataset import example_id

    assert cli.main(["--dataset", str(setup.path)]) == 0
    kwargs = setup.client_cls.return_value.list_examples.call_args.kwargs
    assert kwargs == {
        "dataset_name": "akkad",
        "example_ids": [example_id("akkad", "Who founded Akkad?")],
    }


def test_main_requires_api_key(setup, monkeypatch, capsys):
    monkeypatch.delenv("LANGSMITH_API_KEY")
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    assert cli.main(["--dataset", str(setup.path)]) == 2
    assert "LANGSMITH_API_KEY" in capsys.readouterr().err
    setup.boot.assert_not_called()


def test_main_exports_env_before_key_check(setup, monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY")
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.setattr(
        cli,
        "export_langsmith_env",
        lambda *a, **k: monkeypatch.setenv("LANGSMITH_API_KEY", "from-dotenv"),
    )
    assert cli.main(["--dataset", str(setup.path)]) == 0


def test_main_bootstrap_error_stops_before_langsmith(setup, capsys):
    setup.boot.side_effect = BootstrapError(
        "user 'x' has no files in index 'File Collection'"
    )
    assert cli.main(["--dataset", str(setup.path), "--user", "x"]) == 2
    assert "has no files" in capsys.readouterr().err
    setup.client_cls.assert_not_called()


def test_summarize_skips_none():
    rows = [
        {"evaluation_results": {"results": [EvaluationResult(key="k", score=s)]}}
        for s in (1.0, None, 0.0)
    ]
    assert cli.summarize(rows) == {"k": (0.5, 2)}


def test_row_delay_sleeps_before_each_row(setup, mocker):
    sleeps = mocker.patch.object(cli.time, "sleep")
    assert cli.main(["--dataset", str(setup.path), "--row-delay", "15"]) == 0
    sleeps.assert_called_once_with(15.0)  # one row in the fixture dataset


def test_no_row_delay_by_default(setup, mocker):
    sleeps = mocker.patch.object(cli.time, "sleep")
    assert cli.main(["--dataset", str(setup.path)]) == 0
    sleeps.assert_not_called()
