"""Run a dataset through the RAG pipeline and score it in LangSmith.

Run from the repository root (theflow reads ``./flowsettings.py``)::

    python -m ktem.evals.run --dataset path/to/questions.jsonl --user admin
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from langsmith import Client
from langsmith.evaluation import evaluate

from kotaemon.utils.tracing import export_langsmith_env

from .bootstrap import BootstrapError, bootstrap
from .dataset import DatasetError, load_dataset, sync_dataset
from .evaluators import build_evaluators
from .runner import make_target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ktem.evals.run",
        description="Evaluate the simple RAG pipeline against a LangSmith dataset.",
    )
    parser.add_argument("--dataset", required=True, help="CSV or JSONL file")
    parser.add_argument(
        "--dataset-name", help="LangSmith dataset name (default: file stem)"
    )
    parser.add_argument(
        "--user", help="whose files and settings to use (needed with user management)"
    )
    parser.add_argument(
        "--index", help="index name or id (default: the first file index)"
    )
    parser.add_argument(
        "--citation",
        choices=["highlight", "inline", "off"],
        help="citation mode (default: the user's setting)",
    )
    parser.add_argument(
        "--llm-scoring",
        action="store_true",
        help="keep the LLM relevance scorer on (one LLM call per retrieved doc)",
    )
    parser.add_argument("--experiment-prefix", help="LangSmith experiment prefix")
    parser.add_argument(
        "--max-concurrency", type=int, default=2, help="parallel rows (default 2)"
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="delete LangSmith examples that are no longer in the file",
    )
    return parser


def _rows(results) -> Iterable[Mapping]:
    return results


def summarize(rows: Iterable[Mapping]) -> dict[str, tuple[float | None, int]]:
    """Mean of non-None scores per metric, and how many rows were scored."""
    scores: dict[str, list[float]] = {}
    for row in rows:
        for result in row["evaluation_results"]["results"]:
            bucket = scores.setdefault(result.key, [])
            if result.score is not None:
                bucket.append(float(result.score))
    return {
        key: (sum(values) / len(values) if values else None, len(values))
        for key, values in scores.items()
    }


def _error(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    export_langsmith_env()
    if not (os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")):
        return _error("LANGSMITH_API_KEY is not set (env or .env)")

    dataset_name = args.dataset_name or Path(args.dataset).stem
    try:
        examples = load_dataset(args.dataset)
        ctx = bootstrap(
            username=args.user,
            index=args.index,
            citation=args.citation,
            llm_scoring=args.llm_scoring,
            dataset_name=dataset_name,
        )
    except (DatasetError, BootstrapError) as exc:
        return _error(str(exc))

    client = Client()
    plan = sync_dataset(client, dataset_name, examples, prune=args.prune)
    print(
        f"dataset {dataset_name}: {len(plan.create)} created, "
        f"{len(plan.update)} updated, {len(plan.delete)} deleted"
    )

    results = evaluate(
        make_target(ctx.build_pipeline, ctx.citation_mode),
        data=dataset_name,
        evaluators=build_evaluators(ctx.judge),
        experiment_prefix=args.experiment_prefix
        or f"{dataset_name}-{ctx.citation_mode}",
        max_concurrency=args.max_concurrency,
        metadata=ctx.metadata,
        client=client,
    )

    print(f"experiment: {getattr(results, 'experiment_name', '?')}")
    for key, (mean, count) in summarize(_rows(results)).items():
        shown = "-" if mean is None else f"{mean:.2f}"
        print(f"  {key:<18} {shown:>5}  ({count} scored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
