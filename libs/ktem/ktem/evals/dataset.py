"""Load eval datasets (CSV / JSONL) and sync them to LangSmith by stable id.

Must not import ``ktem.app`` or theflow settings.
"""

import csv
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

# Fixed forever: changing it changes every example id.
NAMESPACE = uuid.UUID("6f1c3c8e-7a52-4a3e-9a51-2d9c4f0b8e11")


class DatasetError(ValueError):
    """The dataset file is malformed."""


@dataclass(frozen=True)
class EvalExample:
    question: str
    reference_answer: str
    source_file: str | None = None
    page: str | None = None

    def inputs(self) -> dict:
        return {"question": self.question}

    def outputs(self) -> dict:
        out: dict[str, Any] = {"reference_answer": self.reference_answer}
        if self.source_file is not None:
            out["source_file"] = self.source_file
        if self.page is not None:
            out["page"] = self.page
        return out


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _read_rows(path: Path) -> list[tuple[int, dict]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            return list(enumerate(csv.DictReader(f), start=1))
    if suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8-sig") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append((line_no, json.loads(line)))
                except json.JSONDecodeError as exc:
                    raise DatasetError(f"line {line_no}: invalid JSON ({exc})")
        return rows
    raise DatasetError(f"{path}: dataset must be a .csv or .jsonl file")


def load_dataset(path: str | Path) -> list[EvalExample]:
    examples: list[EvalExample] = []
    seen: set[str] = set()
    for line_no, row in _read_rows(Path(path)):
        question = _clean(row.get("question"))
        if not question:
            raise DatasetError(f"line {line_no}: empty question")
        reference = _clean(row.get("reference_answer"))
        if not reference:
            raise DatasetError(f"line {line_no}: empty reference_answer")
        if question in seen:
            raise DatasetError(f"line {line_no}: duplicate question {question!r}")
        seen.add(question)
        examples.append(
            EvalExample(
                question=question,
                reference_answer=reference,
                source_file=_clean(row.get("source_file")),
                page=_clean(row.get("page")),
            )
        )
    return examples


def example_id(dataset_name: str, question: str) -> uuid.UUID:
    return uuid.uuid5(NAMESPACE, dataset_name + "\n" + question.strip())


@dataclass
class SyncPlan:
    create: list[tuple[uuid.UUID, EvalExample]] = field(default_factory=list)
    update: list[tuple[uuid.UUID, EvalExample]] = field(default_factory=list)
    delete: list[uuid.UUID] = field(default_factory=list)
    # in LangSmith but not in the file, kept because --prune was not given
    stale: int = 0


def plan_sync(
    dataset_name: str,
    examples: Sequence[EvalExample],
    existing: Iterable[Any],
    prune: bool,
) -> SyncPlan:
    """Diff the file against LangSmith (0.1.147 has no example upsert)."""
    current = {str(e.id): e for e in existing}
    plan = SyncPlan()
    wanted: set[str] = set()
    for ex in examples:
        ex_id = example_id(dataset_name, ex.question)
        wanted.add(str(ex_id))
        old = current.get(str(ex_id))
        if old is None:
            plan.create.append((ex_id, ex))
        elif (old.inputs or {}) != ex.inputs() or (old.outputs or {}) != ex.outputs():
            plan.update.append((ex_id, ex))
    missing = [uuid.UUID(key) for key in current if key not in wanted]
    if prune:
        plan.delete = missing
    else:
        plan.stale = len(missing)
    return plan


def sync_dataset(
    client: Any,
    dataset_name: str,
    examples: Sequence[EvalExample],
    prune: bool = False,
) -> SyncPlan:
    if client.has_dataset(dataset_name=dataset_name):
        dataset = client.read_dataset(dataset_name=dataset_name)
        existing = list(client.list_examples(dataset_id=dataset.id))
    else:
        dataset = client.create_dataset(
            dataset_name, description="kotaemon eval dataset"
        )
        existing = []

    plan = plan_sync(dataset_name, examples, existing, prune)
    if plan.create:
        client.create_examples(
            inputs=[ex.inputs() for _, ex in plan.create],
            outputs=[ex.outputs() for _, ex in plan.create],
            ids=[ex_id for ex_id, _ in plan.create],
            dataset_id=dataset.id,
        )
    if plan.update:
        client.update_examples(
            example_ids=[ex_id for ex_id, _ in plan.update],
            inputs=[ex.inputs() for _, ex in plan.update],
            outputs=[ex.outputs() for _, ex in plan.update],
        )
    for ex_id in plan.delete:
        client.delete_example(ex_id)
    return plan
