"""Build the app headless for one user and one index, as the chat page would."""

import subprocess
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from ktem.db.engine import engine
from ktem.db.models import Settings, User
from ktem.index.file.index import FileIndex
from sqlmodel import Session, select

from .runner import EvalQAPipeline

CITATION_KEY = "reasoning.options.simple.highlight_citation"


class BootstrapError(RuntimeError):
    """The requested user, index or files are not usable."""


@dataclass
class EvalContext:
    build_pipeline: Callable[[], EvalQAPipeline]
    judge: Callable[[str], str]
    citation_mode: str
    metadata: dict


def resolve_user_id(username: str | None, user_management: bool) -> str:
    if not user_management:
        return "default"
    if not username:
        raise BootstrapError("user management is on: pass --user USERNAME")
    with Session(engine) as session:
        user = session.exec(
            select(User).where(User.username_lower == username.lower())
        ).first()
    if user is None:
        raise BootstrapError(f"user {username!r} not found")
    return user.id


def load_user_settings(defaults: dict, user_id: str) -> dict:
    """App defaults overlaid with the user's saved settings, key by key."""
    with Session(engine) as session:
        row = session.exec(select(Settings).where(Settings.user == user_id)).first()
    saved = row.setting if row else {}
    return {**deepcopy(defaults), **deepcopy(saved)}


def apply_overrides(
    settings: dict, index_id: int, citation: str | None, llm_scoring: bool
) -> dict:
    out = dict(settings)
    if citation is not None:
        out[CITATION_KEY] = citation
    # the scorer only reorders docs for display; skip its per-doc LLM calls
    out[f"index.options.{index_id}.use_llm_reranking"] = llm_scoring
    return out


def select_index(indices: Sequence[Any], name_or_id: str | None) -> FileIndex:
    if name_or_id is None:
        for index in indices:
            if type(index) is FileIndex:
                return index
        raise BootstrapError("no plain file index found; pass --index")
    for index in indices:
        if index.name.lower() == name_or_id.lower() or str(index.id) == name_or_id:
            return index
    names = ", ".join(f"{index.name} (id {index.id})" for index in indices)
    raise BootstrapError(f"index {name_or_id!r} not found; available: {names}")


def git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return None


def bootstrap(
    username: str | None,
    index: str | None,
    citation: str | None,
    llm_scoring: bool,
    dataset_name: str,
) -> EvalContext:
    from ktem.app import BaseApp
    from ktem.llms.manager import llms

    app = BaseApp()
    user_id = resolve_user_id(username, app.f_user_management)
    idx = select_index(app.index_manager.indices, index)
    settings = apply_overrides(
        load_user_settings(app.default_settings.flatten(), user_id),
        idx.id,
        citation,
        llm_scoring,
    )

    selected = ["all", [], user_id]
    if not idx.get_selected_ids(selected):
        raise BootstrapError(
            f"user {username or user_id!r} has no files in index {idx.name!r}"
        )

    def build_pipeline() -> EvalQAPipeline:
        retrievers = idx.get_retriever_pipelines(settings, user_id, selected)
        state = {"app": {"regen": False}, "pipeline": {}}
        return EvalQAPipeline.get_pipeline(settings, state, retrievers)

    judge_llm = llms.get_default()

    def judge(prompt: str) -> str:
        return judge_llm(prompt).text

    prefix = f"index.options.{idx.id}."
    metadata = {
        "dataset": dataset_name,
        "user": username or user_id,
        "index": idx.name,
        "llm": settings.get("reasoning.options.simple.llm")
        or llms.get_default_name(),
        "judge_llm": llms.get_default_name(),
        "citation": settings[CITATION_KEY],
        "retrieval_mode": settings.get(prefix + "retrieval_mode"),
        "num_retrieval": settings.get(prefix + "num_retrieval"),
        "use_reranking": settings.get(prefix + "use_reranking"),
        "llm_scoring": llm_scoring,
        "git_sha": git_sha(),
    }
    return EvalContext(build_pipeline, judge, settings[CITATION_KEY], metadata)
