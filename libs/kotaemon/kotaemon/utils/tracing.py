"""Optional LangSmith tracing helpers.

Tracing stays off unless LangSmith is configured (``LANGSMITH_TRACING=true``) or
a caller enables it, as ``langsmith.evaluation.evaluate`` does. Every helper
degrades to a no-op when ``langsmith`` cannot be imported.
"""

import contextvars
import os
from typing import Any, Callable, Sequence, TypeVar

F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")
R = TypeVar("R")

LANGSMITH_ENV_VARS = (
    "LANGSMITH_API_KEY",
    "LANGSMITH_TRACING",
    "LANGSMITH_PROJECT",
    "LANGSMITH_ENDPOINT",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_PROJECT",
    "LANGCHAIN_ENDPOINT",
)


def traceable(**kwargs) -> Callable[[F], F]:
    """Return ``langsmith.traceable(**kwargs)``, or an identity decorator."""
    try:
        from langsmith import traceable as ls_traceable
    except ImportError:
        return lambda func: func
    return ls_traceable(**kwargs)


def maybe_wrap_openai(client: T) -> T:
    """Wrap an OpenAI client so its calls are traced (pass-through when off)."""
    try:
        from langsmith.wrappers import wrap_openai
    except ImportError:
        return client
    return wrap_openai(client)


def run_in_context(fn: Callable[..., R]) -> Callable[..., R]:
    """Bind ``fn`` to a copy of the current context, for use in another thread.

    Call it once per thread or per ``submit``: one context cannot be entered by
    two threads at the same time.
    """
    ctx = contextvars.copy_context()
    return lambda *args, **kwargs: ctx.run(fn, *args, **kwargs)


def reduce_answer(items: Sequence[Any]) -> dict[str, str]:
    """``reduce_fn`` for streamed pipelines: keep only the final answer text.

    langsmith appends the generator's return value (the answer ``Document``,
    which has no channel) to the collected items. If the generator raised before
    returning, fall back to the chat text as the UI would show it.
    """
    if (
        items
        and getattr(items[-1], "channel", None) is None
        and hasattr(items[-1], "text")
    ):
        return {"answer": items[-1].text}
    text = ""
    for item in items:
        if getattr(item, "channel", None) == "chat":
            text = "" if item.content is None else text + str(item.content)
    return {"answer": text}


def export_langsmith_env(search_path: str | None = None) -> None:
    """Copy LangSmith variables from ``.env`` into ``os.environ`` when unset.

    kotaemon reads ``.env`` through python-decouple, which never populates
    ``os.environ``; langsmith only reads ``os.environ``.
    """
    from dotenv import dotenv_values

    directory = os.path.abspath(search_path or os.getcwd())
    while not os.path.isfile(os.path.join(directory, ".env")):
        parent = os.path.dirname(directory)
        if parent == directory:
            return
        directory = parent

    values = dotenv_values(os.path.join(directory, ".env"))
    for name in LANGSMITH_ENV_VARS:
        if not os.environ.get(name) and values.get(name):
            os.environ[name] = values[name]
