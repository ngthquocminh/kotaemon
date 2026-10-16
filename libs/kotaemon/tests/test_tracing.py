import contextvars
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from kotaemon.base import Document
from kotaemon.utils.tracing import (
    export_langsmith_env,
    maybe_wrap_openai,
    reduce_answer,
    run_in_context,
    traceable,
)


def test_traceable_is_noop_without_langsmith(monkeypatch):
    monkeypatch.setitem(sys.modules, "langsmith", None)

    def f():
        return 1

    assert traceable(name="x")(f) is f


def test_maybe_wrap_openai_without_langsmith(monkeypatch):
    monkeypatch.setitem(sys.modules, "langsmith", None)
    monkeypatch.setitem(sys.modules, "langsmith.wrappers", None)
    client = object()
    assert maybe_wrap_openai(client) is client


def test_maybe_wrap_openai_patches_create():
    from openai import OpenAI

    client = maybe_wrap_openai(OpenAI(api_key="sk-test"))
    assert hasattr(client.chat.completions.create, "__wrapped__")


def test_run_in_context_carries_contextvar():
    var = contextvars.ContextVar("v", default="unset")
    var.set("set")
    seen = []
    t = threading.Thread(target=run_in_context(lambda: seen.append(var.get())))
    t.start()
    t.join()
    with ThreadPoolExecutor() as ex:
        ex.submit(run_in_context(lambda: seen.append(var.get()))).result()
    bare = threading.Thread(target=lambda: seen.append(var.get()))
    bare.start()
    bare.join()
    assert seen == ["set", "set", "unset"]


def test_traced_generator_keeps_return_and_threads_nest():
    from langsmith.run_helpers import get_current_run_tree, tracing_context

    seen = {}

    @traceable(name="child")
    def child(tag):
        rt = get_current_run_tree()
        seen[tag] = rt.parent_run_id if rt else None

    @traceable(name="root", reduce_fn=reduce_answer)
    def root():
        seen["root"] = get_current_run_tree().id
        t = threading.Thread(target=run_in_context(lambda: child("thread")))
        t.start()
        t.join()
        with ThreadPoolExecutor() as ex:
            ex.submit(run_in_context(child), "pool").result()
        yield Document(channel="chat", content="Hel")
        return Document(text="Hello.")

    def consume():
        return (yield from root())

    with tracing_context(enabled="local"):
        gen = consume()
        items = []
        try:
            while True:
                items.append(next(gen))
        except StopIteration as stop:
            final = stop.value
    assert [d.content for d in items] == ["Hel"]
    assert final.text == "Hello."
    assert seen["thread"] == seen["root"]
    assert seen["pool"] == seen["root"]


def test_reduce_answer():
    final = Document(text="Hello.")
    assert reduce_answer([Document(channel="chat", content="Hel"), final]) == {
        "answer": "Hello."
    }
    chunks = [
        Document(channel="chat", content="a"),
        Document(channel="chat", content=None),
        Document(channel="chat", content="b"),
        Document(channel="info", content="<div>"),
    ]
    assert reduce_answer(chunks) == {"answer": "b"}
    assert reduce_answer([]) == {"answer": ""}


def test_export_langsmith_env_reads_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "LANGSMITH_PROJECT=from-dotenv\nLANGSMITH_ENDPOINT=https://dotenv\nOTHER=x\n"
    )
    monkeypatch.setenv("LANGSMITH_PROJECT", "")  # empty counts as unset
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://already")
    monkeypatch.setenv("OTHER", "keep")
    export_langsmith_env(str(tmp_path))
    assert os.environ["LANGSMITH_PROJECT"] == "from-dotenv"
    assert os.environ["LANGSMITH_ENDPOINT"] == "https://already"
    assert os.environ["OTHER"] == "keep"


def test_export_langsmith_env_without_dotenv(tmp_path, monkeypatch):
    monkeypatch.setenv("LANGSMITH_PROJECT", "")
    export_langsmith_env(str(tmp_path))  # no .env file: no error
    assert os.environ["LANGSMITH_PROJECT"] == ""
