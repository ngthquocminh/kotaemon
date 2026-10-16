# LangSmith Tracing and Offline Evals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in LangSmith tracing to kotaemon's `simple` RAG pipeline, plus a CLI (`python -m ktem.evals.run`) that runs a question/reference dataset through it and records scored experiments in LangSmith.

**Architecture:** A small tracing shim in `kotaemon.utils.tracing` provides `@traceable`, an OpenAI client wrapper, a context-carrying thread helper and a `.env` exporter. It is applied at fixed seams: the QA pipeline, the retriever, the answer pipelines, the OpenAI client factories, and five thread sites. A new `ktem.evals` package does the rest. It loads and syncs the dataset, builds the app headless for one user and one index, runs each question through a subclass of `FullQAPipeline` that captures the retrieved docs, and scores the outputs with two LLM judges and three deterministic evaluators through `langsmith.evaluation.evaluate`.

**Tech Stack:** Python 3.10, langsmith 0.1.147 (pinned below 0.2 by langchain-core 0.2.x), theflow, openai SDK, SQLModel / SQLAlchemy, pytest + pytest-mock.

**Spec:** `docs/superpowers/specs/2026-10-16-langsmith-evals-design.md` (v3). Read it before starting. Every "why" lives there.

## Global Constraints

- Target the installed `langsmith` 0.1.147 API. Declare `langsmith>=0.1.147,<0.2` in `ktem`. Do not upgrade LangChain or LangSmith.
- Evaluation entry point: `from langsmith.evaluation import evaluate` (0.1.147 has no top-level `langsmith.evaluate`).
- Tracing is opt-in. With no `LANGSMITH_*` variables set, app behaviour must not change.
- `ktem/evals/dataset.py`, `cleaning.py` and `evaluators.py` must not import `ktem.app` or theflow settings.
- v1 evaluates only the `simple` reasoning pipeline (`FullQAPipeline`).
- Example id: `uuid5(NAMESPACE, dataset_name + "\n" + question.strip())`.
- `--max-concurrency` defaults to 2 and is always passed to `evaluate` explicitly, never `None`.
- Judge labels and scores: `correct` / `partially_correct` / `incorrect` → 1 / 0.5 / 0. `grounded` / `partially_grounded` / `ungrounded` → 1 / 0.5 / 0.
- Span names: root `kotaemon.qa` (run_type `chain`), `retrieve` (run_type `retriever`), `answer_with_context` and `answer_with_inline_citation` (run_type `chain`).
- Use `.venv/Scripts/python.exe` on Windows, because the system `python` lacks the deps. Prefix with `PYTHONIOENCODING=utf-8` when output contains non-ASCII.
- kotaemon tests: `cd libs/kotaemon && ../../.venv/Scripts/python.exe -m pytest <path>`.
- ktem tests run **from the repo root** (theflow reads `./flowsettings.py`): `.venv/Scripts/python.exe -m pytest libs/ktem/ktem_tests --ignore=libs/ktem/ktem_tests/test_qa.py`.
- Never run `uv sync` locally: it deletes packages that are not in the lockfile from the dev venv. `uv lock` is fine.
- Work on branch `feat/langsmith-evals`. End every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Before Task 1, commit the revised spec, this plan and the `.gitignore` change (`eval_data/`) as one `docs:` commit.

## Review Focus

1. **Re-running the same dataset file after editing one reference answer** must update that one LangSmith example, not add a duplicate or touch the others. Owned by Task 5: `test_plan_sync_updates_only_changed_outputs`.
2. **The LangSmith key is only in `.env`** (the normal kotaemon setup). The CLI must still find it, and the live app must still trace. Owned by Task 1 (`test_export_langsmith_env_*`) and Task 9 (`test_main_exports_env_before_key_check`).
3. **Vector search returns a "Page thumbnail" doc for the right file and page** but no real chunk. This must not count as a retrieval hit. Owned by Task 6: `test_retrieval_ignores_thumbnails`.
4. **An inline-citation answer whose last citation entry lacks an end phrase** (common with small models) must not crash the row. It counts as an unmatched claim. Owned by Task 7: `test_find_start_end_phrase_accepts_none` and `test_citation_facts_inline_missing_phrase`.
5. **One row fails mid-run** (LLM timeout, provider error). That row's evaluators must return `None`, the run must continue, and the summary means must exclude it rather than count it as 0. Owned by Task 6 (`test_*_missing_outputs_is_none`) and Task 9 (`test_summarize_skips_none`).

---

## File Structure

| File | Responsibility |
|---|---|
| `libs/kotaemon/kotaemon/utils/__init__.py` | new empty package |
| `libs/kotaemon/kotaemon/utils/tracing.py` | shim: `traceable`, `maybe_wrap_openai`, `run_in_context`, `reduce_answer`, `export_langsmith_env` |
| `libs/kotaemon/kotaemon/llms/chats/openai.py` | wrap clients in `ChatOpenAI.prepare_client` / `AzureChatOpenAI.prepare_client` |
| `libs/kotaemon/kotaemon/indices/qa/citation_qa.py` | span on `AnswerWithContextPipeline.stream`; context for citation and mindmap threads |
| `libs/kotaemon/kotaemon/indices/qa/citation_qa_inline.py` | span on `AnswerWithInlineCitation.stream`; context for mindmap thread |
| `libs/kotaemon/kotaemon/indices/rankings/llm_trulens.py` | context for pool submits |
| `libs/kotaemon/kotaemon/indices/qa/utils.py` | `find_start_end_phrase` accepts `None` phrases |
| `libs/ktem/ktem/reasoning/simple.py` | root span on `FullQAPipeline.stream`, retriever span on `retrieve`, context for scorer thread |
| `libs/ktem/ktem/app.py` | call `export_langsmith_env()` in `BaseApp.__init__` |
| `libs/ktem/ktem/index/file/index.py`, `ui.py`, `graph/graph_index.py`, `graph/light_graph_index.py`, `graph/nano_graph_index.py` | headless `FileIndex.get_selected_ids` |
| `libs/ktem/ktem/evals/{__init__,dataset,cleaning,evaluators,runner,bootstrap,run}.py` | eval package |
| `libs/ktem/ktem_tests/eval_stubs.py` | stub retriever / answer pipeline shared by ktem tests |
| `libs/ktem/pyproject.toml`, `uv.lock`, `.env.example`, `.github/workflows/unit-test.yaml` | wiring |

---

### Task 1: Tracing shim

**Files:**
- Create: `libs/kotaemon/kotaemon/utils/__init__.py` (empty), `libs/kotaemon/kotaemon/utils/tracing.py`
- Test: `libs/kotaemon/tests/test_tracing.py`

**Interfaces:**
- Produces:
  - `traceable(**kwargs) -> Callable[[F], F]`: decorator factory, always called with parentheses.
  - `maybe_wrap_openai(client: T) -> T`
  - `run_in_context(fn: Callable[..., R]) -> Callable[..., R]`
  - `reduce_answer(items: Sequence[Any]) -> dict[str, str]`, returning `{"answer": str}`
  - `export_langsmith_env(search_path: str | None = None) -> None`
  - `LANGSMITH_ENV_VARS: tuple[str, ...]`

- [ ] **Step 1: Write the failing tests**

```python
import contextvars
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from kotaemon.base import Document
from kotaemon.utils import tracing
from kotaemon.utils.tracing import (
    export_langsmith_env, maybe_wrap_openai, reduce_answer, run_in_context, traceable,
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
    t.start(); t.join()
    with ThreadPoolExecutor() as ex:
        ex.submit(run_in_context(lambda: seen.append(var.get()))).result()
    bare = threading.Thread(target=lambda: seen.append(var.get()))
    bare.start(); bare.join()
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
        t.start(); t.join()
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
    assert reduce_answer([Document(channel="chat", content="Hel"), final]) == {"answer": "Hello."}
    chunks = [Document(channel="chat", content="a"), Document(channel="chat", content=None),
              Document(channel="chat", content="b"), Document(channel="info", content="<div>")]
    assert reduce_answer(chunks) == {"answer": "b"}
    assert reduce_answer([]) == {"answer": ""}


def test_export_langsmith_env_reads_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "LANGSMITH_PROJECT=from-dotenv\nLANGSMITH_ENDPOINT=https://dotenv\nOTHER=x\n"
    )
    monkeypatch.setenv("LANGSMITH_PROJECT", "")          # empty counts as unset
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://already")
    monkeypatch.setenv("OTHER", "keep")
    export_langsmith_env(str(tmp_path))
    assert os.environ["LANGSMITH_PROJECT"] == "from-dotenv"
    assert os.environ["LANGSMITH_ENDPOINT"] == "https://already"
    assert os.environ["OTHER"] == "keep"


def test_export_langsmith_env_without_dotenv(tmp_path, monkeypatch):
    monkeypatch.setenv("LANGSMITH_PROJECT", "")
    export_langsmith_env(str(tmp_path))                  # no .env file: no error
    assert os.environ["LANGSMITH_PROJECT"] == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd libs/kotaemon && ../../.venv/Scripts/python.exe -m pytest tests/test_tracing.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'kotaemon.utils'`

- [ ] **Step 3: Implement `libs/kotaemon/kotaemon/utils/tracing.py`**

- `traceable` and `maybe_wrap_openai` import `langsmith` / `langsmith.wrappers` **inside** the function and fall back to identity on `ImportError`.
- `run_in_context` takes `contextvars.copy_context()` at wrap time and returns `lambda *a, **k: ctx.run(fn, *a, **k)`.
- `LANGSMITH_ENV_VARS = ("LANGSMITH_API_KEY", "LANGSMITH_TRACING", "LANGSMITH_PROJECT", "LANGSMITH_ENDPOINT", "LANGCHAIN_API_KEY", "LANGCHAIN_TRACING_V2", "LANGCHAIN_PROJECT", "LANGCHAIN_ENDPOINT")`.
- `export_langsmith_env`: `decouple.AutoConfig(search_path=search_path or os.getcwd())`. For each name whose `os.environ.get(name)` is falsy, set it from `config(name, default="")` when that is non-empty.
- `reduce_answer`:

```python
def reduce_answer(items):
    if items and getattr(items[-1], "channel", None) is None and hasattr(items[-1], "text"):
        return {"answer": items[-1].text}
    text = ""
    for item in items:
        if getattr(item, "channel", None) == "chat":
            text = "" if item.content is None else text + str(item.content)
    return {"answer": text}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd libs/kotaemon && ../../.venv/Scripts/python.exe -m pytest tests/test_tracing.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add libs/kotaemon/kotaemon/utils libs/kotaemon/tests/test_tracing.py
git commit -m "feat(kotaemon): add optional LangSmith tracing shim"
```

---

### Task 2: kotaemon trace seams

**Files:**
- Modify: `libs/kotaemon/kotaemon/llms/chats/openai.py:285-305` (`ChatOpenAI.prepare_client`), `:444-466` (`AzureChatOpenAI.prepare_client`)
- Modify: `libs/kotaemon/kotaemon/indices/qa/citation_qa.py:190` (`AnswerWithContextPipeline.stream`), threads at `:224` and `:228`
- Modify: `libs/kotaemon/kotaemon/indices/qa/citation_qa_inline.py:193` (`AnswerWithInlineCitation.stream`), thread at `:224`
- Modify: `libs/kotaemon/kotaemon/indices/rankings/llm_trulens.py:150` (`executor.submit(llm_call)`)
- Test: `libs/kotaemon/tests/test_tracing_seams.py`

**Interfaces:**
- Consumes: Task 1 `traceable`, `maybe_wrap_openai`, `run_in_context`, `reduce_answer`.
- Produces: span names `answer_with_context` and `answer_with_inline_citation`. Thread work now runs in the caller's context.

- [ ] **Step 1: Write the failing tests**

```python
from langsmith.run_helpers import get_current_run_tree, tracing_context

from kotaemon.base import BaseComponent
from kotaemon.base.schema import LLMInterface
from kotaemon.indices.qa.citation_qa import AnswerWithContextPipeline
from kotaemon.indices.qa.citation_qa_inline import AnswerWithInlineCitation
from kotaemon.llms import AzureChatOpenAI, ChatLLM, ChatOpenAI

SEEN: dict = {}


class StubLLM(ChatLLM):
    def invoke(self, messages, **kwargs):
        return LLMInterface(content="FINAL ANSWER\nSargon of Akkad.")

    def stream(self, messages, **kwargs):
        yield LLMInterface(content="FINAL ANSWER\nSargon of Akkad.")


class StubCitation(BaseComponent):
    def run(self, context, question):
        SEEN["citation"] = get_current_run_tree()
        return None


class StubMindmap(BaseComponent):
    def run(self, context, question):
        SEEN["mindmap"] = get_current_run_tree()
        return None


def _drain(gen):
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        return stop.value


def test_openai_clients_are_wrapped():
    llm = ChatOpenAI(api_key="sk-test", model="gpt-4o-mini")
    assert hasattr(llm.prepare_client().chat.completions.create, "__wrapped__")
    assert hasattr(llm.prepare_client(async_version=True).chat.completions.create, "__wrapped__")
    azure = AzureChatOpenAI(azure_endpoint="https://x.openai.azure.com", api_key="k",
                            api_version="2024-02-01", azure_deployment="d")
    assert hasattr(azure.prepare_client().chat.completions.create, "__wrapped__")


def test_citation_and_mindmap_threads_nest_under_answer_span():
    SEEN.clear()
    pipe = AnswerWithContextPipeline(llm=StubLLM(), citation_pipeline=StubCitation(),
                                     create_mindmap_pipeline=StubMindmap(),
                                     enable_citation=True, enable_mindmap=True)
    with tracing_context(enabled="local"):
        answer = _drain(pipe.stream(question="who?", evidence="Sargon of Akkad founded it."))
    assert answer.text == "FINAL ANSWER\nSargon of Akkad."
    assert SEEN["citation"] is not None and SEEN["citation"].name == "answer_with_context"
    assert SEEN["mindmap"] is not None and SEEN["mindmap"].name == "answer_with_context"
    assert SEEN["citation"].outputs == {"answer": "FINAL ANSWER\nSargon of Akkad."}


def test_inline_mindmap_thread_nests_under_inline_span():
    SEEN.clear()
    pipe = AnswerWithInlineCitation(llm=StubLLM(), create_mindmap_pipeline=StubMindmap(),
                                    enable_mindmap=True)
    with tracing_context(enabled="local"):
        _drain(pipe.stream(question="who?", evidence="Sargon of Akkad founded it."))
    assert SEEN["mindmap"].name == "answer_with_inline_citation"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd libs/kotaemon && ../../.venv/Scripts/python.exe -m pytest tests/test_tracing_seams.py -v`
Expected: 3 FAIL. The clients have no `__wrapped__`, and `SEEN["citation"]` is `None`.

- [ ] **Step 3: Apply the seams**

- In both `prepare_client` methods, return `maybe_wrap_openai(<client>)` for both the sync and async branches.
- Decorate `AnswerWithContextPipeline.stream` with `@traceable(name="answer_with_context", run_type="chain", reduce_fn=reduce_answer)`.
- Decorate `AnswerWithInlineCitation.stream` with `@traceable(name="answer_with_inline_citation", run_type="chain", reduce_fn=reduce_answer)`.
- At the three thread sites in these two files, change `threading.Thread(target=X)` to `threading.Thread(target=run_in_context(X))`.
- In `llm_trulens.py`, change `executor.submit(llm_call)` to `executor.submit(run_in_context(llm_call))`.

- [ ] **Step 4: Run tests to verify they pass, then the existing suites for these modules**

Run: `cd libs/kotaemon && ../../.venv/Scripts/python.exe -m pytest tests/test_tracing_seams.py tests/test_llms_chat_models.py tests/test_reranking.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add libs/kotaemon/kotaemon/llms/chats/openai.py libs/kotaemon/kotaemon/indices libs/kotaemon/tests/test_tracing_seams.py
git commit -m "feat(kotaemon): trace answer pipelines, OpenAI clients and worker threads"
```

---

### Task 3: ktem trace seams

**Files:**
- Modify: `libs/ktem/ktem/reasoning/simple.py`: `FullQAPipeline.retrieve` (`:108`), `FullQAPipeline.stream` (`:281`), scorer thread (`:303`)
- Modify: `libs/ktem/ktem/app.py`: `BaseApp.__init__`, first statement
- Create: `libs/ktem/ktem_tests/eval_stubs.py`
- Test: `libs/ktem/ktem_tests/test_tracing_qa.py`

**Interfaces:**
- Consumes: Task 1 shim.
- Produces:
  - `eval_stubs.StubRetriever`, `eval_stubs.StubAnswer(answer_text: str, citation: Any)`, `eval_stubs.SEEN`, `eval_stubs.DOC_TEXT`, `eval_stubs.ANSWER_TEXT`, `eval_stubs.drain(gen) -> return value`. Reused by Tasks 7 and 9.
  - Span names `kotaemon.qa` and `retrieve`.

- [ ] **Step 1: Create the shared stubs `libs/ktem/ktem_tests/eval_stubs.py`**

```python
"""Stub components for tracing and eval tests: no LLM, no index, no network."""
from typing import Any

from langsmith.run_helpers import get_current_run_tree

from kotaemon.base import BaseComponent, Document, RetrievedDocument

DOC_TEXT = "Sargon of Akkad founded the empire. It ruled Mesopotamia from Akkad."
ANSWER_TEXT = "Sargon of Akkad."
SEEN: dict = {}


def drain(gen):
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        return stop.value


class StubRetriever(BaseComponent):
    def run(self, text, **kwargs):
        SEEN["retriever"] = get_current_run_tree()
        return [RetrievedDocument(text=DOC_TEXT, score=0.9, metadata={
            "file_name": "Akkadian Empire.pdf", "page_label": "1"})]

    def generate_relevant_scores(self, query, documents):
        SEEN["scorer"] = get_current_run_tree()
        return documents


class StubAnswer(BaseComponent):
    answer_text: str = ANSWER_TEXT
    citation: Any = None

    def stream(self, question, evidence, **kwargs):
        SEEN["answer"] = get_current_run_tree()
        yield Document(channel="chat", content=self.answer_text)
        return Document(text=self.answer_text, metadata={
            "citation": self.citation, "mindmap": None, "citation_viz": False, "qa_score": None})

    def prepare_citations(self, answer, docs):
        return [], []

    def run(self, *args, **kwargs):
        raise NotImplementedError
```

- [ ] **Step 2: Write the failing test `libs/ktem/ktem_tests/test_tracing_qa.py`**

```python
from langsmith.run_helpers import tracing_context

from ktem.reasoning.simple import FullQAPipeline
from ktem_tests.eval_stubs import ANSWER_TEXT, DOC_TEXT, SEEN, StubAnswer, StubRetriever, drain


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
    assert set(retrieve.outputs) == {"documents"}        # HTML infos dropped
    assert SEEN["scorer"].id == root.id                   # scorer thread kept the context


def test_qa_pipeline_untraced_still_answers():
    pipe = FullQAPipeline(retrievers=[StubRetriever()], answering_pipeline=StubAnswer())
    assert drain(pipe.stream("q", conv_id="c2", history=[])).text == ANSWER_TEXT
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest libs/ktem/ktem_tests/test_tracing_qa.py -v`
Expected: `test_qa_pipeline_spans_nest` FAILS (`SEEN["answer"]` is `None`); the untraced test passes.

- [ ] **Step 4: Apply the seams in `simple.py` and `app.py`**

- `FullQAPipeline.stream`: `@traceable(name="kotaemon.qa", run_type="chain", reduce_fn=reduce_answer)`.
- `FullQAPipeline.retrieve`: `@traceable(name="retrieve", run_type="retriever", process_outputs=_retrieve_outputs)`. Add a module-level `_retrieve_outputs(outputs) -> dict` that maps `outputs[0]` (docs) to `{"documents": [{"page_content": d.text, "metadata": {"file_name": ..., "page_label": ..., "doc_id": d.doc_id, "score": getattr(d, "score", None)}}]}`.
- Scorer thread: `threading.Thread(target=run_in_context(generate_relevant_scores))`.
- `BaseApp.__init__`: call `export_langsmith_env()` (from `kotaemon.utils.tracing`) first, so `.env` LangSmith settings reach langsmith before any span is created.

- [ ] **Step 5: Run the ktem suite**

Run: `.venv/Scripts/python.exe -m pytest libs/ktem/ktem_tests --ignore=libs/ktem/ktem_tests/test_qa.py -v`
Expected: 29 passed (27 existing + 2 new)

- [ ] **Step 6: Commit**

```bash
git add libs/ktem/ktem/reasoning/simple.py libs/ktem/ktem/app.py libs/ktem/ktem_tests/eval_stubs.py libs/ktem/ktem_tests/test_tracing_qa.py
git commit -m "feat(ktem): trace the simple QA pipeline and export LangSmith env from .env"
```

---

### Task 4: Headless file selection

**Files:**
- Modify: `libs/ktem/ktem/index/file/index.py`: add `get_selected_ids`; `:473` uses it
- Modify: `libs/ktem/ktem/index/file/ui.py:1800-1821`: `FileSelector.get_selected_ids` delegates
- Modify: `libs/ktem/ktem/index/file/graph/graph_index.py:28`, `light_graph_index.py:64`, `nano_graph_index.py:64`
- Test: `libs/ktem/ktem_tests/test_file_selection.py`

**Interfaces:**
- Produces: `FileIndex.get_selected_ids(self, selected: Sequence[Any] | None) -> list[str]`, where `selected` is `[mode, file_ids, user_id]`. It holds the logic moved from `ui.py:1800-1821` unchanged, plus: `None` or fewer than 3 items returns `[]`.

- [ ] **Step 1: Write the failing tests**

```python
from types import SimpleNamespace

import pytest
from sqlalchemy import Column, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session

from ktem.index.file.index import FileIndex

Base = declarative_base()


class Source(Base):
    __tablename__ = "index__99__source"
    id = Column(String, primary_key=True)
    user = Column(String)


@pytest.fixture
def make_index(mocker):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        s.add_all([Source(id="f1", user="alice"), Source(id="f2", user="bob"),
                   Source(id="f3", user="alice")])
        s.commit()
    mocker.patch("ktem.index.file.index.engine", engine)

    def _make(private=True):
        idx = FileIndex(app=None, id=99, name="t", config={"private": private})
        idx._resources = {"Source": Source, "Index": None}
        return idx
    return _make


def test_all_private_filters_by_user(make_index):
    assert sorted(make_index().get_selected_ids(["all", [], "alice"])) == ["f1", "f3"]

def test_all_public_returns_every_file(make_index):
    assert sorted(make_index(private=False).get_selected_ids(["all", [], "alice"])) == ["f1", "f2", "f3"]

def test_select_returns_given_ids(make_index):
    assert make_index().get_selected_ids(["select", ["f2"], "alice"]) == ["f2"]

@pytest.mark.parametrize("selected", [["disabled", ["f1"], "alice"], ["all", [], None], None, []])
def test_empty_cases(make_index, selected):
    assert make_index().get_selected_ids(selected) == []

def test_file_selector_delegates():
    from ktem.index.file.ui import FileSelector
    sel = FileSelector.__new__(FileSelector)
    sel._index = SimpleNamespace(get_selected_ids=lambda s: ["x"] if s == ["all", [], "u"] else [])
    assert sel.get_selected_ids(["all", [], "u"]) == ["x"]

def test_get_retriever_pipelines_without_ui(make_index):
    seen = {}

    class StubRetrieverCls:
        @classmethod
        def get_pipeline(cls, settings, config, selected_ids):
            seen["ids"] = sorted(selected_ids)
            return SimpleNamespace()

    idx = make_index()
    idx._retriever_pipeline_cls = [StubRetrieverCls]
    idx._vs = idx._docstore = idx._fs_path = None
    assert idx._selector_ui is None
    assert len(idx.get_retriever_pipelines({}, "alice", ["all", [], "alice"])) == 1
    assert seen["ids"] == ["f1", "f3"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest libs/ktem/ktem_tests/test_file_selection.py -v`
Expected: FAIL with `AttributeError: 'FileIndex' object has no attribute 'get_selected_ids'`

- [ ] **Step 3: Move the logic**

- Add `FileIndex.get_selected_ids` with the body of `ui.py:1800-1821` plus the `None`/short guard. `index.py` needs `from sqlalchemy import select` and `from sqlalchemy.orm import Session`; `engine` is already imported.
- Reduce `FileSelector.get_selected_ids(self, components)` to `return self._index.get_selected_ids(components)`.
- Change all four `self._selector_ui.get_selected_ids(selected)` calls to `self.get_selected_ids(selected)`.

- [ ] **Step 4: Run tests and check no UI call remains**

Run: `.venv/Scripts/python.exe -m pytest libs/ktem/ktem_tests --ignore=libs/ktem/ktem_tests/test_qa.py -v`, then `git grep -n "_selector_ui.get_selected_ids" libs/ktem`
Expected: all pass; the grep prints nothing

- [ ] **Step 5: Commit**

```bash
git add libs/ktem/ktem/index/file libs/ktem/ktem_tests/test_file_selection.py
git commit -m "refactor(ktem): resolve selected file ids on the index, not the UI"
```

---

### Task 5: Dataset loading and LangSmith sync

**Files:**
- Create: `libs/ktem/ktem/evals/__init__.py` (empty), `libs/ktem/ktem/evals/dataset.py`
- Test: `libs/ktem/ktem_tests/test_eval_dataset.py`

**Interfaces:**
- Produces:
  - `NAMESPACE: uuid.UUID = uuid.UUID("6f1c3c8e-7a52-4a3e-9a51-2d9c4f0b8e11")`. Fixed; never change it, or every id changes.
  - `class DatasetError(ValueError)`
  - `@dataclass(frozen=True) class EvalExample(question: str, reference_answer: str, source_file: str | None = None, page: str | None = None)`, with
    - `.inputs() -> {"question"}`
    - `.outputs() -> {"reference_answer", ["source_file"], ["page"]}`, where optional keys appear only when not `None`
  - `load_dataset(path: str | Path) -> list[EvalExample]`
  - `example_id(dataset_name: str, question: str) -> uuid.UUID`
  - `@dataclass class SyncPlan(create: list[tuple[uuid.UUID, EvalExample]], update: list[tuple[uuid.UUID, EvalExample]], delete: list[uuid.UUID])`
  - `plan_sync(dataset_name, examples, existing, prune: bool) -> SyncPlan`. `existing` items have `.id`, `.inputs`, `.outputs`.
  - `sync_dataset(client, dataset_name, examples, prune: bool = False) -> SyncPlan`

- [ ] **Step 1: Write the failing tests**

```python
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ktem.evals.dataset import (
    DatasetError, EvalExample, example_id, load_dataset, plan_sync, sync_dataset,
)

JSONL = (
    '{"question": "Who founded Akkad?", "reference_answer": "Sargon.", "source_file": "Akkadian Empire.pdf", "page": 1}\n'
    "\n"
    '{"question": "Exact speed of light?", "reference_answer": "299,792,458 m/s"}\n'
)


def test_load_jsonl(tmp_path):
    p = tmp_path / "d.jsonl"; p.write_text(JSONL, encoding="utf-8")
    rows = load_dataset(p)
    assert rows[0] == EvalExample("Who founded Akkad?", "Sargon.", "Akkadian Empire.pdf", "1")
    assert rows[1].source_file is None and rows[1].page is None
    assert rows[1].outputs() == {"reference_answer": "299,792,458 m/s"}


def test_load_csv_with_bom_and_blank_optional(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("question,reference_answer,source_file,page\nQ1,A1,f.pdf,\nQ2,A2,,\n",
                 encoding="utf-8-sig")
    rows = load_dataset(p)
    assert rows[0] == EvalExample("Q1", "A1", "f.pdf", None)
    assert rows[1] == EvalExample("Q2", "A2", None, None)


@pytest.mark.parametrize("body, msg", [
    ('{"question": " ", "reference_answer": "a"}\n', "line 1: empty question"),
    ('{"question": "q", "reference_answer": ""}\n', "line 1: empty reference_answer"),
    ('{"question": "q", "reference_answer": "a"}\n{"question": " q ", "reference_answer": "b"}\n',
     "line 2: duplicate question"),
])
def test_load_rejects_bad_rows(tmp_path, body, msg):
    p = tmp_path / "d.jsonl"; p.write_text(body, encoding="utf-8")
    with pytest.raises(DatasetError, match=msg):
        load_dataset(p)


def test_load_rejects_unknown_suffix(tmp_path):
    p = tmp_path / "d.txt"; p.write_text("x")
    with pytest.raises(DatasetError, match=r"\.csv or \.jsonl"):
        load_dataset(p)


def test_example_id_is_deterministic():
    assert example_id("ds", " Who? ") == example_id("ds", "Who?")
    assert example_id("ds", "Who?") != example_id("other", "Who?")
    assert isinstance(example_id("ds", "Who?"), uuid.UUID)


def _existing(ds, ex, outputs=None):
    return SimpleNamespace(id=example_id(ds, ex.question), inputs=ex.inputs(),
                           outputs=outputs if outputs is not None else ex.outputs())


def test_plan_sync_updates_only_changed_outputs():
    a, b = EvalExample("A?", "a"), EvalExample("B?", "b")
    edited_b = EvalExample("B?", "b, edited")
    c = EvalExample("C?", "c")
    plan = plan_sync("ds", [a, edited_b, c], [_existing("ds", a), _existing("ds", b)], prune=False)
    assert [ex for _, ex in plan.create] == [c]
    assert plan.update == [(example_id("ds", "B?"), edited_b)]
    assert plan.delete == []


def test_plan_sync_prune_deletes_missing_only_when_asked():
    a, gone = EvalExample("A?", "a"), EvalExample("Gone?", "g")
    existing = [_existing("ds", a), _existing("ds", gone)]
    assert plan_sync("ds", [a], existing, prune=False).delete == []
    assert plan_sync("ds", [a], existing, prune=True).delete == [example_id("ds", "Gone?")]


def test_sync_dataset_creates_dataset_and_examples():
    client = MagicMock()
    client.has_dataset.return_value = False
    client.create_dataset.return_value = SimpleNamespace(id="ds-id")
    ex = EvalExample("A?", "a", "f.pdf", "2")
    sync_dataset(client, "ds", [ex])
    client.create_dataset.assert_called_once()
    kwargs = client.create_examples.call_args.kwargs
    assert kwargs["inputs"] == [{"question": "A?"}]
    assert kwargs["outputs"] == [{"reference_answer": "a", "source_file": "f.pdf", "page": "2"}]
    assert kwargs["ids"] == [example_id("ds", "A?")]
    assert kwargs["dataset_id"] == "ds-id"
    client.update_examples.assert_not_called()
    client.delete_example.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest libs/ktem/ktem_tests/test_eval_dataset.py -v`
Expected: collection error `No module named 'ktem.evals'`

- [ ] **Step 3: Implement `dataset.py`**

- Choose the format by suffix. Read CSV with `csv.DictReader` and `encoding="utf-8-sig"`. For JSONL, read one JSON object per non-blank line.
- Strip every value. Empty optional fields become `None`. A JSON number `page` becomes `str`.
- Error messages are `line N: ...`, where N counts data rows for CSV and file lines for JSONL. Duplicates compare `question.strip()`.
- `plan_sync` compares existing items with `str(e.id)`. An item goes to `update` when its `inputs` or `outputs` differ from the file.
- `sync_dataset` gets the dataset id from `read_dataset(dataset_name=...)` when `has_dataset` is true. Otherwise it calls `create_dataset(dataset_name, description="kotaemon eval dataset")`, and `existing` is then empty.
  - It calls `create_examples(inputs=, outputs=, ids=, dataset_id=)`, `update_examples(example_ids=, inputs=, outputs=)` and `delete_example(id)` only when the matching list is non-empty.
  - It returns the plan.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest libs/ktem/ktem_tests/test_eval_dataset.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add libs/ktem/ktem/evals/__init__.py libs/ktem/ktem/evals/dataset.py libs/ktem/ktem_tests/test_eval_dataset.py
git commit -m "feat(evals): load eval datasets and sync them to LangSmith by stable id"
```

---

### Task 6: Evaluators

**Files:**
- Create: `libs/ktem/ktem/evals/evaluators.py`
- Test: `libs/ktem/ktem_tests/test_eval_evaluators.py`

**Interfaces:**
- Consumes: target output shape from Task 7: `outputs = {"answer", "raw_answer", "contexts": [{doc_id, file_name, page_label, type, text}], "evidence", "citations": {mode, claimed, matched}}`. `reference_outputs = EvalExample.outputs()` from Task 5.
- Produces:
  - `Judge = Callable[[str], str]`
  - `CORRECTNESS_LABELS = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}`
  - `GROUNDEDNESS_LABELS = {"grounded": 1.0, "partially_grounded": 0.5, "ungrounded": 0.0}`
  - `CORRECTNESS_PROMPT`, `GROUNDEDNESS_PROMPT` (str.format templates)
  - `parse_label(text: str, labels: Iterable[str]) -> str | None`
  - `make_correctness(judge) -> correctness(inputs, outputs, reference_outputs) -> dict`
  - `make_groundedness(judge) -> groundedness(outputs) -> dict`
  - `retrieval_hit(outputs, reference_outputs) -> dict`
  - `reciprocal_rank(outputs, reference_outputs) -> dict`
  - `citation_validity(outputs) -> dict`
  - `build_evaluators(judge) -> list[Callable]`
  - Every evaluator returns `{"key": <name>, "score": float | None, "comment": str}`. Parameter names must stay within langsmith's supported set (`inputs`, `outputs`, `reference_outputs`).

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from ktem.evals.evaluators import (
    citation_validity, make_correctness, make_groundedness, parse_label,
    reciprocal_rank, retrieval_hit,
)

C = ["partially_correct", "incorrect", "correct"]
G = ["partially_grounded", "ungrounded", "grounded"]

def ctx(name, page="1", type_=""):
    return {"doc_id": "d", "file_name": name, "page_label": page, "type": type_, "text": "t"}

OUT = {"answer": "Sargon.", "evidence": "Sargon of Akkad founded it.",
       "contexts": [ctx("other.pdf"), ctx("dir/Halley's Comet.PDF", "3")],
       "citations": {"mode": "highlight", "claimed": 4, "matched": 3}}
REF = {"reference_answer": "Sargon of Akkad.", "source_file": "halley's comet.pdf", "page": "3"}


@pytest.mark.parametrize("text, labels, want", [
    ("LABEL: Partially correct\nREASON: misses date", C, "partially_correct"),
    ("label: partially-correct", C, "partially_correct"),
    ("The answer is incorrect.", C, "incorrect"),
    ("Correct", C, "correct"),
    ("LABEL: ungrounded", G, "ungrounded"),
    ("I cannot tell", C, None),
])
def test_parse_label(text, labels, want):
    assert parse_label(text, labels) == want


def test_correctness_scores_label_and_prompt_has_inputs():
    prompts = []
    judge = lambda p: prompts.append(p) or "LABEL: partially_correct\nREASON: misses date"
    res = make_correctness(judge)({"question": "Who?"}, OUT, REF)
    assert res["key"] == "correctness" and res["score"] == 0.5
    assert "misses date" in res["comment"]
    assert "Who?" in prompts[0] and "Sargon of Akkad." in prompts[0] and "Sargon." in prompts[0]

def test_correctness_empty_answer_scores_zero_without_judge():
    def judge(_):
        raise AssertionError("judge must not be called")
    assert make_correctness(judge)({"question": "q"}, {**OUT, "answer": ""}, REF)["score"] == 0.0

def test_correctness_missing_outputs_is_none():
    assert make_correctness(lambda p: "correct")({"question": "q"}, {}, REF)["score"] is None

def test_correctness_unparseable_or_error_is_none():
    assert make_correctness(lambda p: "no idea")({"question": "q"}, OUT, REF)["score"] is None
    def boom(_):
        raise RuntimeError("boom")
    res = make_correctness(boom)({"question": "q"}, OUT, REF)
    assert res["score"] is None and "boom" in res["comment"]

def test_groundedness():
    assert make_groundedness(lambda p: "LABEL: grounded")(OUT)["score"] == 1.0
    assert make_groundedness(lambda p: "grounded")({**OUT, "evidence": ""})["score"] is None
    assert make_groundedness(lambda p: "grounded")({**OUT, "answer": ""})["score"] is None
    assert make_groundedness(lambda p: "grounded")({})["score"] is None

def test_retrieval_hit_file_and_page():
    assert retrieval_hit(OUT, REF)["score"] == 1.0
    assert retrieval_hit(OUT, {**REF, "page": "4"})["score"] == 0.0
    assert retrieval_hit(OUT, {"reference_answer": "x", "source_file": "halley's comet.pdf"})["score"] == 1.0
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
    assert citation_validity({**OUT, "citations": {"mode": "off", "claimed": 0, "matched": 0}})["score"] is None
    assert citation_validity({**OUT, "citations": {"mode": "inline", "claimed": 0, "matched": 0}})["score"] is None
    assert citation_validity({})["score"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest libs/ktem/ktem_tests/test_eval_evaluators.py -v`
Expected: `ImportError: cannot import name ... from 'ktem.evals.evaluators'` (module missing)

- [ ] **Step 3: Implement `evaluators.py`**

Prompts (exact copy, so the parser and the prompt agree):

```python
CORRECTNESS_PROMPT = """You are grading an answer to a question against a reference answer.

Question: {question}
Reference answer: {reference}
Answer to grade: {answer}

Judge only the facts, not style or length. Is the answer correct, partially correct, or incorrect compared to the reference?

Reply with exactly two lines:
LABEL: <correct | partially_correct | incorrect>
REASON: <one sentence>"""

GROUNDEDNESS_PROMPT = """You are checking whether an answer is supported by the context it was written from.

Context:
{evidence}

Answer:
{answer}

Is every claim in the answer supported by the context (grounded), only some claims (partially_grounded), or are the main claims unsupported (ungrounded)?

Reply with exactly two lines:
LABEL: <grounded | partially_grounded | ungrounded>
REASON: <one sentence>"""
```

`parse_label` algorithm:
1. Lowercase the text and rewrite `partially[\s-]+` to `partially_`.
2. If `label:\s*([a-z_]+)` captures a known label, return it.
3. Otherwise return the label whose first `(?<![a-z_])label(?![a-z_])` match comes earliest.
4. If none matches, return `None`.

Other behaviour:
- `"missing outputs"` means `not outputs` or a missing key. The comment then says `"target failed; no outputs"`.
- When a judge raises, the score is `None` and the comment is `f"judge error: {exc}"`.
- Retrieval matching compares `os.path.basename(...).lower()` of the file names. When `page` is given, it also requires `str(page_label) == str(page)`. Contexts whose `type == "thumbnail"` are skipped.
- `build_evaluators` returns `[make_correctness(judge), make_groundedness(judge), retrieval_hit, reciprocal_rank, citation_validity]`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest libs/ktem/ktem_tests/test_eval_evaluators.py -v`
Expected: 15 passed

- [ ] **Step 5: Commit**

```bash
git add libs/ktem/ktem/evals/evaluators.py libs/ktem/ktem_tests/test_eval_evaluators.py
git commit -m "feat(evals): add judge and retrieval evaluators"
```

---

### Task 7: Answer cleaning, citation facts and the eval target

**Files:**
- Create: `libs/ktem/ktem/evals/cleaning.py`, `libs/ktem/ktem/evals/runner.py`
- Modify: `libs/kotaemon/kotaemon/indices/qa/utils.py:44-49` (`find_start_end_phrase` lowercases `None`)
- Test: `libs/ktem/ktem_tests/test_eval_runner.py`, `libs/kotaemon/tests/test_qa_utils.py`

**Interfaces:**
- Consumes: `eval_stubs` (Task 3); `FullQAPipeline` with Task 3 spans.
- Produces:
  - `cleaning.strip_think(text) -> str`
  - `cleaning.inline_markers(text) -> list[int]`: distinct numbers in first-seen order from `【n】` / `【n, m】`, falling back to `[n]` only when the text has no `【…】` marker. These are the same patterns as `AnswerWithInlineCitation.replace_citation_with_link`.
  - `cleaning.clean_answer(text, mode) -> str`
  - `runner.EvalQAPipeline(FullQAPipeline)`: stores the docs from `retrieve()` in `self._eval_docs`.
  - `runner.citation_facts(mode: str, answer: Document, docs: Sequence[Document]) -> {"mode": str, "claimed": int, "matched": int}`
  - `runner.make_target(build_pipeline: Callable[[], EvalQAPipeline], citation_mode: str) -> Callable[[dict], dict]`

- [ ] **Step 1: Write the failing tests**

`libs/kotaemon/tests/test_qa_utils.py`:

```python
from kotaemon.indices.qa.utils import find_start_end_phrase

DOC = "Sargon of Akkad founded the empire. It ruled Mesopotamia from Akkad."

def test_find_start_end_phrase_accepts_none():
    assert find_start_end_phrase("Sargon of Akkad", None, DOC) == ((0, 15), 15)
    assert find_start_end_phrase(None, None, DOC) == (None, 0)
```

`libs/ktem/ktem_tests/test_eval_runner.py`:

```python
import json

import pytest

from kotaemon.base import Document
from kotaemon.indices.qa.citation import CiteEvidence
from kotaemon.indices.qa.citation_qa_inline import InlineEvidence
from ktem.evals.cleaning import clean_answer, inline_markers, strip_think
from ktem.evals.runner import EvalQAPipeline, citation_facts, make_target
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
    ans = Document(text="a", metadata={"citation": CiteEvidence(evidences=[
        "Sargon of Akkad founded the empire", "The comet returns every 76 years"])})
    assert citation_facts("highlight", ans, DOCS) == {"mode": "highlight", "claimed": 2, "matched": 1}
    none = Document(text="a", metadata={"citation": None})
    assert citation_facts("highlight", none, DOCS) == {"mode": "highlight", "claimed": 0, "matched": 0}

def test_citation_facts_inline_missing_phrase():
    ans = Document(text="Sargon founded it【1】 and ruled【2】", metadata={"citation": [
        InlineEvidence(idx=1, start_phrase="Sargon of Akkad", end_phrase="founded the empire"),
        InlineEvidence(idx=2, start_phrase="Halley comet orbit text", end_phrase=None)]})
    assert citation_facts("inline", ans, DOCS) == {"mode": "inline", "claimed": 2, "matched": 1}

def test_citation_facts_inline_idx_none_uses_position():
    ans = Document(text="X【1】", metadata={"citation": [
        InlineEvidence(idx=None, start_phrase="Sargon of Akkad", end_phrase="founded the empire")]})
    assert citation_facts("inline", ans, DOCS)["matched"] == 1

def test_citation_facts_off():
    assert citation_facts("off", Document(text="a"), DOCS) == {"mode": "off", "claimed": 0, "matched": 0}

def test_target_returns_plain_json_and_builds_fresh_pipelines():
    built = []

    def build():
        p = EvalQAPipeline(retrievers=[StubRetriever()],
                           answering_pipeline=StubAnswer(answer_text="<think>hm</think>Sargon of Akkad."))
        built.append(p)
        return p

    target = make_target(build, "highlight")
    out = target({"question": "Who founded Akkad?"})
    target({"question": "Again?"})
    assert len(built) == 2 and built[0] is not built[1]
    assert out["answer"] == "Sargon of Akkad."
    assert out["raw_answer"] == "<think>hm</think>Sargon of Akkad."
    assert out["contexts"] == [{"doc_id": out["contexts"][0]["doc_id"], "file_name": "Akkadian Empire.pdf",
                                "page_label": "1", "type": "", "text": DOC_TEXT}]
    assert "Content from Akkadian Empire.pdf" in out["evidence"]
    assert out["citations"] == {"mode": "highlight", "claimed": 0, "matched": 0}
    json.dumps(out)

def test_eval_pipeline_keeps_simple_id():
    assert EvalQAPipeline.get_info()["id"] == "simple"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd libs/kotaemon && ../../.venv/Scripts/python.exe -m pytest tests/test_qa_utils.py -v`, then from the repo root `.venv/Scripts/python.exe -m pytest libs/ktem/ktem_tests/test_eval_runner.py -v`
Expected: `AttributeError: 'NoneType' object has no attribute 'lower'`; `No module named 'ktem.evals.cleaning'`

- [ ] **Step 3: Implement**

`find_start_end_phrase`:
- Lowercase only non-`None` phrases. The existing `if sentence is None: continue` then skips the missing one.

`strip_think`:
- Remove every `<think>.*?</think>` (DOTALL). If a stray `</think>` remains, keep only the text after the last one. Strip the result.

`clean_answer`:
- Always apply `strip_think`.
- For `"inline"`, also:
  - cut from `CITATION LIST` onwards
  - remove the marker pattern `inline_markers` matched (`【[\d,\s]+】`, or `\[\d+\]` as the fallback)
  - strip the result.

`EvalQAPipeline.retrieve`:
- Call `super().retrieve(...)`, which carries the Task 3 span.
- Set `self._eval_docs = docs`.
- Return `docs, infos`.

`citation_facts`:
- **highlight:**
  - Quotes come from `metadata["citation"].evidences`, or `[]` when the citation is `None`.
  - A quote is matched when any doc yields a `find_text(quote, doc.text)` span whose `doc.text[start:end]` contains no `"|"`. This is the rule in `citation_qa.py:match_evidence_with_context`.
- **inline:**
  - `claimed` is the set of numbers from `inline_markers(answer.text)`.
  - An evidence entry's number is `ev.idx`, or its 1-based position when `idx` is `None`.
  - It matches when `find_start_end_phrase(ev.start_phrase, ev.end_phrase, doc.text)[0]` is not `None` for any doc.
  - `matched = len(claimed ∩ matched_numbers)`.
- **off:** zeros.

`make_target(build_pipeline, citation_mode)` returns a `target(inputs)` that:
- builds a fresh pipeline per call
- drains `stream(inputs["question"], conv_id=f"eval-{uuid4()}", history=[])` and uses the generator's return value as `answer`; raises `RuntimeError("pipeline returned no answer")` if that is `None`
- reads `docs = getattr(pipeline, "_eval_docs", [])`
- computes `evidence = pipeline.evidence_pipeline(docs).content[1]`
- returns the dict asserted above, with `type` taken from `metadata.get("type", "")`.

- [ ] **Step 4: Run tests to verify they pass**

Run both commands from Step 2.
Expected: 1 passed; 9 passed

- [ ] **Step 5: Commit**

```bash
git add libs/kotaemon/kotaemon/indices/qa/utils.py libs/kotaemon/tests/test_qa_utils.py libs/ktem/ktem/evals/cleaning.py libs/ktem/ktem/evals/runner.py libs/ktem/ktem_tests/test_eval_runner.py
git commit -m "feat(evals): eval target with captured contexts and citation facts"
```

---

### Task 8: Headless bootstrap

**Files:**
- Create: `libs/ktem/ktem/evals/bootstrap.py`
- Test: `libs/ktem/ktem_tests/test_eval_bootstrap.py`

**Interfaces:**
- Consumes: Task 4 `FileIndex.get_selected_ids`; Task 7 `EvalQAPipeline`.
- Produces:
  - `class BootstrapError(RuntimeError)`
  - `@dataclass class EvalContext(build_pipeline: Callable[[], EvalQAPipeline], judge: Callable[[str], str], citation_mode: str, metadata: dict)`
  - `resolve_user_id(username: str | None, user_management: bool) -> str`
  - `load_user_settings(defaults: dict, user_id: str) -> dict`
  - `apply_overrides(settings: dict, index_id: int, citation: str | None, llm_scoring: bool) -> dict`. Returns a new dict.
  - `select_index(indices: Sequence[Any], name_or_id: str | None) -> FileIndex`
  - `git_sha() -> str | None`
  - `bootstrap(username: str | None, index: str | None, citation: str | None, llm_scoring: bool, dataset_name: str) -> EvalContext`

- [ ] **Step 1: Write the failing tests**

```python
import pytest
from sqlmodel import Session, SQLModel, create_engine

from ktem.db.models import Settings, User
from ktem.evals.bootstrap import (
    BootstrapError, apply_overrides, load_user_settings, resolve_user_id, select_index,
)
from ktem.index.file.index import FileIndex

CITE = "reasoning.options.simple.highlight_citation"
LLM_SCORING = "index.options.1.use_llm_reranking"


@pytest.fixture(autouse=True)
def db(mocker):
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        s.add(User(id="u1", username="Alice", username_lower="alice", password="x"))
        s.add(Settings(user="u1", setting={"a": 2, "old": 9}))
        s.commit()
    mocker.patch("ktem.evals.bootstrap.engine", engine)


def test_resolve_user_id():
    assert resolve_user_id("ALICE", True) == "u1"
    assert resolve_user_id("anything", False) == "default"
    with pytest.raises(BootstrapError, match="user 'bob' not found"):
        resolve_user_id("bob", True)
    with pytest.raises(BootstrapError, match="--user"):
        resolve_user_id(None, True)

def test_load_user_settings_overlays_saved_values():
    defaults = {"a": 1, "b": 1}
    assert load_user_settings(defaults, "u1") == {"a": 2, "b": 1, "old": 9}
    fresh = load_user_settings(defaults, "nobody")
    assert fresh == defaults and fresh is not defaults

def test_apply_overrides():
    base = {CITE: "highlight", LLM_SCORING: True}
    out = apply_overrides(base, 1, "inline", llm_scoring=False)
    assert out == {CITE: "inline", LLM_SCORING: False}
    assert base == {CITE: "highlight", LLM_SCORING: True}
    assert apply_overrides(base, 1, None, llm_scoring=True)[CITE] == "highlight"

def test_select_index():
    class LightRAGIndex(FileIndex):
        pass
    graph = LightRAGIndex(app=None, id=3, name="LightRAG Collection", config={})
    files = FileIndex(app=None, id=1, name="File Collection", config={})
    assert select_index([graph, files], None) is files
    assert select_index([graph, files], "file collection") is files
    assert select_index([graph, files], "3") is graph
    with pytest.raises(BootstrapError, match="File Collection"):
        select_index([graph, files], "nope")
    with pytest.raises(BootstrapError):
        select_index([graph], None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest libs/ktem/ktem_tests/test_eval_bootstrap.py -v`
Expected: `No module named 'ktem.evals.bootstrap'`

- [ ] **Step 3: Implement `bootstrap.py`**

- Import `engine` at module level (`from ktem.db.engine import engine`) so tests can patch `ktem.evals.bootstrap.engine`.
- User lookup: `select(User).where(User.username_lower == username.lower())`.
- Settings: the first `Settings` row with `user == user_id`, then `{**defaults, **row.setting}`.
- `select_index`:
  - With `None`, pick the first index whose `type(...) is FileIndex`.
  - Otherwise match `name.lower()` or `str(id)`.
  - The error lists the available names.

`bootstrap()`:
1. Create `app = BaseApp()` (headless; `from ktem.app import BaseApp`).
2. `user_id = resolve_user_id(username, app.f_user_management)`.
3. `idx = select_index(app.index_manager.indices, index)`.
4. `settings = apply_overrides(load_user_settings(app.default_settings.flatten(), user_id), idx.id, citation, llm_scoring)`.
5. `selected = ["all", [], user_id]`. If `idx.get_selected_ids(selected)` is empty, raise `BootstrapError(f"user {username or user_id!r} has no files in index {idx.name!r}")`.
6. `build_pipeline = lambda: EvalQAPipeline.get_pipeline(settings, {"app": {"regen": False}, "pipeline": {}}, idx.get_retriever_pipelines(settings, user_id, selected))`.
7. Judge: `llm = llms.get_default()`, then `judge = lambda prompt: llm(prompt).text`.
8. Metadata, with `p = f"index.options.{idx.id}."`:
   - `{"dataset": dataset_name, "user": username or user_id, "index": idx.name}`
   - `"llm": settings.get("reasoning.options.simple.llm") or llms.get_default_name()`
   - `"judge_llm": llms.get_default_name()`
   - `"citation": settings[CITE]`
   - `"retrieval_mode": settings.get(p + "retrieval_mode")`, `"num_retrieval": settings.get(p + "num_retrieval")`, `"use_reranking": settings.get(p + "use_reranking")`
   - `"llm_scoring": llm_scoring`, `"git_sha": git_sha()`

`git_sha()` runs `git rev-parse --short HEAD` and returns `None` on any error.

`bootstrap()` itself is exercised end-to-end in Task 11.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest libs/ktem/ktem_tests/test_eval_bootstrap.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add libs/ktem/ktem/evals/bootstrap.py libs/ktem/ktem_tests/test_eval_bootstrap.py
git commit -m "feat(evals): build the app headless for one user and one index"
```

---

### Task 9: CLI

**Files:**
- Create: `libs/ktem/ktem/evals/run.py`
- Test: `libs/ktem/ktem_tests/test_eval_cli.py`

**Interfaces:**
- Consumes:
  - Task 1 `export_langsmith_env`
  - Task 5 `load_dataset`, `DatasetError`, `sync_dataset`
  - Task 6 `build_evaluators`
  - Task 7 `make_target`
  - Task 8 `bootstrap`, `BootstrapError`, `EvalContext`
- Produces:
  - `build_parser() -> argparse.ArgumentParser`
  - `summarize(rows: Iterable[Mapping]) -> dict[str, tuple[float | None, int]]`, mapping key to (mean of non-`None` scores, number scored)
  - `main(argv: Sequence[str] | None = None) -> int`
  - Exit code 0 on success and 2 on usage or setup errors.

Flags, exactly as in the spec:
- `--dataset PATH` (required)
- `--dataset-name NAME` (default: file stem)
- `--user USERNAME`
- `--index NAME_OR_ID`
- `--citation {highlight,inline,off}`
- `--llm-scoring`
- `--experiment-prefix STR` (default `f"{dataset_name}-{citation_mode}"`)
- `--max-concurrency N` (int, default 2)
- `--prune`

`main()` order, so that nothing reaches LangSmith before validation:
1. `export_langsmith_env()`
2. Key check: `LANGSMITH_API_KEY` or `LANGCHAIN_API_KEY` in `os.environ`. Otherwise print `error: LANGSMITH_API_KEY is not set (env or .env)` to stderr and return 2.
3. `load_dataset`
4. `bootstrap`
5. `client = Client()`
6. `sync_dataset`; print `dataset <name>: N created, N updated, N deleted`
7. `evaluate(target, data=dataset_name, evaluators=..., experiment_prefix=..., max_concurrency=args.max_concurrency, metadata=ctx.metadata, client=client)`
8. Print `experiment: <results.experiment_name>`, then one `key  mean  (n scored)` line per metric. A mean of `None` prints `-`.

Import `Client` and `evaluate` (`from langsmith.evaluation import evaluate`) at module level so tests can patch them. Also catch `DatasetError` / `BootstrapError` and print `error: <msg>`.

- [ ] **Step 1: Write the failing tests**

```python
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langsmith.evaluation import EvaluationResult

from ktem.evals import run as cli
from ktem.evals.bootstrap import BootstrapError, EvalContext
from ktem.evals.runner import EvalQAPipeline
from ktem_tests.eval_stubs import StubAnswer, StubRetriever

ROWS = ('{"question": "Who founded Akkad?", "reference_answer": "Sargon of Akkad.", '
        '"source_file": "Akkadian Empire.pdf", "page": "1"}\n')


def fake_evaluate(calls, examples):
    def _evaluate(target, *, data, evaluators, experiment_prefix, max_concurrency, metadata, client):
        calls.update(data=data, prefix=experiment_prefix, max_concurrency=max_concurrency,
                     metadata=metadata)
        rows = []
        for ex in examples:
            outputs = target(ex.inputs())
            args = {"inputs": ex.inputs(), "outputs": outputs, "reference_outputs": ex.outputs()}
            results = [EvaluationResult(**ev(**{k: args[k] for k in inspect.signature(ev).parameters}))
                       for ev in evaluators]
            rows.append({"evaluation_results": {"results": results}})
        return SimpleNamespace(experiment_name="exp-1", rows=rows)
    return _evaluate


@pytest.fixture
def setup(tmp_path, monkeypatch, mocker):
    monkeypatch.setattr(cli, "export_langsmith_env", lambda *a, **k: None)
    monkeypatch.setenv("LANGSMITH_API_KEY", "test")
    path = tmp_path / "akkad.jsonl"; path.write_text(ROWS, encoding="utf-8")
    ctx = EvalContext(
        build_pipeline=lambda: EvalQAPipeline(retrievers=[StubRetriever()], answering_pipeline=StubAnswer()),
        judge=lambda prompt: "LABEL: correct\nREASON: ok",
        citation_mode="highlight", metadata={"index": "File Collection"})
    boot = mocker.patch.object(cli, "bootstrap", return_value=ctx)
    client_cls = mocker.patch.object(cli, "Client")
    mocker.patch.object(cli, "sync_dataset",
                        return_value=SimpleNamespace(create=[1], update=[], delete=[]))
    calls = {}
    mocker.patch.object(cli, "evaluate",
                        side_effect=fake_evaluate(calls, cli.load_dataset(path)))
    mocker.patch.object(cli, "_rows", side_effect=lambda results: results.rows)
    return SimpleNamespace(path=path, ctx=ctx, boot=boot, client_cls=client_cls, calls=calls)


def test_main_happy_path(setup, capsys):
    assert cli.main(["--dataset", str(setup.path)]) == 0
    out = capsys.readouterr().out
    assert "dataset akkad: 1 created, 0 updated, 0 deleted" in out
    assert "experiment: exp-1" in out
    assert "correctness" in out and "1.00" in out and "retrieval_hit" in out
    assert setup.calls == {"data": "akkad", "prefix": "akkad-highlight",
                           "max_concurrency": 2, "metadata": {"index": "File Collection"}}

def test_main_requires_api_key(setup, monkeypatch, capsys):
    monkeypatch.delenv("LANGSMITH_API_KEY"); monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    assert cli.main(["--dataset", str(setup.path)]) == 2
    assert "LANGSMITH_API_KEY" in capsys.readouterr().err
    setup.boot.assert_not_called()

def test_main_exports_env_before_key_check(setup, monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY")
    monkeypatch.setattr(cli, "export_langsmith_env",
                        lambda *a, **k: monkeypatch.setenv("LANGSMITH_API_KEY", "from-dotenv"))
    assert cli.main(["--dataset", str(setup.path)]) == 0

def test_main_bootstrap_error_stops_before_langsmith(setup, capsys):
    setup.boot.side_effect = BootstrapError("user 'x' has no files in index 'File Collection'")
    assert cli.main(["--dataset", str(setup.path), "--user", "x"]) == 2
    assert "has no files" in capsys.readouterr().err
    setup.client_cls.assert_not_called()

def test_summarize_skips_none():
    rows = [{"evaluation_results": {"results": [EvaluationResult(key="k", score=s)]}}
            for s in (1.0, None, 0.0)]
    assert cli.summarize(rows) == {"k": (0.5, 2)}
```

Also add to `run.py` a one-line helper `_rows(results) -> Iterable[Mapping]` that returns `results`. `main` calls `summarize(_rows(results))`, which keeps the real `ExperimentResults` iteration in one patchable place.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest libs/ktem/ktem_tests/test_eval_cli.py -v`
Expected: `ImportError: cannot import name 'run' from 'ktem.evals'`

- [ ] **Step 3: Implement `run.py`** per the Interfaces block and the order above. Add `if __name__ == "__main__": sys.exit(main())`.

- [ ] **Step 4: Run tests, the full ktem suite, and the CLI help from the repo root**

Run:
- `.venv/Scripts/python.exe -m pytest libs/ktem/ktem_tests --ignore=libs/ktem/ktem_tests/test_qa.py -v`
- `.venv/Scripts/python.exe -m ktem.evals.run --help`

Expected: all pass; the help lists the nine flags

- [ ] **Step 5: Commit**

```bash
git add libs/ktem/ktem/evals/run.py libs/ktem/ktem_tests/test_eval_cli.py
git commit -m "feat(evals): add the ktem.evals.run CLI"
```

---

### Task 10: Dependency, env example and CI

**Files:**
- Modify: `libs/ktem/pyproject.toml` (`dependencies`), `uv.lock`, `.env.example` (append), `.github/workflows/unit-test.yaml` (new step after "Test kotaemon with pytest")

- [ ] **Step 1: Declare the dependency.** Add `"langsmith>=0.1.147,<0.2",` to `ktem`'s `dependencies`.

- [ ] **Step 2: Relock without upgrades**

Run: `uv lock`, then `git diff uv.lock | grep -E '^[-+]version'`
Expected: `uv lock` succeeds; the grep prints nothing, so no package version changed and only ktem's dependency list gained `langsmith`

- [ ] **Step 3: Append to `.env.example`** (exact copy):

```
# LangSmith tracing and evals (optional).
# Prompts, retrieved document chunks and answers are sent to LangSmith.
# Required for `python -m ktem.evals.run`:
# LANGSMITH_API_KEY=
# Also trace the live app (eval runs are always traced):
# LANGSMITH_TRACING=true
# LANGSMITH_PROJECT=kotaemon
# For EU or self-hosted LangSmith:
# LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

- [ ] **Step 4: Add the CI step** after "Test kotaemon with pytest":

```yaml
      - name: Test ktem with pytest
        run: |
          source .venv/bin/activate
          pytest libs/ktem/ktem_tests --ignore=libs/ktem/ktem_tests/test_qa.py
```

- [ ] **Step 5: Verify locally**

Run:
- `uv lock --check`
- `.venv/Scripts/python.exe -m pytest libs/ktem/ktem_tests --ignore=libs/ktem/ktem_tests/test_qa.py -q`
- `cd libs/kotaemon && ../../.venv/Scripts/python.exe -m pytest -q`

Expected: lock is up to date; ktem all pass; kotaemon has no new failures versus `main`. Record the pre-existing failures, if any, by running the same command on a `main` checkout.

- [ ] **Step 6: Commit**

```bash
git add libs/ktem/pyproject.toml uv.lock .env.example .github/workflows/unit-test.yaml
git commit -m "build: declare langsmith for ktem, document env vars, run ktem tests in CI"
```

---

### Task 11: End-to-end check (needs the human: a LangSmith key and a working LLM)

Uses the local sample `eval_data/uda_wiki10/` (gitignored; see its README). The executor stops here and asks the human partner to provide the key and run, or to confirm that the executor may run it with the key from `.env`.

- [ ] **Step 1:** Start the app (`.venv/Scripts/python.exe app.py`). As the chosen user, upload `eval_data/uda_wiki10/pdfs/*.pdf` to "File Collection" and wait for indexing to finish.
- [ ] **Step 2:** Run `.venv/Scripts/python.exe -m ktem.evals.run --dataset eval_data/uda_wiki10/dataset.jsonl --dataset-name uda_wiki10 --user <username>`.
  Expected:
  - `dataset uda_wiki10: 25 created, 0 updated, 0 deleted`
  - a LangSmith experiment URL
  - per-metric means for 25 rows
  - `retrieval_hit` mean > 0.5
- [ ] **Step 3:** In LangSmith, open one row. Expected tree: target → `kotaemon.qa` → `retrieve` (retriever, documents with file name and page) → `answer_with_context` → the LLM run. The citation LLM call sits inside `answer_with_context`, not as a separate root. Judge calls sit under the `evaluators` project. Check this for the configured model type (OpenAI-compatible or LangChain-based).
- [ ] **Step 4:** Find the row "Who is the most recent justice to join the Supreme Court?". Expected: the answer names Neil Gorsuch and `correctness` is 1. If it names a later justice, the answer did not come from retrieval; check `retrieval_hit` for that row.
- [ ] **Step 5:** Run Step 2 again. Expected: `0 created, 0 updated, 0 deleted`, and a second experiment that can be compared with the first.
- [ ] **Step 6:** Start the app with `LANGSMITH_TRACING=true` in `.env` and ask one question in the chat UI. Expected: one `kotaemon.qa` trace in the `LANGSMITH_PROJECT` project. Then remove the variable, restart, ask again. Expected: no new trace.
