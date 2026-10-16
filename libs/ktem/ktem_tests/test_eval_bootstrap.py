import pytest
from ktem.db.models import Settings, User
from ktem.evals.bootstrap import (
    BootstrapError,
    apply_overrides,
    load_user_settings,
    resolve_user_id,
    select_index,
)
from ktem.index.file.index import FileIndex
from sqlmodel import Session, SQLModel, create_engine

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
