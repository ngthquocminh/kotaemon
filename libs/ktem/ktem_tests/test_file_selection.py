from types import SimpleNamespace

import pytest
from ktem.index.file.index import FileIndex
from sqlalchemy import Column, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session

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
        s.add_all(
            [
                Source(id="f1", user="alice"),
                Source(id="f2", user="bob"),
                Source(id="f3", user="alice"),
            ]
        )
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
    ids = make_index(private=False).get_selected_ids(["all", [], "alice"])
    assert sorted(ids) == ["f1", "f2", "f3"]


def test_select_returns_given_ids(make_index):
    assert make_index().get_selected_ids(["select", ["f2"], "alice"]) == ["f2"]


@pytest.mark.parametrize(
    "selected", [["disabled", ["f1"], "alice"], ["all", [], None], None, []]
)
def test_empty_cases(make_index, selected):
    assert make_index().get_selected_ids(selected) == []


def test_file_selector_delegates():
    from ktem.index.file.ui import FileSelector

    sel = FileSelector.__new__(FileSelector)
    sel._index = SimpleNamespace(
        get_selected_ids=lambda s: ["x"] if s == ["all", [], "u"] else []
    )
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
