import json
import pathlib

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from multi_doc.exception.custom_exception import DocumentPortalException
from multi_doc.src.document_chat.retrieval import ConversationalRAG
from multi_doc.src.document_ingestion.data_ingestion import FaissManager

INDEX_PATH = "faiss_index/s1"


def _build_index(path: str = INDEX_PATH) -> str:
    fm = FaissManager(index_dir=pathlib.Path(path))
    fm.load_or_create(
        texts=["Transformers rely on self-attention.", "RNNs process tokens one at a time."],
        metadatas=[
            {"source": "paper.pdf", "filename": "paper.pdf", "page": 2},
            {"source": "paper.pdf", "filename": "paper.pdf", "page": 4},
        ],
    )
    return path


def test_conversationalrag_error_handling(tmp_dirs, stub_model_loader):
    rag = ConversationalRAG(session_id="s1")
    with pytest.raises(DocumentPortalException):
        rag.invoke("hello")
    with pytest.raises(DocumentPortalException):
        rag.load_retriever_from_faiss(index_path="faiss_index/does_not_exist")


def test_invoke_before_loading_retriever_explains_what_is_missing(tmp_dirs, stub_model_loader):
    rag = ConversationalRAG(session_id="s1")
    with pytest.raises(DocumentPortalException, match="not initialized"):
        rag.invoke("hello")


def test_load_retriever_builds_chain_and_answers(tmp_dirs, stub_model_loader):
    index_path = _build_index()
    rag = ConversationalRAG(session_id="s1")
    retriever = rag.load_retriever_from_faiss(index_path=index_path)
    assert retriever is not None
    assert rag.invoke("What do transformers rely on?") == "stubbed answer"


def test_follow_up_question_with_chat_history(tmp_dirs, stub_model_loader):
    index_path = _build_index()
    rag = ConversationalRAG(session_id="s1")
    rag.load_retriever_from_faiss(index_path=index_path)
    history = [
        HumanMessage(content="What do transformers rely on?"),
        AIMessage(content="Self-attention."),
    ]
    assert rag.invoke("And RNNs?", chat_history=history) == "stubbed answer"


def test_retriever_uses_explicit_arguments(tmp_dirs, stub_model_loader):
    index_path = _build_index()
    rag = ConversationalRAG(session_id="s1")
    rag.load_retriever_from_faiss(index_path=index_path, k=2, search_type="similarity")
    assert rag.retriever.search_kwargs["k"] == 2
    assert rag.retriever.search_type == "similarity"


def test_retriever_defaults_come_from_config(tmp_dirs, stub_model_loader):
    index_path = _build_index()
    rag = ConversationalRAG(session_id="s1")
    rag.load_retriever_from_faiss(index_path=index_path)
    assert rag.retriever.search_kwargs["k"] == 3
    assert rag.retriever.search_type == "similarity"


def test_index_built_with_another_embedding_model_is_rejected(tmp_dirs, stub_model_loader):
    index_path = _build_index()
    meta_file = pathlib.Path(index_path) / "ingested_meta.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    meta["embedding_model"] = "some-other-model"
    meta_file.write_text(json.dumps(meta), encoding="utf-8")

    rag = ConversationalRAG(session_id="s1")
    with pytest.raises(DocumentPortalException):
        rag.load_retriever_from_faiss(index_path=index_path)


def test_invoke_with_sources_returns_files_and_pages(tmp_dirs, stub_model_loader):
    if not hasattr(ConversationalRAG, "invoke_with_sources"):
        pytest.skip("invoke_with_sources is not in this version of retrieval.py")

    index_path = _build_index()
    rag = ConversationalRAG(session_id="s1")
    rag.load_retriever_from_faiss(index_path=index_path)

    result = rag.invoke_with_sources("What do transformers rely on?")
    assert result["answer"] == "stubbed answer"
    found = {(s["filename"], s["page"]) for s in result["sources"]}
    assert found == {("paper.pdf", 3), ("paper.pdf", 5)}