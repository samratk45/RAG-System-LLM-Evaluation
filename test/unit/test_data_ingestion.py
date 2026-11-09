import pathlib
import re

import pytest
from langchain_core.documents import Document

from multi_doc.exception.custom_exception import DocumentPortalException
from multi_doc.src.document_ingestion.data_ingestion import (
    ChatIngestor,
    FaissManager,
    generate_session_id,
)


class _Upload:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def getbuffer(self) -> bytes:
        return self._content


def _sample_text(words: int = 300) -> bytes:
    return " ".join(f"word{i}" for i in range(words)).encode("utf-8")


def test_generate_session_id_format_and_uniqueness():
    a = generate_session_id()
    b = generate_session_id()
    assert a != b
    assert re.fullmatch(r"session_\d{8}_\d{6}_[0-9a-f]{8}", a)
    assert re.fullmatch(r"session_\d{8}_\d{6}_[0-9a-f]{8}", b)


def test_session_id_is_safe_for_folder_names():
    assert re.fullmatch(r"[A-Za-z0-9_\-]{1,64}", generate_session_id())


def test_chat_ingestor_resolve_dir_uses_session_dirs(tmp_dirs, stub_model_loader):
    ing = ChatIngestor(temp_base="data", faiss_base="faiss_index", use_session_dirs=True)
    assert ing.session_id
    assert str(ing.temp_dir).endswith(ing.session_id)
    assert str(ing.faiss_dir).endswith(ing.session_id)
    assert ing.temp_dir.is_dir()
    assert ing.faiss_dir.is_dir()


def test_chat_ingestor_without_session_dirs_uses_base_folders(tmp_dirs, stub_model_loader):
    ing = ChatIngestor(temp_base="data", faiss_base="faiss_index", use_session_dirs=False)
    assert ing.temp_dir == pathlib.Path("data")
    assert ing.faiss_dir == pathlib.Path("faiss_index")


def test_split_chunks_respect_size_and_overlap(tmp_dirs, stub_model_loader):
    ing = ChatIngestor(temp_base="data", faiss_base="faiss_index", use_session_dirs=True)
    text = " ".join(f"word{i}" for i in range(400))
    docs = [Document(page_content=text, metadata={"source": "x.txt"})]
    chunks = ing._split(docs, chunk_size=500, chunk_overlap=100)
    assert len(chunks) >= 2
    assert all(len(c.page_content) <= 500 for c in chunks)
    last_word_of_first = chunks[0].page_content.split()[-1]
    assert last_word_of_first in chunks[1].page_content


def test_split_drops_tiny_chunks(tmp_dirs, stub_model_loader):
    ing = ChatIngestor(temp_base="data", faiss_base="faiss_index", use_session_dirs=True)
    docs = [Document(page_content="ok", metadata={"source": "x.txt"})]
    assert ing._split(docs, chunk_size=500, chunk_overlap=50) == []


def test_faiss_manager_skips_documents_already_indexed_at_creation(tmp_dirs, stub_model_loader):
    fm = FaissManager(index_dir=pathlib.Path("faiss_index/test"))
    fm.load_or_create(texts=["hello", "world"], metadatas=[{"source": "a"}, {"source": "b"}])
    docs = [Document(page_content="hello", metadata={"source": "a"})]
    assert fm.add_documents(docs) == 0


def test_faiss_manager_adds_only_new_documents(tmp_dirs, stub_model_loader):
    fm = FaissManager(index_dir=pathlib.Path("faiss_index/test"))
    fm.load_or_create(texts=["hello", "world"], metadatas=[{"source": "a"}, {"source": "b"}])
    new_doc = [Document(page_content="brand new text", metadata={"source": "c"})]
    assert fm.add_documents(new_doc) == 1
    assert fm.add_documents(new_doc) == 0


def test_faiss_manager_distinguishes_chunks_from_the_same_file(tmp_dirs, stub_model_loader):
    fm = FaissManager(index_dir=pathlib.Path("faiss_index/test"))
    fm.load_or_create(texts=["first chunk"], metadatas=[{"source": "same.pdf"}])
    more = [
        Document(page_content="second chunk", metadata={"source": "same.pdf"}),
        Document(page_content="third chunk", metadata={"source": "same.pdf"}),
    ]
    assert fm.add_documents(more) == 2


def test_faiss_manager_reloads_existing_index(tmp_dirs, stub_model_loader):
    index_dir = pathlib.Path("faiss_index/test")
    FaissManager(index_dir=index_dir).load_or_create(texts=["hello"], metadatas=[{"source": "a"}])
    reloaded = FaissManager(index_dir=index_dir).load_or_create()
    assert len(reloaded.similarity_search("hello", k=1)) == 1


def test_faiss_manager_rejects_index_built_with_another_embedding_model(tmp_dirs, stub_model_loader):
    index_dir = pathlib.Path("faiss_index/test")
    FaissManager(index_dir=index_dir).load_or_create(texts=["hello"], metadatas=[{"source": "a"}])
    other = FaissManager(index_dir=index_dir)
    other.embedding_name = "some-other-model"
    with pytest.raises(DocumentPortalException):
        other.load_or_create()


def test_build_retriever_indexes_uploaded_file(tmp_dirs, stub_model_loader):
    ing = ChatIngestor(temp_base="data", faiss_base="faiss_index", use_session_dirs=True)
    retriever = ing.built_retriver([_Upload("notes.txt", _sample_text())])
    assert (ing.faiss_dir / "index.faiss").exists()
    assert (ing.faiss_dir / "index.pkl").exists()
    results = retriever.invoke("word1")
    assert len(results) >= 1
    assert results[0].metadata.get("filename", "").startswith("notes")


def test_build_retriever_twice_keeps_index_consistent(tmp_dirs, stub_model_loader):
    ing = ChatIngestor(temp_base="data", faiss_base="faiss_index", use_session_dirs=True)
    first = ing.built_retriver([_Upload("notes.txt", _sample_text())])
    count_after_first = first.vectorstore.index.ntotal
    assert count_after_first > 1

    second = ing.built_retriver([_Upload("other.txt", _sample_text(100))])
    assert second.vectorstore.index.ntotal > count_after_first


def test_build_retriever_rejects_unsupported_file(tmp_dirs, stub_model_loader):
    ing = ChatIngestor(temp_base="data", faiss_base="faiss_index", use_session_dirs=True)
    with pytest.raises(DocumentPortalException):
        ing.built_retriver([_Upload("malware.xyz", b"data")])