import copy
import os
import pathlib
import sys

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("GROQ_API_KEY", "dummy")
os.environ.setdefault("GOOGLE_API_KEY", "dummy")
os.environ.setdefault("LLM_PROVIDER", "groq")

import main


FAKE_CONFIG = {
    "embedding_model": {"provider": "huggingface", "model_name": "fake-embed", "normalize": True},
    "chunking": {"chunk_size": 200, "chunk_overlap": 20, "min_chunk_chars": 10},
    "retriever": {"top_k": 3, "search_type": "similarity", "fetch_k": 6, "lambda_mult": 0.7},
    "active_llm": "groq",
    "llm": {
        "groq": {
            "provider": "groq",
            "model_name": "fake-llm",
            "temperature": 0.0,
            "max_output_tokens": 128,
        }
    },
}


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def clear_sessions():
    def _clear():
        for name in ("SESSIONS", "HISTORY", "RAGS"):
            store = getattr(main, name, None)
            if store is not None:
                store.clear()

    _clear()
    yield
    _clear()


@pytest.fixture
def tmp_dirs(tmp_path: pathlib.Path):
    data_dir = tmp_path / "data"
    faiss_dir = tmp_path / "faiss_index"
    data_dir.mkdir(parents=True, exist_ok=True)
    faiss_dir.mkdir(parents=True, exist_ok=True)
    cwd = pathlib.Path.cwd()
    try:
        os.chdir(tmp_path)
        yield {"data": data_dir, "faiss": faiss_dir}
    finally:
        os.chdir(cwd)


class _StubEmbeddings:
    def embed_query(self, text: str):
        return [0.0, 0.1, 0.2]

    def embed_documents(self, texts):
        return [[0.0, 0.1, 0.2] for _ in texts]

    def __call__(self, text: str):
        return [0.0, 0.1, 0.2]


def _stub_llm():
    return RunnableLambda(lambda _: AIMessage(content="stubbed answer"))


@pytest.fixture
def stub_model_loader(monkeypatch):
    from multi_doc.utils import model_loader as ml_mod
    import multi_doc.src.document_ingestion.data_ingestion as di
    import multi_doc.src.document_chat.retrieval as r

    class FakeApiKeyMgr:
        def __init__(self):
            self.api_keys = {"GROQ_API_KEY": "x", "GOOGLE_API_KEY": "y"}

        def get(self, key: str) -> str:
            return self.api_keys[key]

    class FakeModelLoader:
        def __init__(self):
            self.api_key_mgr = FakeApiKeyMgr()
            self.config = copy.deepcopy(FAKE_CONFIG)

        def load_embeddings(self):
            return _StubEmbeddings()

        def load_llm(self):
            return _stub_llm()

    monkeypatch.setattr(ml_mod, "ApiKeyManager", FakeApiKeyMgr)
    monkeypatch.setattr(ml_mod, "ModelLoader", FakeModelLoader)
    monkeypatch.setattr(di, "ModelLoader", FakeModelLoader)
    monkeypatch.setattr(r, "ModelLoader", FakeModelLoader)
    yield FakeModelLoader


@pytest.fixture
def stub_ingestor(monkeypatch):
    import multi_doc.src.document_ingestion.data_ingestion as di

    class FakeIngestor:
        def __init__(self, use_session_dirs=True, session_id=None, **kwargs):
            self.use_session = use_session_dirs
            self.session_id = session_id or "sess_test"
            self.faiss_dir = pathlib.Path(kwargs.get("faiss_base", "faiss_index")) / self.session_id

        def built_retriver(self, uploaded_files, **kwargs):
            return None

        build_retriever = built_retriver

    monkeypatch.setattr(di, "ChatIngestor", FakeIngestor)
    monkeypatch.setattr(main, "ChatIngestor", FakeIngestor)
    yield FakeIngestor


@pytest.fixture
def stub_rag(monkeypatch):
    import multi_doc.src.document_chat.retrieval as r

    class FakeRAG:
        def __init__(self, session_id=None, retriever=None, model_loader=None):
            self.session_id = session_id
            self.retriever = retriever

        def load_retriever_from_faiss(self, index_path, **kwargs):
            return None

        def invoke(self, user_input, chat_history=None):
            return "stubbed answer"

        def invoke_with_sources(self, user_input, chat_history=None):
            return {"answer": "stubbed answer", "sources": []}

    monkeypatch.setattr(r, "ConversationalRAG", FakeRAG)
    monkeypatch.setattr(main, "ConversationalRAG", FakeRAG)
    yield FakeRAG