import pytest

import main
from multi_doc.exception.custom_exception import DocumentPortalException

SESSION_ID = "sess_test"

needs_new_main = pytest.mark.skipif(
    not hasattr(main, "HISTORY"),
    reason="this test needs the updated main.py (HISTORY and RAGS state)",
)


def _store():
    return main.HISTORY if hasattr(main, "HISTORY") else main.SESSIONS


def _register(session_id: str = SESSION_ID):
    _store()[session_id] = []


class _FailsOnLoad:
    def __init__(self, session_id=None, **kwargs):
        pass

    def load_retriever_from_faiss(self, *args, **kwargs):
        raise DocumentPortalException("fail load /secret/path", None)


class _FailsOnInvokeWithPortalError:
    def __init__(self, session_id=None, **kwargs):
        pass

    def load_retriever_from_faiss(self, *args, **kwargs):
        return None

    def invoke(self, user_input, chat_history=None):
        raise DocumentPortalException("llm down", None)

    invoke_with_sources = invoke


class _FailsOnInvokeUnexpectedly:
    def __init__(self, session_id=None, **kwargs):
        pass

    def load_retriever_from_faiss(self, *args, **kwargs):
        return None

    def invoke(self, user_input, chat_history=None):
        raise RuntimeError("secret internal detail")

    invoke_with_sources = invoke


def test_chat_invalid_session_returns_400(client, clear_sessions, stub_rag):
    resp = client.post("/chat", json={"session_id": "nope", "message": "hi"})
    assert resp.status_code == 400
    assert "Invalid or expired" in resp.json()["detail"]


def test_chat_empty_message_returns_400(client, clear_sessions, stub_rag):
    _register()
    resp = client.post("/chat", json={"session_id": SESSION_ID, "message": "   "})
    assert resp.status_code == 400
    assert "Message cannot be empty" in resp.json()["detail"]


def test_chat_success_returns_answer_and_appends_history(client, clear_sessions, stub_rag):
    _register()
    resp = client.post("/chat", json={"session_id": SESSION_ID, "message": "Hello"})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "stubbed answer"
    assert len(_store()[SESSION_ID]) == 2


@needs_new_main
def test_chat_keeps_only_the_most_recent_messages(client, clear_sessions, stub_rag):
    _register()
    for i in range(main.MAX_HISTORY_MESSAGES):
        resp = client.post("/chat", json={"session_id": SESSION_ID, "message": f"question {i}"})
        assert resp.status_code == 200
    assert len(_store()[SESSION_ID]) == main.MAX_HISTORY_MESSAGES


@needs_new_main
def test_chat_reuses_one_rag_per_session(client, clear_sessions, stub_rag, monkeypatch):
    created = []

    class CountingRAG(stub_rag):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(main, "ConversationalRAG", CountingRAG)
    _register()
    for text in ("first", "second", "third"):
        assert client.post("/chat", json={"session_id": SESSION_ID, "message": text}).status_code == 200
    assert len(created) == 1


@needs_new_main
def test_chat_accepts_session_whose_index_is_on_disk(client, clear_sessions, stub_rag, tmp_dirs):
    index_dir = tmp_dirs["faiss"] / "sess_disk"
    index_dir.mkdir(parents=True)
    (index_dir / "index.faiss").write_bytes(b"")
    resp = client.post("/chat", json={"session_id": "sess_disk", "message": "Hello"})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "stubbed answer"


@needs_new_main
def test_chat_rejects_session_ids_that_could_escape_the_index_folder(client, clear_sessions, stub_rag):
    resp = client.post("/chat", json={"session_id": "../secrets", "message": "hi"})
    assert resp.status_code == 422


@needs_new_main
def test_chat_rejects_overlong_messages(client, clear_sessions, stub_rag):
    _register()
    resp = client.post("/chat", json={"session_id": SESSION_ID, "message": "x" * 2001})
    assert resp.status_code == 422


def test_chat_requires_both_fields(client, clear_sessions, stub_rag):
    assert client.post("/chat", json={"message": "hi"}).status_code == 422
    assert client.post("/chat", json={"session_id": SESSION_ID}).status_code == 422


@needs_new_main
def test_chat_load_failure_returns_502_without_internal_details(client, clear_sessions, monkeypatch):
    _register()
    monkeypatch.setattr(main, "ConversationalRAG", _FailsOnLoad)
    resp = client.post("/chat", json={"session_id": SESSION_ID, "message": "hi"})
    assert resp.status_code == 502
    detail = resp.json()["detail"].lower()
    assert "fail load" not in detail
    assert "secret" not in detail


@needs_new_main
def test_chat_model_error_returns_502_without_internal_details(client, clear_sessions, monkeypatch):
    _register()
    monkeypatch.setattr(main, "ConversationalRAG", _FailsOnInvokeWithPortalError)
    resp = client.post("/chat", json={"session_id": SESSION_ID, "message": "hi"})
    assert resp.status_code == 502
    assert "llm down" not in resp.json()["detail"].lower()


@needs_new_main
def test_chat_unexpected_error_returns_500_without_internal_details(client, clear_sessions, monkeypatch):
    _register()
    monkeypatch.setattr(main, "ConversationalRAG", _FailsOnInvokeUnexpectedly)
    resp = client.post("/chat", json={"session_id": SESSION_ID, "message": "hi"})
    assert resp.status_code == 500
    assert "secret" not in resp.json()["detail"].lower()


@needs_new_main
def test_failed_chat_does_not_add_to_history(client, clear_sessions, monkeypatch):
    _register()
    monkeypatch.setattr(main, "ConversationalRAG", _FailsOnInvokeUnexpectedly)
    client.post("/chat", json={"session_id": SESSION_ID, "message": "hi"})
    assert _store()[SESSION_ID] == []