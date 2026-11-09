import io

import pytest

import main
from multi_doc.exception.custom_exception import DocumentPortalException


def _txt(name: str = "note.txt", content: bytes = b"hello world"):
    return ("files", (name, io.BytesIO(content), "text/plain"))


class _FailsWithPortalError:
    def __init__(self, *args, **kwargs):
        self.session_id = "sess_test"

    def built_retriver(self, *args, **kwargs):
        raise DocumentPortalException("boom /secret/path", None)


class _FailsUnexpectedly:
    def __init__(self, *args, **kwargs):
        self.session_id = "sess_test"

    def built_retriver(self, *args, **kwargs):
        raise RuntimeError("secret internal detail")


def test_upload_success_returns_session_and_indexed(client, clear_sessions, stub_ingestor, tmp_dirs):
    resp = client.post("/upload", files=[_txt()])
    assert resp.status_code == 200
    data = resp.json()
    assert data["indexed"] is True
    assert data["session_id"]


def test_upload_registers_empty_history_for_new_session(client, clear_sessions, stub_ingestor, tmp_dirs):
    resp = client.post("/upload", files=[_txt()])
    session_id = resp.json()["session_id"]
    store = main.HISTORY if hasattr(main, "HISTORY") else main.SESSIONS
    assert store[session_id] == []


def test_upload_accepts_several_supported_types(client, clear_sessions, stub_ingestor, tmp_dirs):
    files = [_txt("a.txt"), _txt("b.md"), _txt("c.pdf"), _txt("d.docx")]
    assert client.post("/upload", files=files).status_code == 200


def test_upload_without_files_returns_422(client, clear_sessions, stub_ingestor):
    assert client.post("/upload").status_code == 422


def test_upload_unsupported_type_returns_400(client, clear_sessions, stub_ingestor, tmp_dirs):
    resp = client.post("/upload", files=[_txt("malware.exe")])
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.json()["detail"]


def test_upload_file_without_extension_returns_400(client, clear_sessions, stub_ingestor, tmp_dirs):
    resp = client.post("/upload", files=[_txt("README")])
    assert resp.status_code == 400


def test_upload_one_bad_file_rejects_the_whole_batch(client, clear_sessions, stub_ingestor, tmp_dirs):
    resp = client.post("/upload", files=[_txt("good.txt"), _txt("bad.exe")])
    assert resp.status_code == 400
    assert "bad.exe" in resp.json()["detail"]


@pytest.mark.skipif(not hasattr(main, "MAX_FILES_PER_UPLOAD"), reason="needs the updated main.py")
def test_upload_too_many_files_returns_400(client, clear_sessions, stub_ingestor, tmp_dirs):
    files = [_txt(f"f{i}.txt") for i in range(main.MAX_FILES_PER_UPLOAD + 1)]
    resp = client.post("/upload", files=files)
    assert resp.status_code == 400
    assert "at most" in resp.json()["detail"]


@pytest.mark.skipif(not hasattr(main, "HISTORY"), reason="needs the updated main.py")
def test_upload_portal_error_returns_422_without_internal_details(client, clear_sessions, monkeypatch, tmp_dirs):
    monkeypatch.setattr(main, "ChatIngestor", _FailsWithPortalError)
    resp = client.post("/upload", files=[_txt()])
    assert resp.status_code == 422
    detail = resp.json()["detail"].lower()
    assert "boom" not in detail
    assert "secret" not in detail


@pytest.mark.skipif(not hasattr(main, "HISTORY"), reason="needs the updated main.py")
def test_upload_unexpected_error_returns_500_without_internal_details(client, clear_sessions, monkeypatch, tmp_dirs):
    monkeypatch.setattr(main, "ChatIngestor", _FailsUnexpectedly)
    resp = client.post("/upload", files=[_txt()])
    assert resp.status_code == 500
    assert "secret" not in resp.json()["detail"].lower()


@pytest.mark.skipif(not hasattr(main, "HISTORY"), reason="needs the updated main.py")
def test_failed_upload_does_not_create_a_session(client, clear_sessions, monkeypatch, tmp_dirs):
    monkeypatch.setattr(main, "ChatIngestor", _FailsWithPortalError)
    client.post("/upload", files=[_txt()])
    assert main.HISTORY == {}