from __future__ import annotations
import re
import uuid
from pathlib import Path
from typing import Iterable, List
from multi_doc.logger import GLOBAL_LOGGER as log
from multi_doc.exception.custom_exception import DocumentPortalException
from multi_doc.utils.document_ops import SUPPORTED_EXTENSIONS  # single source of truth

MAX_FILE_MB = 25


def _read_bytes(uf) -> bytes:
    """Read the raw bytes from a Starlette UploadFile, a file-like object, or an adapter."""
    # Starlette UploadFile exposes the underlying buffer as .file
    if hasattr(uf, "file") and hasattr(uf.file, "read"):
        if hasattr(uf.file, "seek"):
            uf.file.seek(0)  # the stream may already have been read once
        data = uf.file.read()
    elif hasattr(uf, "read"):
        data = uf.read()
    else:
        # Adapter objects (e.g. FastAPIFileAdapter) expose getbuffer()
        buf = getattr(uf, "getbuffer", None)
        if not callable(buf):
            raise ValueError("Unsupported uploaded file object; no readable interface")
        data = buf()

    if isinstance(data, memoryview):
        data = data.tobytes()
    return data


def save_uploaded_files(uploaded_files: Iterable, target_dir: Path) -> List[Path]:
    """Save uploaded files (Streamlit/FastAPI-like) and return local paths."""
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        saved: List[Path] = []
        for uf in uploaded_files:
            # Starlette UploadFile has .filename; generic objects have .name
            name = getattr(uf, "filename", None) or getattr(uf, "name", "file")
            ext = Path(name).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                log.warning("Unsupported file skipped", filename=name)
                continue

            data = _read_bytes(uf)
            if not data:
                log.warning("Empty file skipped", filename=name)
                continue
            if len(data) > MAX_FILE_MB * 1024 * 1024:
                log.warning("File too large, skipped", filename=name, max_mb=MAX_FILE_MB)
                continue

            safe_stem = re.sub(r"[^a-zA-Z0-9_\-]", "_", Path(name).stem).lower()[:60] or "file"
            out = target_dir / f"{safe_stem}_{uuid.uuid4().hex[:6]}{ext}"
            out.write_bytes(data)

            saved.append(out)
            log.info("File saved for ingestion", uploaded=name, saved_as=str(out))
        return saved
    except Exception as e:
        log.error("Failed to save uploaded files", error=str(e), dir=str(target_dir))
        raise DocumentPortalException("Failed to save uploaded files", e) from e