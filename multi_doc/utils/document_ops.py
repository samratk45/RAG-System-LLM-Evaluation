from __future__ import annotations
from pathlib import Path
from typing import Iterable, List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from multi_doc.logger import GLOBAL_LOGGER as log
from multi_doc.exception.custom_exception import DocumentPortalException
from fastapi import UploadFile

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def _loader_for(p: Path):
    ext = p.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(p))
    if ext == ".docx":
        return Docx2txtLoader(str(p))
    if ext in {".txt", ".md"}:
       
        return TextLoader(str(p), encoding="utf-8-sig")
    return None


def load_documents(paths: Iterable[Path]) -> List[Document]:
    """Load docs using the appropriate loader based on extension.

    A file that fails to load is logged and skipped, so one bad upload
    doesn't discard the rest. If nothing loads at all, an error is raised.
    """
    docs: List[Document] = []
    failed: List[str] = []
    try:
        for p in paths:
            p = Path(p)
            loader = _loader_for(p)
            if loader is None:
                log.warning("Unsupported extension skipped", path=str(p))
                continue

            try:
                loaded = loader.load()
            except Exception as e:
                log.error("Failed loading file", path=str(p), error=str(e))
                failed.append(p.name)
                continue

            
            loaded = [d for d in loaded if d.page_content and d.page_content.strip()]
            if not loaded:
                log.warning("No extractable text in file", path=str(p))
                failed.append(p.name)
                continue

           
            for d in loaded:
                d.metadata["filename"] = p.name
            docs.extend(loaded)

        if not docs:
            raise ValueError(
                "No text could be extracted from the uploaded files"
                + (f" (failed: {', '.join(failed)})" if failed else "")
            )

        if failed:
            log.warning("Some files were skipped", files=failed)
        log.info("Documents loaded", count=len(docs))
        return docs
    except Exception as e:
        log.error("Failed loading documents", error=str(e))
        raise DocumentPortalException("Error loading documents", e) from e


class FastAPIFileAdapter:
    """Adapt FastAPI UploadFile to a simple object with .name and .getbuffer()."""

    def __init__(self, uf: UploadFile):
        self._uf = uf
        self.name = uf.filename or "file"

    def getbuffer(self) -> bytes:
        self._uf.file.seek(0)
        return self._uf.file.read()