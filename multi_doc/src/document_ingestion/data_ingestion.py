from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from multi_doc.utils.model_loader import ModelLoader
from multi_doc.logger import GLOBAL_LOGGER as log
from multi_doc.exception.custom_exception import DocumentPortalException
import json
import uuid
from datetime import datetime
from multi_doc.utils.file_io import save_uploaded_files
from multi_doc.utils.document_ops import load_documents
import hashlib


def generate_session_id() -> str:
    """Generate a unique session ID with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    return f"session_{timestamp}_{unique_id}"


class ChatIngestor:
    def __init__(
        self,
        temp_base: str = "data",
        faiss_base: str = "faiss_index",
        use_session_dirs: bool = True,
        session_id: Optional[str] = None,
    ):
        try:
            self.model_loader = ModelLoader()
            self.cfg = self.model_loader.config  # values come from config.yaml

            self.use_session = use_session_dirs
            self.session_id = session_id or generate_session_id()

            self.temp_base = Path(temp_base)
            self.temp_base.mkdir(parents=True, exist_ok=True)
            self.faiss_base = Path(faiss_base)
            self.faiss_base.mkdir(parents=True, exist_ok=True)

            self.temp_dir = self._resolve_dir(self.temp_base)
            self.faiss_dir = self._resolve_dir(self.faiss_base)

            log.info(
                "ChatIngestor initialized",
                session_id=self.session_id,
                temp_dir=str(self.temp_dir),
                faiss_dir=str(self.faiss_dir),
                sessionized=self.use_session,
            )
        except Exception as e:
            log.error("Failed to initialize ChatIngestor", error=str(e))
            raise DocumentPortalException("Initialization error in ChatIngestor", e) from e

    def _resolve_dir(self, base: Path):
        if self.use_session:
            d = base / self.session_id  
            d.mkdir(parents=True, exist_ok=True)
            return d
        return base

    def _split(self, docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(docs)

        
        min_chars = self.cfg.get("chunking", {}).get("min_chunk_chars", 100)
        chunks = [c for c in chunks if len(c.page_content.strip()) >= min_chars]

        log.info("Documents split", chunks=len(chunks), chunk_size=chunk_size, overlap=chunk_overlap)
        return chunks

    def built_retriver(
        self,
        uploaded_files: Iterable,
        *,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        k: Optional[int] = None,
        search_type: Optional[str] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
    ):
        try:
           
            ch_cfg = self.cfg.get("chunking", {})
            rt_cfg = self.cfg.get("retriever", {})
            chunk_size = chunk_size or ch_cfg.get("chunk_size", 750)
            chunk_overlap = chunk_overlap if chunk_overlap is not None else ch_cfg.get("chunk_overlap", 80)
            k = k or rt_cfg.get("top_k", 8)
            search_type = search_type or rt_cfg.get("search_type", "mmr")
            fetch_k = fetch_k or rt_cfg.get("fetch_k", 30)
            lambda_mult = lambda_mult if lambda_mult is not None else rt_cfg.get("lambda_mult", 0.7)

            paths = save_uploaded_files(uploaded_files, self.temp_dir)
            docs = load_documents(paths)
            if not docs:
                raise ValueError("No valid documents loaded")

            chunks = self._split(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            if not chunks:
                raise ValueError("Documents produced no usable chunks")

            fm = FaissManager(self.faiss_dir, self.model_loader)

            texts = [c.page_content for c in chunks]
            metas = [c.metadata for c in chunks]

            vs = fm.load_or_create(texts=texts, metadatas=metas)

            
            added = fm.add_documents(chunks)
            log.info("FAISS index updated", added=added, index=str(self.faiss_dir))

            search_kwargs: Dict[str, Any] = {"k": k}
            if search_type == "mmr":
                
                search_kwargs["fetch_k"] = max(fetch_k, k)  # fetch_k must be >= k
                search_kwargs["lambda_mult"] = lambda_mult
                log.info("Using MMR search", k=k, fetch_k=search_kwargs["fetch_k"], lambda_mult=lambda_mult)

            return vs.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

        except Exception as e:
            log.error("Failed to build retriever", error=str(e))
            raise DocumentPortalException("Failed to build retriever", e) from e

    build_retriever = built_retriver


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}  



class FaissManager:
    def __init__(self, index_dir: Path, model_loader: Optional[ModelLoader] = None):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.meta_path = self.index_dir / "ingested_meta.json"
        self._meta: Dict[str, Any] = {"rows": {}}

        if self.meta_path.exists():
            try:
                self._meta = json.loads(self.meta_path.read_text(encoding="utf-8")) or {"rows": {}}
            except Exception:
                self._meta = {"rows": {}}
        self._meta.setdefault("rows", {})

        self.model_loader = model_loader or ModelLoader()
        self.emb = self.model_loader.load_embeddings()
        self.embedding_name = self.model_loader.config["embedding_model"]["model_name"]
        self.vs: Optional[FAISS] = None

    def _exists(self) -> bool:
        return (self.index_dir / "index.faiss").exists() and (self.index_dir / "index.pkl").exists()

    @staticmethod
    def _fingerprint(text: str, md: Dict[str, Any]) -> str:
       
        src = md.get("source") or md.get("file_path") or ""
        rid = md.get("row_id")
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return f"{src}::{'' if rid is None else rid}::{content_hash}"

    def _save_meta(self):
        self._meta["embedding_model"] = self.embedding_name
        self.meta_path.write_text(json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_documents(self, docs: List[Document]) -> int:
        if self.vs is None:
            raise RuntimeError("Call load_or_create() before add_documents().")

        new_docs: List[Document] = []
        for d in docs:
            key = self._fingerprint(d.page_content, d.metadata or {})
            if key in self._meta["rows"]:
                continue
            self._meta["rows"][key] = True
            new_docs.append(d)

        if new_docs:
            self.vs.add_documents(new_docs)
            self.vs.save_local(str(self.index_dir))
            self._save_meta()
        return len(new_docs)

    def load_or_create(self, texts: Optional[List[str]] = None, metadatas: Optional[List[dict]] = None):
        if self._exists():
            stored = self._meta.get("embedding_model")
            if stored != self.embedding_name:
                raise DocumentPortalException(
                    f"Index at '{self.index_dir}' was built with embedding model "
                    f"'{stored or 'unknown'}', but config uses '{self.embedding_name}'. "
                    "Delete the index folder and re-ingest.",
                    None,
                )
            self.vs = FAISS.load_local(
                str(self.index_dir),
                embeddings=self.emb,
                allow_dangerous_deserialization=True,
            )
            return self.vs

        if not texts:
            raise DocumentPortalException("No existing FAISS index and no data to create one", None)

        self.vs = FAISS.from_texts(texts=texts, embedding=self.emb, metadatas=metadatas or None)
        self.vs.save_local(str(self.index_dir))

        for i, t in enumerate(texts):
            md = metadatas[i] if metadatas else {}
            self._meta["rows"][self._fingerprint(t, md or {})] = True
        self._save_meta()
        return self.vs