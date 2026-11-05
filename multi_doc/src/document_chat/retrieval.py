import json
import os
from operator import itemgetter
from pathlib import Path
from typing import List, Optional, Dict, Any

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch
from langchain_community.vectorstores import FAISS

from multi_doc.utils.model_loader import ModelLoader
from multi_doc.exception.custom_exception import DocumentPortalException
from multi_doc.logger import GLOBAL_LOGGER as log
from multi_doc.prompts.prompt_library import PROMPT_REGISTRY
from multi_doc.model.models import PromptType, ChatAnswer
from pydantic import ValidationError


class ConversationalRAG:
    """
    LCEL-based Conversational RAG with lazy retriever initialization.

    Usage:
        rag = ConversationalRAG(session_id="abc")
        rag.load_retriever_from_faiss(index_path="faiss_index/abc")
        answer = rag.invoke("What is ...?", chat_history=[])
    """

    def __init__(self, session_id: Optional[str], retriever=None, model_loader: Optional[ModelLoader] = None):
        try:
            self.session_id = session_id

            
            self.model_loader = model_loader or ModelLoader()
            self.cfg = self.model_loader.config

            self.llm = self._load_llm()
            self.contextualize_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.CONTEXTUALIZE_QUESTION.value
            ]
            self.qa_prompt: ChatPromptTemplate = PROMPT_REGISTRY[PromptType.CONTEXT_QA.value]

            self.retriever = retriever
            self.chain = None
            if self.retriever is not None:
                self._build_lcel_chain()

            log.info("ConversationalRAG initialized", session_id=self.session_id)
        except DocumentPortalException:
            raise
        except Exception as e:
            log.error("Failed to initialize ConversationalRAG", error=str(e))
            raise DocumentPortalException("Initialization error in ConversationalRAG", e) from e


    def load_retriever_from_faiss(
        self,
        index_path: str,
        k: Optional[int] = None,
        index_name: str = "index",
        search_type: Optional[str] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        search_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Load FAISS vectorstore from disk and build retriever + LCEL chain.
        Anything not passed in comes from the `retriever` section of config.yaml.
        """
        try:
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"FAISS index directory not found: {index_path}")

            rt = self.cfg.get("retriever", {})
            k = k or rt.get("top_k", 8)
            search_type = search_type or rt.get("search_type", "mmr")
            fetch_k = fetch_k or rt.get("fetch_k", 30)
            lambda_mult = lambda_mult if lambda_mult is not None else rt.get("lambda_mult", 0.7)

            self._check_embedding_model(index_path)

            embeddings = self.model_loader.load_embeddings()
            vectorstore = FAISS.load_local(
                index_path,
                embeddings,
                index_name=index_name,
                allow_dangerous_deserialization=True,  # ok because we only load indexes we built
            )

            if search_kwargs is None:
                search_kwargs = {"k": k}
                if search_type == "mmr":
                    search_kwargs["fetch_k"] = max(fetch_k, k)  # MMR requires fetch_k >= k
                    search_kwargs["lambda_mult"] = lambda_mult

            self.retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
            self._build_lcel_chain()

            log.info(
                "FAISS retriever loaded successfully",
                index_path=index_path,
                index_name=index_name,
                search_type=search_type,
                k=k,
                fetch_k=search_kwargs.get("fetch_k"),
                lambda_mult=search_kwargs.get("lambda_mult"),
                session_id=self.session_id,
            )
            return self.retriever

        except DocumentPortalException:
            raise
        except Exception as e:
            log.error("Failed to load retriever from FAISS", error=str(e))
            raise DocumentPortalException("Loading error in ConversationalRAG", e) from e

    def invoke(self, user_input: str, chat_history: Optional[List[BaseMessage]] = None) -> str:
        """Invoke the LCEL pipeline."""
        try:
            if self.chain is None:
                raise DocumentPortalException(
                    "RAG chain not initialized. Call load_retriever_from_faiss() before invoke().", None
                )
            chat_history = chat_history or []
            payload = {"input": user_input, "chat_history": chat_history}
            answer = self.chain.invoke(payload)
            if not answer:
                log.warning("No answer generated", session_id=self.session_id)
                return "no answer generated."

            # Validate answer type and length using the Pydantic model
            try:
                answer = ChatAnswer(answer=str(answer)).answer
            except ValidationError as ve:
                log.error("Invalid chat answer", error=str(ve))
                raise DocumentPortalException("Invalid chat answer", ve) from ve

            log.info(
                "Chain invoked successfully",
                session_id=self.session_id,
                answer_preview=str(answer)[:150],
            )
            return answer
        except DocumentPortalException:
            raise  
        except Exception as e:
            log.error("Failed to invoke ConversationalRAG", error=str(e))
            raise DocumentPortalException("Invocation error in ConversationalRAG", e) from e

    

    def _load_llm(self):
        try:
            llm = self.model_loader.load_llm()
            if not llm:
                raise ValueError("LLM could not be loaded")
            log.info("LLM loaded successfully", session_id=self.session_id)
            return llm
        except Exception as e:
            log.error("Failed to load LLM", error=str(e))
            raise DocumentPortalException("LLM loading error in ConversationalRAG", e) from e

    def _check_embedding_model(self, index_path: str) -> None:
        """Fail early, with a clear message, if the index was built with a different embedding model."""
        meta_file = Path(index_path) / "ingested_meta.json"
        if not meta_file.exists():
            return
        try:
            stored = json.loads(meta_file.read_text(encoding="utf-8")).get("embedding_model")
        except Exception:
            return
        current = self.cfg["embedding_model"]["model_name"]
        if stored and stored != current:
            raise DocumentPortalException(
                f"Index was built with '{stored}' but config uses '{current}'. "
                "Re-ingest your documents or restore the original embedding model.",
                None,
            )

    @staticmethod
    def _format_docs(docs) -> str:
        """Label each chunk with its source so the model can cite file name and page."""
        parts = []
        for i, d in enumerate(docs, 1):
            md = getattr(d, "metadata", None) or {}
            name = md.get("filename") or Path(str(md.get("source", "unknown"))).name
            page = md.get("page")  
            label = f"{name}, page {page + 1}" if isinstance(page, int) else name
            parts.append(f"[Source {i}: {label}]\n{getattr(d, 'page_content', str(d))}")
        return "\n\n".join(parts)

    def _build_lcel_chain(self):
        try:
            if self.retriever is None:
                raise DocumentPortalException("No retriever set before building chain", None)

            
            question_rewriter = (
                {"input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser()
            )

            
            maybe_rewrite = RunnableBranch(
                (lambda x: bool(x.get("chat_history")), question_rewriter),
                itemgetter("input"),
            )

            
            retrieve_docs = maybe_rewrite | self.retriever | self._format_docs

         
            self.chain = (
                {
                    "context": retrieve_docs,
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                }
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
            )

            log.info("LCEL graph built successfully", session_id=self.session_id)
        except DocumentPortalException:
            raise
        except Exception as e:
            log.error("Failed to build LCEL chain", error=str(e), session_id=self.session_id)
            raise DocumentPortalException("Failed to build LCEL chain", e) from e