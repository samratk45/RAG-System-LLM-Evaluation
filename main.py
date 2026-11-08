from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from multi_doc.exception.custom_exception import DocumentPortalException
from multi_doc.logger import GLOBAL_LOGGER as log
from multi_doc.src.document_chat.retrieval import ConversationalRAG
from multi_doc.src.document_ingestion.data_ingestion import ChatIngestor
from multi_doc.utils.document_ops import SUPPORTED_EXTENSIONS


DATA_BASE = "data"
FAISS_BASE = "faiss_index"
MAX_FILES_PER_UPLOAD = 10
MAX_HISTORY_MESSAGES = 10  


ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")
    if o.strip()
]


app = FastAPI(title="MultiDocChat", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,  
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
static_dir = BASE_DIR / "static"
templates_dir = BASE_DIR / "templates"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))



HISTORY: Dict[str, List[BaseMessage]] = {}  
RAGS: Dict[str, ConversationalRAG] = {}  



class FastAPIFileAdapter:
    """Adapt FastAPI UploadFile to a simple object with .name and .getbuffer()."""

    def __init__(self, uf: UploadFile):
        self._uf = uf
        self.name = uf.filename or "file"

    def getbuffer(self) -> bytes:
        self._uf.file.seek(0)
        return self._uf.file.read()



class UploadResponse(BaseModel):
    session_id: str
    indexed: bool
    message: str | None = None


class ChatRequest(BaseModel):
    
    session_id: str = Field(pattern=r"^[A-Za-z0-9_\-]{1,64}$")
    message: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    answer: str



def _session_exists(session_id: str) -> bool:
    """A session is valid if it is in memory OR its index is still on disk (survives server restarts)."""
    return session_id in HISTORY or (Path(FAISS_BASE) / session_id / "index.faiss").exists()


def _get_rag(session_id: str) -> ConversationalRAG:
    """Build the RAG chain once per session and reuse it (loading the LLM, embeddings and index is slow)."""
    rag = RAGS.get(session_id)
    if rag is None:
        rag = ConversationalRAG(session_id=session_id)
        
        rag.load_retriever_from_faiss(index_path=str(Path(FAISS_BASE) / session_id))
        RAGS[session_id] = rag
    return rag



@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html")


@app.post("/upload", response_model=UploadResponse)
def upload(files: List[UploadFile] = File(...)) -> UploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    if len(files) > MAX_FILES_PER_UPLOAD:
        raise HTTPException(status_code=400, detail=f"Upload at most {MAX_FILES_PER_UPLOAD} files at a time")

    bad = [f.filename or "file" for f in files if Path(f.filename or "").suffix.lower() not in SUPPORTED_EXTENSIONS]
    if bad:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {', '.join(bad)}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    try:
        ingestor = ChatIngestor(temp_base=DATA_BASE, faiss_base=FAISS_BASE, use_session_dirs=True)
        ingestor.built_retriver([FastAPIFileAdapter(f) for f in files])
        session_id = ingestor.session_id
    except DocumentPortalException as e:
        log.error("Upload failed", error=str(e))   
        raise HTTPException(
            status_code=422,
            detail="Could not process the uploaded files. Make sure they contain readable text.",
        )
    except Exception as e:
        log.error("Unexpected upload error", error=str(e))
        raise HTTPException(status_code=500, detail="Something went wrong while indexing your files.")

    HISTORY[session_id] = []
    return UploadResponse(session_id=session_id, indexed=True, message="Indexing complete")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    if not _session_exists(session_id):
        raise HTTPException(status_code=400, detail="Invalid or expired session_id. Re-upload documents.")

    try:
        rag = _get_rag(session_id)
        history = HISTORY.setdefault(session_id, [])
        answer = rag.invoke(message, chat_history=history)
    except DocumentPortalException as e:
        log.error("Chat failed", session_id=session_id, error=str(e))
        raise HTTPException(status_code=502, detail="The assistant could not answer right now. Please try again.")
    except Exception as e:
        log.error("Unexpected chat error", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail="Something went wrong.")

    history.extend([HumanMessage(content=message), AIMessage(content=answer)])
    del history[:-MAX_HISTORY_MESSAGES]  

    return ChatResponse(answer=answer)



if __name__ == "__main__":
    import uvicorn

    
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )