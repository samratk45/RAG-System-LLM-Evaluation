from pydantic import BaseModel, Field
from typing import Annotated, List, Optional
from enum import Enum

# session ids become folder names (faiss_index/<session_id>), so only allow safe characters.
# This blocks values like "../../etc" from reaching the file system.
SessionId = Annotated[str, Field(pattern=r"^[A-Za-z0-9_\-]{1,64}$")]


class ChatAnswer(BaseModel):
    """Validate chat answer type and length."""
    # Generous upper bound: a valid long answer should not fail the whole request.
    answer: Annotated[str, Field(min_length=1, max_length=20000)]


class PromptType(str, Enum):
    CONTEXTUALIZE_QUESTION = "contextualize_question"
    CONTEXT_QA = "context_qa"


class UploadResponse(BaseModel):
    session_id: str
    indexed: bool
    message: Optional[str] = None
    files_indexed: List[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    session_id: SessionId
    message: Annotated[str, Field(min_length=1, max_length=2000)]


class SourceRef(BaseModel):
    filename: str
    page: Optional[int] = None  # 1-based page number when the source has pages


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceRef] = Field(default_factory=list)