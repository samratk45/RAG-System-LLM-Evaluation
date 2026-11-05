import os
import sys
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from multi_doc.src.document_ingestion.data_ingestion import ChatIngestor
from multi_doc.src.document_chat.retrieval import ConversationalRAG

load_dotenv(find_dotenv(usecwd=True), override=True)

_here = Path(__file__).resolve().parent

PROJECT_ROOT = next((p for p in (_here, _here.parent, Path.cwd()) if (p / "data").is_dir()), _here)
DATA_DIR = PROJECT_ROOT / "data"

TEST_FILES = [
    "NIPS-2017-attention-is-all-you-need-Paper.pdf",
]
SUPPORTED = {".pdf", ".docx", ".txt", ".md"}


def find_test_files() -> list[Path]:
    found = [DATA_DIR / name for name in TEST_FILES if (DATA_DIR / name).exists()]
    for name in TEST_FILES:
        if not (DATA_DIR / name).exists():
            print(f"File does not exist: {DATA_DIR / name}")

    if not found:
        candidates = sorted(p for p in DATA_DIR.glob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
        if candidates:
            print(f"Using {candidates[0].name} instead.")
            found = candidates[:1]
    return found


def test_document_ingestion_and_rag():
    uploaded_files = []
    try:
        for path in find_test_files():
            uploaded_files.append(open(path, "rb"))

        if not uploaded_files:
            print("No valid files to upload. Put a PDF/MD/TXT/DOCX file in the data/ folder.")
            sys.exit(1)

        ci = ChatIngestor(temp_base="data", faiss_base="faiss_index", use_session_dirs=True)
        ci.built_retriver(uploaded_files)

        for f in uploaded_files:
            try:
                f.close()
            except Exception:
                pass

        session_id = ci.session_id
        index_dir = str(ci.faiss_dir)  

        rag = ConversationalRAG(session_id=session_id)
        rag.load_retriever_from_faiss(
            index_path=index_dir,
            index_name=os.getenv("FAISS_INDEX_NAME", "index"),
        )

        chat_history = []
        print("\nType 'exit' to quit the chat.\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting chat.")
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "q", ":q"}:
                print("Goodbye!")
                break

            try:
                answer = rag.invoke(user_input, chat_history=chat_history)
            except Exception as e:
                print(f"Assistant error: {str(e)[:300]}")
                continue

            print("Assistant:", answer)

            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=answer))

    except SystemExit:
        raise
    except Exception as e:
        print(f"Test failed: {str(e)}")
        sys.exit(1)
    finally:
        for f in uploaded_files:
            try:
                f.close()
            except Exception:
                pass


if __name__ == "__main__":
    test_document_ingestion_and_rag()