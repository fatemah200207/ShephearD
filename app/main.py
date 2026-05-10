import os
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

try:
    from app.rag_service import (
        PDF_FOLDER,
        create_vector_db,
        list_uploaded_pdfs,
        load_qa_chain,
        smart_answer,
    )
except ModuleNotFoundError:
    from app.rag_service import (
        PDF_FOLDER,
        create_vector_db,
        list_uploaded_pdfs,
        load_qa_chain,
        smart_answer,
    )

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "15"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

app = FastAPI(title="Shepheard Hotel AI Chatbot")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

qa_chain = None


class Query(BaseModel):
    question: str


@app.on_event("startup")
def startup_event():
    global qa_chain
    os.makedirs(PDF_FOLDER, exist_ok=True)

    if list_uploaded_pdfs() and os.path.exists(os.getenv("CHROMA_DB_PATH", "vector_db")):
        qa_chain = load_qa_chain()


@app.get("/")
def home():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "pdf_count": len(list_uploaded_pdfs()),
    }


@app.get("/files")
def files():
    return {"files": list_uploaded_pdfs()}


@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    global qa_chain

    os.makedirs(PDF_FOLDER, exist_ok=True)
    saved_files = []
    rejected_files = []

    for uploaded_file in files:
        original_name = uploaded_file.filename or "uploaded.pdf"

        if not original_name.lower().endswith(".pdf"):
            rejected_files.append({"file": original_name, "reason": "Only PDF files are allowed."})
            continue

        content = await uploaded_file.read()
        if len(content) > MAX_FILE_SIZE_BYTES:
            rejected_files.append({"file": original_name, "reason": f"File is larger than {MAX_FILE_SIZE_MB} MB."})
            continue

        safe_name = Path(original_name).name
        destination = Path(PDF_FOLDER) / safe_name

        counter = 1
        while destination.exists():
            destination = Path(PDF_FOLDER) / f"{destination.stem}_{counter}{destination.suffix}"
            counter += 1

        destination.write_bytes(content)
        saved_files.append(destination.name)

    if not saved_files:
        return {
            "status": "error",
            "message": "Please upload at least one valid PDF file.",
            "saved_files": saved_files,
            "rejected_files": rejected_files,
            "files": list_uploaded_pdfs(),
        }

    message = create_vector_db()
    qa_chain = load_qa_chain()

    return {
        "status": "success",
        "message": message,
        "saved_files": saved_files,
        "rejected_files": rejected_files,
        "files": list_uploaded_pdfs(),
    }


@app.post("/chat")
def chat(query: Query):
    global qa_chain

    question = query.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if not list_uploaded_pdfs():
        return {
            "question": question,
            "answer": "Please upload one or more PDF files first, then ask your question.",
        }

    if qa_chain is None:
        create_vector_db()
        qa_chain = load_qa_chain()

    answer = smart_answer(question, qa_chain)

    return {
        "question": question,
        "answer": answer,
    }
