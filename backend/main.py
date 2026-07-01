from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
import chromadb
import os
from rag import generate_answer

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ChromaDB setup
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="pdf_documents")


# Request model
class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "RAG Backend Running"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    # Save uploaded PDF temporarily
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Extract text from PDF
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    # Guard: check if any text was extracted
    if not text.strip():
        # Clean up the file even on failure
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=400,
            detail="Could not extract text from this PDF. It may be a scanned image-based PDF."
        )

    # Chunk text WITH OVERLAP so answers aren't split across chunk boundaries
    chunk_size = 500
    overlap = 100
    chunks = []

    i = 0
    while i < len(text):
        chunk = text[i:i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        i += chunk_size - overlap  # move forward with overlap

    # Guard: check if chunks were created
    if not chunks:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=400,
            detail="No valid text chunks could be created from this PDF."
        )

    # Clear ALL previous chunks — this app works on ONE active document at a time.
    # This fixes the old bug where a new PDF's chunks were mixed with the old one.
    try:
        existing = collection.get()
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    # Store new chunks in ChromaDB
    collection.add(
        documents=chunks,
        ids=[f"{file.filename}_{i}" for i in range(len(chunks))]
    )

    # Delete the raw PDF file from disk — we've already extracted and stored
    # everything we need in ChromaDB, so there's no reason to keep it around.
    if os.path.exists(file_path):
        os.remove(file_path)

    return {
        "message": "PDF uploaded successfully",
        "chunks_created": len(chunks)
    }


@app.get("/chunks")
def get_chunks():
    data = collection.get()
    return {
        "total_chunks": len(data["documents"]),
        "documents": data["documents"][:5]
    }


@app.post("/clear")
def clear_document():
    """
    Clears the currently indexed document. Call this from the frontend
    when the user navigates away or closes the tab (e.g. via a
    `beforeunload` event using navigator.sendBeacon), so nothing lingers
    once they're done with it.
    """
    try:
        existing = collection.get()
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
        return {"message": "Document cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
def ask_question(request: QuestionRequest):

    results = collection.query(
        query_texts=[request.question],
        n_results=5
    )

    if not results["documents"] or not results["documents"][0]:
        return {
            "question": request.question,
            "answer": "No document has been uploaded yet. Please upload a PDF first."
        }

    context = "\n".join(results["documents"][0])

    answer = generate_answer(context, request.question)

    return {
        "question": request.question,
        "answer": answer
    }