import os
import shutil
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.atlas_rag import get_rag_model

app = FastAPI(
    title="Atlas PDF Engine",
    description="Backend API for Atlas PDF Engine & PDF Library Management",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prevent static caching for live dev reloads
@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

current_model_mode = "live"
rag_engine = get_rag_model(current_model_mode)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")
os.makedirs(DATA_DIR, exist_ok=True)

class QueryRequest(BaseModel):
    prompt: Optional[str] = "Summarize the primary context of this document."
    context_type: Optional[str] = "notebook_lm"
    page_selection: Optional[str] = None
    top_k: Optional[int] = 4

class ToggleDocumentRequest(BaseModel):
    filename: str
    is_active: bool

class ToggleModelRequest(BaseModel):
    mode: str  # "mock" or "live"

@app.get("/api/status")
def get_status():
    return {
        "status": "online",
        "active_mode": current_model_mode,
        "pipeline": rag_engine.get_pipeline_info()
    }

@app.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    file_path = os.path.join(DATA_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    ingest_result = rag_engine.ingest_document(file_path, file.filename)
    return {
        "status": "success",
        "filename": file.filename,
        "ingest_result": ingest_result
    }

@app.post("/api/query")
def process_query(req: QueryRequest):
    return rag_engine.query(
        prompt=req.prompt or "Provide a summary.",
        context_type=req.context_type or "notebook_lm",
        page_selection=req.page_selection,
        top_k=req.top_k or 4
    )

# PDF Library Management Endpoints
@app.get("/api/documents")
def list_library_documents():
    return {
        "status": "success",
        "documents": rag_engine.get_all_documents()
    }

@app.post("/api/documents/toggle")
def toggle_document_active(req: ToggleDocumentRequest):
    res = rag_engine.toggle_document_active(req.filename, req.is_active)
    return res

@app.delete("/api/documents/{filename}")
def delete_library_document(filename: str):
    file_path = os.path.join(DATA_DIR, filename)
    res = rag_engine.delete_document_context(filename, file_path)
    return res

@app.get("/api/contexts")
def list_contexts():
    return {
        "status": "success",
        "contexts": rag_engine.get_all_contexts()
    }

@app.delete("/api/contexts/{chunk_id}")
def delete_single_context(chunk_id: str):
    res = rag_engine.delete_context(chunk_id)
    return res

@app.post("/api/contexts/clear")
def clear_all_contexts():
    res = rag_engine.clear_all_contexts()
    return res

@app.post("/api/toggle-model")
def toggle_model(req: ToggleModelRequest):
    global current_model_mode, rag_engine
    if req.mode.lower() not in ["mock", "live"]:
        raise HTTPException(status_code=400, detail="Mode must be 'mock' or 'live'")
    current_model_mode = req.mode.lower()
    rag_engine = get_rag_model(current_model_mode)
    return {
        "status": "success",
        "active_mode": current_model_mode,
        "message": f"Switched engine to {current_model_mode.upper()} mode."
    }

# Mount static directories
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
    css_dir = os.path.join(FRONTEND_DIR, "css")
    js_dir = os.path.join(FRONTEND_DIR, "js")
    if os.path.exists(css_dir):
        app.mount("/css", StaticFiles(directory=css_dir), name="css")
    if os.path.exists(js_dir):
        app.mount("/js", StaticFiles(directory=js_dir), name="js")

@app.get("/")
def serve_index():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    return {"message": "Backend API is running. Frontend files at /frontend."}
