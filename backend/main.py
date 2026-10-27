import logging
from fastapi import File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from config import app, CORS_ORIGINS, MAX_UPLOAD_MB
import services

logger = logging.getLogger("adaptive_rag.api")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

class ChatRequest(BaseModel):
    query: str = Field(min_length=2, max_length=4000)
    strategy: str = Field(default="auto", pattern="^(auto|way1|way2)$")
    latency_sensitive: bool = False

@app.get("/health")
def health():
    return {"status": "ok", "service": "adaptive-explainable-hybrid-rag"}

@app.get("/storage-stats")
def storage_stats():
    try:
        return services.storage_stats()
    except Exception as exc:
        logger.exception("Could not get storage statistics")
        raise HTTPException(status_code=503, detail="Storage statistics are temporarily unavailable") from exc

@app.post("/upload")
async def handle_upload(file: UploadFile = File(...)):
    filename = (file.filename or "uploaded-file").replace("\\", "/").split("/")[-1]
    allowed = {".pdf", ".docx", ".txt", ".md", ".csv", ".json", ".py", ".js", ".jsx", ".html", ".css", ".xml", ".png", ".jpg", ".jpeg", ".webp", ".mp3", ".wav", ".mp4"}
    import os
    from pathlib import Path
    if Path(filename).suffix.lower() not in allowed:
        raise HTTPException(status_code=415, detail="Unsupported file type")
    content = await file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_MB} MB limit")
    try:
        result = await services.ingest_file(content, filename, file.content_type or "application/octet-stream")
        return {"status": "success", **result}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Upload/indexing failed for %s", filename)
        raise HTTPException(status_code=500, detail="Upload indexing failed. Check backend logs and service credentials.") from exc

@app.post("/chat")
async def handle_chat(payload: ChatRequest):
    try:
        return await services.retrieve(payload.query.strip(), payload.strategy, payload.latency_sensitive)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Chat request failed")
        raise HTTPException(status_code=503, detail="Retrieval failed. Check database connectivity and backend logs.") from exc

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)
