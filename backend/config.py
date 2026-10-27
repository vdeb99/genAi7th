import os
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.rate_limiters import InMemoryRateLimiter

load_dotenv()
app = FastAPI(title="Adaptive Explainable Hybrid RAG", version="2.0.0")
rate_limiter = InMemoryRateLimiter(requests_per_second=float(os.getenv("GEMINI_REQUESTS_PER_SECOND", "0.2")), max_bucket_size=1)
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "multimodal_collection_v3")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
CORS_ORIGINS = [x.strip() for x in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",") if x.strip()]
