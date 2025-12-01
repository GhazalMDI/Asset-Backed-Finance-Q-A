from fastapi import FastAPI
from RAG.api import router as rag_router
import RAG.rag_core as rag

app = FastAPI(
    title="RAG System",
    version="1.0.0",
    description="a simple rag backend using fastApi"
)

app.include_router(rag_router,prefix="/api")

@app.get("/")
def home():
    return {"message": "RAG API is running!", "routes": ["/api/v1/answer"]}

