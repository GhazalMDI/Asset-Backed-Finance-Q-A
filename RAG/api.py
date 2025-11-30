from fastapi import APIRouter,HTTPException
from pydantic import BaseModel
import time,uuid
import RAG.rag_core as rag   

router = APIRouter()

class AnswerRequest(BaseModel):
    question:str
    k:int=3

@router.post("/v1/answer")
def answer(req:AnswerRequest):
    if not rag.ready:
        raise HTTPException(status_code=503,detail="Service not Ready")
    
    start_time = time.time()
    request_id = str(uuid.uuid4())

    top_k = rag.retrieve_top_k(req.question, req.k)
    answer_text = rag.generate_answer_llm(req.question, top_k)

    latency_ms = int((time.time() - start_time) * 1000)

    citations = [{"doc_id": s["doc_id"], "sent_start": s["sent_index"]} for s in top_k]
    retrieved_doc_ids = [s["doc_id"] for s in top_k]
    confidence = float(sum([s["score"] for s in top_k]) / len(top_k))

    return{
        "answer":answer_text,
        "citation":citations,
        "retrieved":retrieved_doc_ids,
        "confidence":confidence,
        "latency_ms":latency_ms,
        "request_id":request_id
    }

@router.get("/healthz")
def healthz():
    return {"status":"ok"}

@router.get("/readyz")
def readyz():
    return {"ready": rag.ready}
