from fastapi import APIRouter,HTTPException,Request
from pydantic import BaseModel
import time,uuid
import RAG.rag_core as rag   
import logging, json, hashlib
import os
import RAG.rag_core as rag_core


logger = logging.getLogger("rag_service")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "1000"))



router = APIRouter()

class AnswerRequest(BaseModel):
    question:str
    k:int=3


@router.get("/healthz")
def healthz():
    return {"status":"ok"}

@router.get("/readyz")
def readyz():
    if getattr(rag_core, "ready", False):
        return {"ready": True}
    else:
        raise HTTPException(status_code=503, detail="Not ready")


@router.post("/v1/answer")
async  def answer(req:AnswerRequest, request: Request):
    if not isinstance(req.question, str):
        raise HTTPException(status_code=400, detail="Invalid JSON")
    if len(req.question) > MAX_QUESTION_CHARS:
        raise HTTPException(status_code=413, detail=f"Question exceeds max length {MAX_QUESTION_CHARS}")

    if not getattr(rag_core, "ready", False):
        raise HTTPException(status_code=503, detail="Service not ready")

    
    start_time = time.time()
    request_id = str(uuid.uuid4())

    t_retr_start = time.time()
    try:
        top_k = rag.retrieve_top_k(req.question, req.k)
    except Exception as e:
        logger.exception("retrieval error")
        raise HTTPException(status_code=500, detail=str(e))

    t_retr_end = time.time()

    t_gen_start = time.time()
    answer_text = rag.generate_answer_llm(req.question, top_k)
    t_gen_end = time.time()

    latency_ms = int((time.time() - start_time) * 1000)

    citations = [{"doc_id": s["doc_id"], "sent_start": s["sent_index"], "sent_end": s["sent_index"]} for s in top_k]
    retrieved_doc_ids = [s["doc_id"] for s in top_k]
    confidence = float(sum([s["score"] for s in top_k]) / len(top_k))

    prediction_entry = {
    "q_id": str(uuid.uuid4()),  
    "answer": answer_text,
    "citations": citations,
    "retrieved_doc_ids": retrieved_doc_ids,
    "confidence": confidence,
    "latency_ms": latency_ms
    }
    
    with open("predictions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(prediction_entry, ensure_ascii=False) + "\n")


    log = {
        "timestamp": time.time(),
        "level": "INFO",
        "request_id": request_id,
        "question_hash": hashlib.md5(req.question.encode()).hexdigest(),
        "latency_ms": latency_ms,
        "stage_durations_ms": {
            "retrieval": int((t_retr_end - t_retr_start) * 1000),
            "generation": int((t_gen_end - t_gen_start) * 1000)
        },
        "retrieved_k": len(top_k)
    }
    logger.info(json.dumps(log, ensure_ascii=False))

    return{
        "answer":answer_text,
        "citations":citations,
        "retrieved":retrieved_doc_ids,
        "confidence":confidence,
        "latency_ms":latency_ms,
        "request_id":request_id
    }


