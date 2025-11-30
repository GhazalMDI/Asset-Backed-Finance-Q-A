from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv

import numpy as np
import json
import os
import time

INDEX_PATH = 'Data/index.npy'
META_PATH = 'Data/meta.jsonl'
DOCS_PATH = 'Data/docs.jsonl'
OUTPUT_PATH = 'predictions.jsonl'

load_dotenv()

client = OpenAI(base_url="http://localhost:1234/v1/",api_key="llma3.2-1b-readcv")

embedding = np.load(INDEX_PATH)

meta = []
with open(META_PATH,"r",encoding="utf-8") as f:
    for line in f:
        meta.append(json.loads(line))

docs = {}
with open(DOCS_PATH,"r",encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)
        docs[doc['doc_id']]=doc

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
def question_to_embedding(question):
    return model.encode([question])[0]

def retrive_top_k(question,k=3):
    q_emb = question_to_embedding(question).reshape(1,-1)
    sims = cosine_similarity(q_emb,embedding).flatten()

    topk_idx = np.argsort(sims)[::-1][:k]
    results = []
    for idx in topk_idx:
        m = meta[idx]
        doc = docs[m["doc_id"]]
        sent_text = doc["sentences"][m["sent_index"]]
        results.append({
            "doc_id":m["doc_id"],
            "sent_index":m["sent_index"],
            "sentence":sent_text,
            "score":float(sims[idx])
        })
    return results


def generate_answer_llm(question,top_sentences):
    try:
        context = "\n".join([f"- {s['sentence']}" for s in top_sentences])
        print("hi ll 1")
        prompt = f"""
        Use the following context to answer the question concisely.
        Include citations if possible in format {{doc_id, sent_index}}.

        Context:
            {context}
        Querstion:
            {question}
    Answer(short,grounded):
    """
        print("hi ll 2")
        resp = client.chat.completions.create(
            model="llma3.2-1b-readcv",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        answer_text = resp.choices[0].message.content.strip()
    except Exception as e:
        print("LLM ERROR:", e)
        answer_text = "".join([s['sentence'] for s in top_sentences])

    citations = [{"doc_id": s["doc_id"],"sent_start":s["sent_index"],"sent_end": s["sent_index"]} for s in top_sentences]
    retrieved_doc_ids = [s["doc_id"] for s in top_sentences]
    confidence = float(np.mean([s["score"] for s in top_sentences]))

    return {
        "answer":answer_text,
        "citations":citations,
        "retrieved_doc_ids":retrieved_doc_ids,
        "confidence":confidence
    }



# if __name__ == "__main__":
#     test_questions = [
#         {"q_id": "T001", "question": "Are receivables from obligors rated below B- eligible?"},
#         {"q_id": "T002", "question": "What is the maximum delinquency for eligible receivables?"}
#     ]
#     predictions = []
#     for q in test_questions:
#         start_time = time.time()
#         top_k = retrive_top_k(q["question"], k=3)
#         result = generate_answer_llm(q["question"], top_k)
#         latency_ms = int((time.time() - start_time) * 1000)
#         prediction = {"q_id": q["q_id"], **result, "latency_ms": latency_ms}
#         predictions.append(prediction)

#     with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
#         for p in predictions:
#             f.write(json.dumps(p) + "\n")

#     print(f"Predictions saved to {OUTPUT_PATH}")