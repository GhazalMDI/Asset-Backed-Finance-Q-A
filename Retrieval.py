from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

import numpy as np
import json

INDEX_PATH = 'Data/index.npy'
META_PATH = 'Data/meta.jsonl'
DOCS_PATH = 'Data/docs.jsonl'

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
        client = OpenAI()
        context = "\n".join([f"- {s['sentence']}" for s in top_sentences])
        prompt = f"""
        Use the following context to answer the question concisely.
        Include citations if possible in format {{doc_id, sent_index}}.

        Context:
            {context}
        Querstion:
            {question}
    Answer(short,grounded):
    """
        resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role":"user","content":prompt}],
                temperature=0
        )
        answer_text = resp.choices[0].message.content.strip()
    except:
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
