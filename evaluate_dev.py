import json
import RAG.rag_core as rag

with open("Data/dev_qa.jsonl", "r", encoding="utf-8") as f:
    dev_set = [json.loads(l) for l in f]

results = []

for q in dev_set:
    topk = rag.retrieve_top_k(q["question"], k=3)
    answer = rag.generate_answer_llm(q["question"], topk)
    results.append({
        "q_id": q["q_id"],
        "predicted_answer": answer["answer"],
        "citations": answer["citations"]
    })

with open("Data/dev_predictions.jsonl", "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("Dev predictions saved to dev_predictions.jsonl")
