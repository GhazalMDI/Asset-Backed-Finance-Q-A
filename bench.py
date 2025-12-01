import json, time, psutil, platform
from statistics import median
import RAG.rag_core as rag
import os

with open("Data/test_q.jsonl","r",encoding="utf-8") as f:
    questions = [json.loads(l)["question"] for l in f]
    print("hiii")

latencies, retrievals, generations = [], [], []
errors = 0
print("hi 2")
for q in questions:
    print("hi 3")
    start = time.time()
    try:
        print("hi 4")
        t0 = time.time()
        topk = rag.retrieve_top_k(q)
        t1 = time.time()
        ans = rag.generate_answer_llm(q, topk)
        t2 = time.time()
        retrievals.append((t1-t0)*1000)
        generations.append((t2-t1)*1000)
        latencies.append((t2-start)*1000)
    except Exception as e:
        print("Error processing question:", e)
        errors += 1

# اگر هیچ سوالی موفق نبود، لیست‌ها خالی است -> مقدار صفر
if latencies:
    p50_latency = median(latencies)
    p95_latency = sorted(latencies)[int(0.95*len(latencies))]
    mean_retrieval = sum(retrievals)/len(retrievals)
    p95_retrieval = sorted(retrievals)[int(0.95*len(retrievals))]
    mean_generation = sum(generations)/len(generations)
    p95_generation = sorted(generations)[int(0.95*len(generations))]
else:
    p50_latency = p95_latency = mean_retrieval = p95_retrieval = mean_generation = p95_generation = 0

bench = {
    "hardware": {
        "cpu": platform.processor(),
        "ram_gb": psutil.virtual_memory().total / 1e9,
        "gpu": "none"
    },
    "n": len(questions),
    "latency_ms": {"p50": p50_latency, "p95": p95_latency},
    "error_rate": errors/len(questions) if questions else 0,
    "retrieval_ms": {"mean": mean_retrieval, "p95": p95_retrieval},
    "generation_ms": {"mean": mean_generation, "p95": p95_generation}
}

# مطمئن شو مسیر وجود دارد
os.makedirs("Data", exist_ok=True)
with open("Data/bench_results.json","w",encoding="utf-8") as f:
    json.dump(bench,f,indent=4)

print("Bench results saved to Data/bench_results.json")
