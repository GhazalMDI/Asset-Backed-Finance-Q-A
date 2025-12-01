RAG-Lite Q&A Service – Asset-Backed Finance Corpus
📍 Context

This project provides a lightweight Retrieval-Augmented Generation (RAG) Q&A system that answers questions from an asset-backed finance corpus.
It supports retrieval of relevant passages with citations and LLM-based answer generation.

🎯 Objective

Run the service locally in a Docker container and query it via HTTP API:

Start the service

Check readiness and health

Send questions to /v1/answer

Get answers with citations and confidence

🧰 How to Run
1. Build the Docker image

From the project root:

docker-compose build


Ensure your docker-compose.yml, Dockerfile, RAG/Data, and requirements.txt are present.

2. Start the service

Foreground:

docker-compose up


Detached mode:

docker-compose up -d


Stop the service:

docker-compose down

3. Check service status

Health check:

GET http://localhost:8000/healthz


Readiness check (returns 503 until index/model ready):

GET http://localhost:8000/readyz

4. Query the Q&A system

Endpoint:

POST http://localhost:8000/v1/answer
Content-Type: application/json


Request body:

{
    "question": "What is the maximum delinquency for eligible receivables?",
    "k": 3
}


Response example:

{
    "answer": "No. Receivables from obligors rated below B- are ineligible.",
    "citations": [{"doc_id": "D001", "sent_start": 1, "sent_end": 1}],
    "retrieved": ["D001", "D004", "D002"],
    "confidence": 0.81,
    "latency_ms": 124,
    "request_id": "uuid-1234"
}

🧾 Notes

RAG/Data/docs.jsonl must exist before the service is ready.

MAX_QUESTION_CHARS environment variable can limit question length (default 1000).

Logs are JSON-formatted with request IDs, latency, and retrieval/generation times.

Benchmarks can be generated using bench.py, saved to RAG/Data/bench_results.json.