# Adaptive Explainable Hybrid RAG

A React + FastAPI RAG app using Google Gemini, Neo4j and Qdrant. It supports two retrieval paths and explainable routing.

## Retrieval strategies

- **Way 1 — Graph-first + vector fallback:** query Neo4j first. If graph retrieval fails or is insufficient, answer using indexed Qdrant chunks alongside any graph evidence.
- **Way 2 — Parallel graph + vector:** retrieve from Neo4j and Qdrant concurrently, then combine evidence and ask Gemini to answer from that evidence.
- **Auto:** use an AHP-derived weighted decision matrix over query intent, estimated graph coverage, vector relevance, cross-document need and latency/cost preference.

The response includes routing scores, metric values, AHP weights and consistency ratio, reasons, actual retrieval path, storage statistics and retrieved chunk previews. These are explainability signals, not calibrated probabilities.

## Project layout

```text
RAG_Agents_GenAI/
├── backend/
│   ├── main.py          # FastAPI endpoints
│   ├── services.py      # ingestion, Neo4j/Qdrant retrieval, answer generation
│   ├── router.py        # AHP weights and weighted decision matrix
│   ├── evaluation.py    # confusion matrix, precision, recall and F1
│   ├── config.py        # environment configuration
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/src/App.jsx # UI and routing explanation panel
├── docker-compose.yml
├── .env.example
└── .gitignore
```

## Setup

1. Rotate any credentials previously included in a ZIP or Git history. Copy `.env.example` to `.env` and fill in a valid Google API key and a strong Neo4j password.
2. Ensure Docker Desktop is running.
3. From the project root:

   ```bash
   cp .env.example .env
   # Edit .env and set GOOGLE_API_KEY and NEO4J_PASSWORD
   docker compose up --build
   ```

4. Frontend: http://localhost:3000
5. API docs: http://localhost:8001/docs
6. Health: http://localhost:8001/health

Existing Neo4j data directories may retain the previous database password. If Neo4j rejects the new password, update the password for that existing database or back up the data before recreating local storage. Do not delete a database volume without backing up any important data.

## API examples

`POST /chat`

```json
{
  "query": "Compare the technologies discussed across the uploaded documents",
  "strategy": "auto",
  "latency_sensitive": false
}
```

Set `strategy` to `way1`, `way2`, or `auto`.

`POST /upload` accepts a multipart file. Supported extensions: PDF, DOCX, TXT, Markdown, CSV, JSON, common source files, PNG/JPEG/WebP, MP3/WAV and MP4. Text-based PDF and DOCX files are extracted and chunked. For files without extractable text, Gemini is asked for a searchable description; this depends on the selected Gemini model supporting the supplied media type.

## Evaluation

`backend/evaluation.py` provides `routing_metrics(y_true, y_pred)` for a labeled test set. Label each query with its preferred strategy, run the router, and compare the predictions. Report accuracy, per-class precision/recall/F1 and the confusion matrix. Compare against a fixed-rule baseline before claiming improvement.

## Important limitations

- AHP weights are initial expert-defined values and should be validated against a representative labeled dataset.
- Vector scores are normalized heuristics, not calibrated relevance probabilities; check behavior against the installed LangChain/Qdrant versions.
- Gemini embedding dimensionality must match the Qdrant collection vector size. This setup assumes 3072 dimensions; change the collection size if the configured embedding model returns another dimension. Existing collections cannot be resized in place.
- GraphCypherQAChain can generate database queries. Use a dedicated read-only Neo4j account for chat retrieval and keep credentials private.
- This is a development baseline, not a production security certification. Add authentication, observability, and stronger document-level access control before deployment.
