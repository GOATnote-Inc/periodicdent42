# Frontend Demo Index

| Demo | Route | Backend | Inputs | Outputs | Screenshot |
| --- | --- | --- | --- | --- | --- |
| Hybrid RAG Chat | `/demos/rag-chat` | Next.js proxy → FastAPI `/api/chat` | Natural language question (textarea) | Answer, citations, guardrail outcomes, vector stats | _Pending (capture after backend QA)_ |

## Usage Notes

1. Start the backend with `make run.api` (FastAPI on `http://localhost:8000`).
2. In another shell, run `make demo` to boot the Next.js workspace (`apps/web`).
3. Visit `http://localhost:3000/demos/rag-chat` and submit a question. If the backend is offline the demo falls back to a simulated answer and shows an error banner.
4. Screenshots and recordings should be saved under `docs/screenshots/` once captured.

