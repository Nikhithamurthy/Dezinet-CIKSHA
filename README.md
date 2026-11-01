# Dezinet CIKSHA: AI Knowledge Engine & Enterprise Search

Production-grade RAG system with hybrid retrieval, streaming responses,
guardrails and observability. Built for the Dezinet AI/ML program.

## Tech stack
Python 3.11, FastAPI, Pydantic v2, PostgreSQL + pgvector, Redis,
LangChain/LangGraph, RAGAS, Langfuse, Docker Compose.

## Setup
```
uv sync
copy .env.example .env
uv run uvicorn app.main:app --reload
```
API docs: http://127.0.0.1:8000/docs

## Quality checks
```
uv run ruff check .
uv run mypy app
uv run pytest
```

## Progress
- [x] Task 01: Project setup, typing, linting, tests
- [x] Task 02: Async FastAPI + SSE streaming
- [ ] Tasks 03-15: in progress