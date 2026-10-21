import ipaddress
import os
import time
from functools import lru_cache
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from rate_limit import (
    IpRateLimiter,
    RateLimitExceeded,
    RateLimitPolicy,
    RateLimitStorageFull,
)

app = FastAPI(title="Delta Workshops RAG API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def positive_int_env(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name, str(default)).strip().lower()
    if value not in {"true", "false"}:
        raise ValueError(f"{name} must be set to 'true' or 'false'.")
    return value == "true"


TRUST_PROXY_HEADERS = env_flag("TRUST_PROXY_HEADERS")
EMBEDDING_RATE_LIMIT = RateLimitPolicy(
    "embedding", positive_int_env("EMBEDDING_RATE_LIMIT_PER_MINUTE", 60)
)
LLM_RATE_LIMIT = RateLimitPolicy(
    "llm", positive_int_env("LLM_RATE_LIMIT_PER_MINUTE", 5)
)
ip_rate_limiter = IpRateLimiter()


def get_client_ip(request: Request) -> str:
    if TRUST_PROXY_HEADERS:
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            try:
                return str(ipaddress.ip_address(real_ip.strip()))
            except ValueError:
                pass
    return request.client.host if request.client else "unknown"


def enforce_rate_limits(request: Request, *policies: RateLimitPolicy) -> None:
    try:
        ip_rate_limiter.enforce(get_client_ip(request), policies)
    except RateLimitExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Too many {exc.bucket} requests from this IP. "
                f"Try again in {exc.retry_after} seconds."
            ),
            headers={
                "Retry-After": str(exc.retry_after),
                "X-RateLimit-Limit": str(exc.limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(exc.reset_at),
            },
        ) from exc
    except RateLimitStorageFull as exc:
        raise HTTPException(
            status_code=503,
            detail="Rate limiting is temporarily at capacity. Please retry shortly.",
            headers={"Retry-After": "60"},
        ) from exc


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def encode_texts(texts: list[str]) -> list[list[float]]:
    try:
        return get_model().encode(texts, normalize_embeddings=True).tolist()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not load the Hugging Face embedding model: {exc}",
        ) from exc


def embed(text: str) -> list[float]:
    return encode_texts([text])[0]


def split_into_chunks(words: list[str], chunk_size: int, overlap: int) -> list[str]:
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + chunk_size]))
        if start + chunk_size >= len(words):
            break
    return chunks


class ChunkRequest(BaseModel):
    text: Annotated[str, Field(max_length=10000)]
    chunk_size: Annotated[int, Field(ge=1, le=1000)]
    overlap: Annotated[int, Field(ge=0, le=999)]
    query: str | None = Field(default=None, min_length=1, max_length=200)


class EmbedRequest(BaseModel):
    text: Annotated[str, Field(min_length=1, max_length=200)]


class SimilarityRequest(BaseModel):
    text_a: Annotated[str, Field(min_length=1, max_length=100)]
    text_b: Annotated[str, Field(min_length=1, max_length=100)]


class GenerateRequest(BaseModel):
    system_prompt: Annotated[str, Field(min_length=1, max_length=2000)]
    user_prompt: Annotated[str, Field(min_length=1, max_length=1000)]
    context: Annotated[str, Field(max_length=10000)]
    chunk_size: Annotated[int, Field(ge=12, le=120)] = 48
    overlap: Annotated[int, Field(ge=0, le=40)] = 10


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "embedding_model": MODEL_NAME}


@app.post("/api/chunk")
def chunk_text(request: ChunkRequest, http_request: Request) -> dict:
    if request.overlap >= request.chunk_size:
        raise HTTPException(
            status_code=422,
            detail="Overlap must be smaller than the chunk size.",
        )

    started = time.perf_counter()
    words = request.text.split()
    chunks = split_into_chunks(words, request.chunk_size, request.overlap)
    elapsed_ms = (time.perf_counter() - started) * 1000
    similarities: list[float] = []
    if request.query and chunks:
        enforce_rate_limits(http_request, EMBEDDING_RATE_LIMIT)
        vectors = encode_texts([request.query, *chunks])
        query_vector, *chunk_vectors = vectors
        similarities = [
            round(sum(a * b for a, b in zip(query_vector, vector)), 4)
            for vector in chunk_vectors
        ]
    return {
        "chunks": chunks,
        "count": len(chunks),
        "similarities": similarities,
        "processing_ms": round(elapsed_ms, 3),
    }


@app.post("/api/embed")
def create_embedding(request: EmbedRequest, http_request: Request) -> dict:
    enforce_rate_limits(http_request, EMBEDDING_RATE_LIMIT)
    vector = embed(request.text)
    return {"vector": vector, "dimensions": len(vector), "model": MODEL_NAME}


@app.post("/api/similarity")
def compare_sentences(request: SimilarityRequest, http_request: Request) -> dict:
    enforce_rate_limits(http_request, EMBEDDING_RATE_LIMIT)
    vector_a, vector_b = encode_texts([request.text_a, request.text_b])
    cosine_similarity = sum(a * b for a, b in zip(vector_a, vector_b))
    score = max(-1.0, min(1.0, cosine_similarity))

    if score >= 0.75:
        interpretation = "These sentences are closely related in meaning."
    elif score >= 0.45:
        interpretation = "There’s some shared meaning between these sentences."
    else:
        interpretation = "These sentences express fairly different ideas."
    return {
        "similarity": round(score, 4),
        "interpretation": interpretation,
        "model": MODEL_NAME,
    }


@app.post("/api/generate")
def generate_answer(
    request: GenerateRequest, http_request: Request
) -> dict[str, list[str] | str]:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Generation needs a Google API key. Set GOOGLE_API_KEY and restart the API.",
        )
    if request.overlap >= request.chunk_size:
        raise HTTPException(
            status_code=422,
            detail="Overlap must be smaller than the chunk size.",
        )

    enforce_rate_limits(
        http_request, EMBEDDING_RATE_LIMIT, LLM_RATE_LIMIT
    )

    try:
        from google import genai
        from google.genai import types

        chunks = split_into_chunks(
            request.context.split(), request.chunk_size, request.overlap
        )

        if chunks:
            vectors = encode_texts([request.user_prompt, *chunks])
            query_vector, *chunk_vectors = vectors
            ranked_chunks = sorted(
                zip(chunks, chunk_vectors),
                key=lambda item: sum(
                    a * b for a, b in zip(query_vector, item[1])
                ),
                reverse=True,
            )
            retrieved_chunks = [chunk for chunk, _ in ranked_chunks[:3]]
        else:
            retrieved_chunks = []

        retrieved_context = "\n\n".join(retrieved_chunks)
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash"),
            contents=(
                f"Context:\n{retrieved_context}"
                f"\n\nUser question:\n{request.user_prompt}"
            ),
            config=types.GenerateContentConfig(
                system_instruction=request.system_prompt
            ),
        )
        if not response.text:
            raise HTTPException(
                status_code=502,
                detail="Google returned an empty answer. Please try again.",
            )
        return {"answer": response.text, "retrieved_chunks": retrieved_chunks}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Google generation failed: {exc}",
        ) from exc
