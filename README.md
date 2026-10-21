# Delta Workshops — RAG, hands-on

A visual, hands-on introduction to retrieval-augmented generation, built with React, FastAPI, Hugging Face embeddings, and Google Gemini. The chunking lab compares fixed-size, sentence-preserving, and embedding-based semantic chunks against an editable question using cosine similarity. Semantic chunking starts a new group when adjacent sentence similarity drops below 0.45. Tutorial and lab titles animate from a decorative vector-like placeholder into their actual headings as they enter view.

## Run with Docker Compose

Docker Engine and the Compose plugin are required.

```bash
cp .env.example .env
```

Add your Google AI API key to `.env` to enable Gemini generation. Then build and start the app:

```bash
docker compose up --build
```

Open [http://localhost:8080](http://localhost:8080). The frontend container serves the site and proxies `/api` requests to the FastAPI container. The API is not published directly to the host. The Hugging Face model is downloaded on its first use and kept in a persistent Docker volume.

Stop the services with `docker compose down`. To also remove the downloaded model cache, use `docker compose down --volumes`.

## Run locally for development

Start the API in one terminal:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Start the frontend in another terminal:

```bash
npm install
npm run dev
```

The first embedding request downloads `sentence-transformers/all-MiniLM-L6-v2`, a compact 384-dimensional sentence embedding model. Chunk similarity scores also require this model. Set `GOOGLE_API_KEY` in the backend environment to enable generation.

`GOOGLE_MODEL` defaults to `gemini-2.5-flash`. `EMBEDDING_MODEL` defaults to `sentence-transformers/all-MiniLM-L6-v2`. `PORT` changes the Compose host port (default `8080`). Typography uses **Silkscreen** for pixel-style headings, loaded from Google Fonts to match the supplied visual reference, with **DM Sans** for body copy and numbers.

The embedding model loads when the backend starts, so the first user request does not pay the model-download and initialization delay. The API becomes healthy after the model is ready; the first startup can take longer if the model needs to be downloaded.

Embedding and generation requests are handled asynchronously while the CPU-bound embedding model runs in a worker thread. Only two embedding operations run at once per backend process by default; other requests wait without blocking the API event loop. Set `EMBEDDING_MAX_CONCURRENCY` to tune this cap for your server. This improves how the API handles concurrent connections, but does not make model inference itself faster, and the right cap depends on available CPU and memory.

The embedding endpoints (`/api/embed`, `/api/similarity`, and chunk scoring when a query is supplied) allow 60 requests per IP per minute by default. `/api/generate` allows 5 requests per IP per minute and also consumes an embedding request. Override these limits with `EMBEDDING_RATE_LIMIT_PER_MINUTE` and `LLM_RATE_LIMIT_PER_MINUTE`. The API returns `429 Too Many Requests` with a `Retry-After` header when a limit is reached. The Compose setup trusts `X-Real-IP` from its Nginx frontend; when running the API behind a different proxy, enable `TRUST_PROXY_HEADERS` only if that proxy overwrites `X-Real-IP`. Limits and the embedding concurrency cap are per process and are not shared across multiple backend replicas.

# Check Out the Website
[delta_workshops_rag](rag.deltaforce.club)
