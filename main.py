import os
import hmac
import hashlib
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from github_client import get_installation_token, get_repo_files, get_file_content
from chunker import chunk_file
from embedder import get_embedding, get_embeddings_batch
from qdrant_store import get_client, ensure_collection, delete_repo_chunks, upsert_chunks, search_chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

COLLECTION = os.environ["QDRANT_COLLECTION"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Al arrancar: crear colección si no existe
    client = get_client()
    ensure_collection(client, COLLECTION)
    log.info(f"Qdrant collection '{COLLECTION}' lista")
    yield


app = FastAPI(title="Tennis Doc Indexer", lifespan=lifespan)


# ─────────────────────────────────────────
# Webhook de GitHub — se dispara en cada push
# ─────────────────────────────────────────

def _verify_signature(body: bytes, signature: str) -> bool:
    secret = os.environ["WEBHOOK_SECRET"].encode()
    expected = "sha256=" + hmac.new(secret, body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


@app.post("/webhook")
async def github_webhook(request: Request, background_tasks: BackgroundTasks):
    body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256", "")

    if not _verify_signature(body, signature):
        raise HTTPException(status_code=401, detail="Firma inválida")

    event = request.headers.get("X-GitHub-Event", "")
    payload = await request.json()

    # Solo procesamos push a la rama principal
    if event == "push":
        repo_name = payload.get("repository", {}).get("full_name", "")
        ref = payload.get("ref", "")
        default_branch = payload.get("repository", {}).get("default_branch", "main")

        if ref == f"refs/heads/{default_branch}" and repo_name:
            log.info(f"Push detectado en {repo_name} — iniciando re-indexación")
            background_tasks.add_task(index_repo, repo_name)

    return JSONResponse({"status": "ok"})


# ─────────────────────────────────────────
# Indexación de un repo completo
# ─────────────────────────────────────────

async def index_repo(full_repo_name: str):
    """Indexa todos los archivos de un repo. Corre en background."""
    try:
        owner, repo = full_repo_name.split("/", 1)
        log.info(f"Indexando {full_repo_name}...")

        token = get_installation_token()
        files = get_repo_files(token, owner, repo)
        log.info(f"Encontrados {len(files)} archivos en {full_repo_name}")

        client = get_client()
        # Borrar chunks anteriores de este repo para re-indexar limpio
        delete_repo_chunks(client, COLLECTION, full_repo_name)

        total_chunks = 0
        for file_info in files:
            content = get_file_content(token, owner, repo, file_info["path"])
            if not content or not content.strip():
                continue

            chunks = chunk_file(content, file_info["path"], full_repo_name)
            if not chunks:
                continue

            texts = [c["metadata"]["embed_text"] for c in chunks]
            embeddings = await get_embeddings_batch(texts)
            upsert_chunks(client, COLLECTION, chunks, embeddings)
            total_chunks += len(chunks)

        log.info(f"Indexación completa: {full_repo_name} — {total_chunks} chunks guardados")

    except Exception as e:
        log.error(f"Error indexando {full_repo_name}: {e}")


# ─────────────────────────────────────────
# Endpoint manual para indexar un repo
# ─────────────────────────────────────────

class IndexRequest(BaseModel):
    repo: str  # formato: "org/repo-name"


@app.post("/index")
async def manual_index(req: IndexRequest, background_tasks: BackgroundTasks):
    """Dispara indexación manual de un repo."""
    background_tasks.add_task(index_repo, req.repo)
    return {"status": "indexación iniciada", "repo": req.repo}


# ─────────────────────────────────────────
# Endpoint de búsqueda semántica
# ─────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    repo: str | None = None  # si se omite, busca en todos los repos
    limit: int = 6


@app.post("/search")
async def search(req: SearchRequest):
    """Busca chunks relevantes para una pregunta."""
    query_vector = await get_embedding(req.query)
    client = get_client()
    results = search_chunks(client, COLLECTION, query_vector, req.repo, req.limit)
    return {"results": results}


# ─────────────────────────────────────────
# Health check
# ─────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}
