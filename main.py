import hmac
import hashlib
import logging
import os
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, Depends, Header
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from io import BytesIO

from github_client import get_installation_token, get_repo_files, get_file_content, GitHubTokenExpired
from chunker import chunk_file
from embedder import get_embedding, get_embeddings_batch
from qdrant_store import get_client, ensure_collection, delete_repo_chunks, upsert_chunks, search_chunks, ping_client
from config import QDRANT_COLLECTION, WEBHOOK_SECRET, VECTOR_SIZE, JAVAPARSER_URL, INDEXER_API_KEY

import ast_parser
import graph_store
import rag_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Regex simple para validar formato org/repo
_REPO_PATTERN = re.compile(r"^[\w.-]+/[\w.-]+$")

# Detecta referencias a archivos .sql dentro de strings (Java, XML, properties, etc.)
_SQL_REF_RE = re.compile(r'["\']([^"\']*\.sql)["\']', re.IGNORECASE)


def _resolve_sql_path(java_file_path: str, sql_ref: str, all_file_paths: set[str]) -> str | None:
    """
    Resuelve una referencia relativa a un archivo SQL dentro del repo.
    Prueba: relativo al archivo Java, luego en resources, luego por nombre.
    """
    sql_ref = sql_ref.lstrip("/")

    # 1) Relativo al directorio del archivo Java
    java_dir = os.path.dirname(java_file_path)
    candidate = os.path.join(java_dir, sql_ref).replace("\\", "/")
    if candidate in all_file_paths:
        return candidate

    # 2) Bajo src/main/resources o src/test/resources (classpath)
    for prefix in ("src/main/resources/", "src/test/resources/", "resources/", ""):
        candidate = prefix + sql_ref
        if candidate in all_file_paths:
            return candidate

    # 3) Buscar por nombre exacto en cualquier parte del repo
    sql_basename = os.path.basename(sql_ref)
    for path in all_file_paths:
        if path.endswith("/" + sql_basename) or path == sql_basename:
            return path

    return None


async def _inline_sql_references(
    chunks: list[dict],
    token: str,
    owner: str,
    repo: str,
    branch: str,
    all_file_paths: set[str],
    sql_files_map: dict[str, dict],
) -> list[dict]:
    """
    Busca referencias a archivos .sql dentro de los chunks y adjunta
    el contenido SQL al mismo chunk (text + embed_text).
    Si el SQL es muy grande se trunca para no romper los límites de embedding.
    """
    _MAX_INLINE_SQL_CHARS = 6000
    for chunk in chunks:
        for match in _SQL_REF_RE.finditer(chunk["text"]):
            sql_ref = match.group(1)
            resolved = _resolve_sql_path(chunk["metadata"]["file_path"], sql_ref, all_file_paths)
            if resolved and resolved in sql_files_map:
                try:
                    sql_content = get_file_content(token, owner, repo, resolved, ref=branch)
                    if sql_content and sql_content.strip():
                        if len(sql_content) > _MAX_INLINE_SQL_CHARS:
                            sql_content = (
                                sql_content[:_MAX_INLINE_SQL_CHARS]
                                + f"\n-- ... SQL truncado ({len(sql_content)} chars originales) ... --\n"
                            )
                        sql_header = f"\n\n-- Referenced SQL: {resolved} --\n"
                        chunk["text"] += sql_header + sql_content
                        chunk["metadata"]["embed_text"] += sql_header + sql_content
                except Exception as exc:
                    log.debug(f"No se pudo leer SQL referenciado {resolved}: {exc}")
    return chunks


# ─────────────────────────────────────────
# Lifespan — validación de entorno al arrancar
# ─────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not QDRANT_COLLECTION:
        log.error("La variable de entorno QDRANT_COLLECTION no está definida")
        raise RuntimeError("QDRANT_COLLECTION es obligatoria")

    client = get_client()
    ensure_collection(client, QDRANT_COLLECTION)
    log.info(f"Qdrant collection '{QDRANT_COLLECTION}' lista (dimensión {VECTOR_SIZE})")

    # Inicializar Neo4j
    try:
        graph_store.init_schema()
        neo4j_ok = graph_store.ping()
        if neo4j_ok:
            log.info("Neo4j conectado y schema inicializado")
        else:
            log.warning("Neo4j no responde al ping")
    except Exception as exc:
        log.warning("No se pudo inicializar Neo4j: %s", exc)

    # Verificar JavaParser
    try:
        import httpx
        resp = httpx.get(f"{JAVAPARSER_URL}/health", timeout=5.0)
        if resp.status_code == 200:
            log.info("JavaParser service disponible en %s", JAVAPARSER_URL)
        else:
            log.warning("JavaParser service respondió con status %s", resp.status_code)
    except Exception as exc:
        log.warning("JavaParser service no disponible: %s", exc)

    yield
    graph_store.close_driver()


app = FastAPI(title="Tennis Doc Indexer", lifespan=lifespan)

# ─────────────────────────────────────────
# Protección de endpoints (API Key)
# ─────────────────────────────────────────

def verify_api_key(authorization: str | None = Header(None)):
    """Valida el header Authorization Bearer si INDEXER_API_KEY está configurada."""
    if not INDEXER_API_KEY:
        return True
    if not authorization:
        raise HTTPException(status_code=401, detail="Falta header Authorization")
    if authorization != f"Bearer {INDEXER_API_KEY}":
        raise HTTPException(status_code=403, detail="API key inválida")
    return True

# CORS básico para permitir llamadas desde el frontend / Open WebUI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────
# Webhook de GitHub — se dispara en cada push
# ─────────────────────────────────────────

def _verify_signature(body: bytes, signature: str) -> bool:
    secret = WEBHOOK_SECRET.encode()
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

    # Solo procesamos push a la rama que se especificó
    if event == "push":
        repo_name = payload.get("repository", {}).get("full_name", "")
        ref = payload.get("ref", "")
        default_branch = payload.get("repository", {}).get("default_branch", "main")

        if ref.startswith("refs/heads/") and repo_name:
            branch = ref.replace("refs/heads/", "")
            log.info(f"Push detectado en {repo_name} @ {branch} — iniciando re-indexación")
            background_tasks.add_task(index_repo, repo_name, branch)

    return JSONResponse({"status": "ok"})


# ─────────────────────────────────────────
# Indexación de un repo completo
# ─────────────────────────────────────────

async def index_repo(full_repo_name: str, branch: str = "HEAD"):
    """
    Indexa todos los archivos de un repo (rama específica) con pipeline profesional:
    AST → Grafo (Neo4j) + Vectores (Qdrant). Corre en background.
    """
    try:
        owner, repo = full_repo_name.split("/", 1)
        log.info(f"Indexando {full_repo_name} @ {branch}...")

        token = get_installation_token()
        files = get_repo_files(token, owner, repo, ref=branch)
        log.info(f"Encontrados {len(files)} archivos en {full_repo_name} @ {branch}")

        client = get_client()
        # Asegurar que la colección existe (puede haber sido borrada manualmente)
        ensure_collection(client, QDRANT_COLLECTION)
        # Borrar datos previos de esta rama en Qdrant y Neo4j
        delete_repo_chunks(client, QDRANT_COLLECTION, full_repo_name, branch=branch)
        try:
            graph_store.clear_repo(full_repo_name, branch)
        except Exception as exc:
            log.warning("No se pudo limpiar Neo4j para %s@%s: %s", full_repo_name, branch, exc)

        all_file_paths = {f["path"] for f in files}
        sql_files_map = {f["path"]: f for f in files if f["path"].lower().endswith(".sql")}
        log.info(f"Archivos SQL detectados en el repo: {len(sql_files_map)}")

        total_entities = 0
        total_chunks = 0
        total_files = 0
        all_entities: list = []
        all_chunks: list = []

        for file_info in files:
            attempt = 0
            max_token_retries = 2
            processed = False
            while attempt < max_token_retries and not processed:
                try:
                    content = get_file_content(token, owner, repo, file_info["path"], ref=branch)
                    if not content or not content.strip():
                        log.debug(f"Archivo vacío o sin contenido: {file_info['path']}")
                        processed = True
                        continue

                    # Detectar lenguaje desde extensión
                    language = _detect_language_from_path(file_info["path"])

                    # ── Pipeline AST + Grafo ──
                    if language:
                        entities, chunks = ast_parser.parse_file_to_chunks_and_entities(
                            content, language, full_repo_name, branch, file_info["path"]
                        )
                    else:
                        # Fallback: chunking clásico para lenguajes no soportados por AST
                        chunks = chunk_file(content, file_info["path"], full_repo_name, branch=branch)
                        entities = []

                    if not chunks:
                        log.warning(f"Sin chunks para {file_info['path']} (tamaño {len(content)} chars)")
                        processed = True
                        continue

                    # Si es Java, resolver referencias a SQL inline
                    if file_info["path"].lower().endswith(".java") and sql_files_map:
                        chunks = await _inline_sql_references_ast(
                            chunks, token, owner, repo, branch, all_file_paths, sql_files_map
                        )

                    all_entities.extend(entities)
                    all_chunks.extend(chunks)
                    total_entities += len(entities)
                    total_files += 1

                    # Batch flush cada 500 entidades / 1000 chunks para no saturar memoria
                    if len(all_entities) >= 500:
                        await _flush_to_graph(client, all_entities, all_chunks)
                        total_chunks += len(all_chunks)
                        all_entities = []
                        all_chunks = []

                    log.info(f"Parseado {file_info['path']}: {len(content)} chars → {len(entities)} entidades, {len(chunks)} chunks")
                    processed = True

                except GitHubTokenExpired:
                    attempt += 1
                    if attempt < max_token_retries:
                        log.warning(f"Token expirado procesando {file_info['path']}, renovando token ({attempt}/{max_token_retries})...")
                        token = get_installation_token()
                    else:
                        log.error(f"Token sigue expirado después de {max_token_retries} intentos. Saltando {file_info['path']}")
                        processed = True

                except Exception as exc:
                    log.warning(f"Error procesando {file_info['path']} en {full_repo_name}: {exc}")
                    processed = True

        # Flush final
        if all_entities or all_chunks:
            await _flush_to_graph(client, all_entities, all_chunks)
            total_chunks += len(all_chunks)

        log.info(f"Indexación completa: {full_repo_name} @ {branch} — {total_files} archivos, {total_entities} entidades, {total_chunks} chunks guardados")

    except Exception as e:
        log.error(f"Error indexando {full_repo_name} @ {branch}: {e}")


def _detect_language_from_path(file_path: str) -> str | None:
    ext = file_path.lower().split(".")[-1] if "." in file_path else ""
    mapping = {
        "java": "java", "py": "python", "js": "javascript", "ts": "typescript",
        "jsx": "javascript", "tsx": "typescript", "go": "go",
    }
    return mapping.get(ext)


async def _flush_to_graph(client, entities: list, chunks: list):
    """Persiste entidades en Neo4j y chunks en Qdrant."""
    if entities:
        try:
            graph_store.upsert_entities(entities)
        except Exception as exc:
            log.error("Error guardando entidades en Neo4j: %s", exc)
    if chunks:
        try:
            texts = [c["metadata"]["embed_text"] for c in chunks]
            embeddings = await get_embeddings_batch(texts)
            upsert_chunks(client, QDRANT_COLLECTION, chunks, embeddings)
        except Exception as exc:
            log.error("Error guardando chunks en Qdrant: %s", exc)


async def _inline_sql_references_ast(
    chunks: list[dict],
    token: str,
    owner: str,
    repo: str,
    branch: str,
    all_file_paths: set[str],
    sql_files_map: dict[str, dict],
) -> list[dict]:
    """Resuelve referencias a archivos .sql dentro de chunks y adjunta el contenido SQL.
    Si el SQL es muy grande se trunca para no romper los límites de embedding."""
    _MAX_INLINE_SQL_CHARS = 6000
    for chunk in chunks:
        for match in _SQL_REF_RE.finditer(chunk["text"]):
            sql_ref = match.group(1)
            resolved = _resolve_sql_path(chunk["metadata"].get("file_path", ""), sql_ref, all_file_paths)
            if resolved and resolved in sql_files_map:
                try:
                    sql_content = get_file_content(token, owner, repo, resolved, ref=branch)
                    if sql_content and sql_content.strip():
                        if len(sql_content) > _MAX_INLINE_SQL_CHARS:
                            sql_content = (
                                sql_content[:_MAX_INLINE_SQL_CHARS]
                                + f"\n-- ... SQL truncado ({len(sql_content)} chars originales) ... --\n"
                            )
                        sql_header = f"\n\n-- Referenced SQL: {resolved} --\n"
                        chunk["text"] += sql_header + sql_content
                        chunk["metadata"]["embed_text"] += sql_header + sql_content
                except Exception as exc:
                    log.debug(f"No se pudo leer SQL referenciado {resolved}: {exc}")
    return chunks


# ─────────────────────────────────────────
# Endpoint manual para indexar un repo
# ─────────────────────────────────────────

class IndexRequest(BaseModel):
    repo: str    # formato: "org/repo-name"
    branch: str = "HEAD"  # rama a indexar (por defecto la default del repo)

    @field_validator("repo")
    @classmethod
    def validate_repo_format(cls, v: str) -> str:
        if not _REPO_PATTERN.match(v):
            raise ValueError("El campo 'repo' debe tener el formato 'org/repo-name'")
        return v


@app.post("/index", dependencies=[Depends(verify_api_key)])
async def manual_index(req: IndexRequest, background_tasks: BackgroundTasks):
    """Dispara indexación manual de un repo (rama opcional)."""
    background_tasks.add_task(index_repo, req.repo, req.branch)
    return {"status": "indexación iniciada", "repo": req.repo, "branch": req.branch}


# ─────────────────────────────────────────
# Endpoint de búsqueda semántica
# ─────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    repo: str | None = None     # si se omite, busca en todos los repos
    branch: str | None = None   # si se omite, busca en todas las ramas del repo
    limit: int = 6


@app.post("/search", dependencies=[Depends(verify_api_key)])
async def search(req: SearchRequest):
    """Busca chunks relevantes para una pregunta."""
    log.info(f"Búsqueda recibida: query='{req.query[:60]}...' repo={req.repo} branch={req.branch} limit={req.limit}")

    try:
        query_vector = await get_embedding(req.query)
        log.info(f"Embedding generado: dims={len(query_vector)} sample={query_vector[:3]}")
    except Exception as exc:
        log.error(f"Error generando embedding de búsqueda: {exc}")
        raise HTTPException(status_code=502, detail=f"Error al generar el embedding: {exc}")

    client = get_client()
    try:
        results = search_chunks(client, QDRANT_COLLECTION, query_vector, req.repo, req.branch, req.limit)
        log.info(f"Búsqueda completada: {len(results)} resultados")
        for i, r in enumerate(results):
            log.info(f"  Result {i+1}: score={r['score']:.3f} file={r['file_path'][:60]}")
    except Exception as exc:
        log.error(f"Error en búsqueda Qdrant: {exc}")
        raise HTTPException(status_code=500, detail=f"Error en búsqueda Qdrant: {exc}")

    return {"results": results}


# ─────────────────────────────────────────
# Búsqueda aumentada: archivos completos
# ─────────────────────────────────────────

class SearchAugmentedRequest(BaseModel):
    query: str
    repo: str | None = None
    branch: str | None = None
    max_files: int = 10      # cuántos archivos completos traer
    vector_limit: int = 50   # cuántos chunks vectoriales usar para identificar archivos


@app.post("/search-augmented", dependencies=[Depends(verify_api_key)])
async def search_augmented(req: SearchAugmentedRequest):
    """
    Búsqueda híbrida:
    1. Busca vectorialmente para identificar los archivos más relevantes.
    2. Trae TODOS los chunks de esos archivos (no solo los top-K).
    """
    log.info(f"Búsqueda aumentada: query='{req.query[:60]}...' repo={req.repo} branch={req.branch}")

    try:
        query_vector = await get_embedding(req.query)
    except Exception as exc:
        log.error(f"Error generando embedding: {exc}")
        raise HTTPException(status_code=502, detail=f"Error al generar el embedding: {exc}")

    client = get_client()

    # 1) Búsqueda vectorial amplia para identificar archivos candidatos
    try:
        vector_results = search_chunks(
            client, QDRANT_COLLECTION, query_vector,
            req.repo, req.branch, req.vector_limit,
        )
    except Exception as exc:
        log.error(f"Error en búsqueda vectorial: {exc}")
        raise HTTPException(status_code=500, detail=f"Error en búsqueda Qdrant: {exc}")

    if not vector_results:
        return {"results": [], "files_fetched": 0}

    # 2) Agrupar por file_path y puntuar archivos (suma de scores)
    from collections import defaultdict
    file_scores = defaultdict(float)
    for r in vector_results:
        file_scores[r["file_path"]] += r["score"]

    top_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)[:req.max_files]
    log.info(f"Archivos más relevantes: {[f[0] for f in top_files]}")

    # 3) Traer TODOS los chunks de cada archivo top
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    all_chunks = []
    for file_path, _ in top_files:
        must = [
            FieldCondition(key="repo", match=MatchValue(value=req.repo)) if req.repo else None,
            FieldCondition(key="branch", match=MatchValue(value=req.branch)) if req.branch else None,
            FieldCondition(key="file_path", match=MatchValue(value=file_path)),
        ]
        must = [c for c in must if c is not None]

        offset = None
        file_chunks = []
        while True:
            results = client.scroll(
                collection_name=QDRANT_COLLECTION,
                scroll_filter=Filter(must=must),
                limit=100,
                offset=offset,
                with_payload=True,
            )
            if hasattr(results, 'points'):
                points = results.points
                offset = results.next_page_offset
            else:
                points = results[0]
                offset = results[1]

            if not points:
                break

            for p in points:
                file_chunks.append({
                    "score": 0.0,
                    "repo": p.payload.get("repo"),
                    "branch": p.payload.get("branch", ""),
                    "file_path": p.payload.get("file_path"),
                    "language": p.payload.get("language"),
                    "text": p.payload.get("text", ""),
                })

            if offset is None:
                break

        # Ordenar chunks por posición para reconstruir el archivo en orden
        file_chunks.sort(key=lambda x: x.get("position", 0))
        all_chunks.extend(file_chunks)

    log.info(f"Búsqueda aumentada completada: {len(all_chunks)} chunks de {len(top_files)} archivos")
    return {"results": all_chunks, "files_fetched": len(top_files)}


# ─────────────────────────────────────────
# Búsqueda híbrida con Grafo (LlamaIndex)
# ─────────────────────────────────────────

class SearchGraphRequest(BaseModel):
    query: str
    repo: str | None = None
    branch: str | None = None
    limit: int = 12
    graph_depth: int = 2


@app.post("/search-graph", dependencies=[Depends(verify_api_key)])
async def search_graph_endpoint(req: SearchGraphRequest):
    """
    Búsqueda híbrida: vectorial (Qdrant) + grafo (Neo4j) + keyword.
    Usa LlamaIndex como orquestador de retrieval.
    """
    log.info(f"Búsqueda grafo: query='{req.query[:60]}...' repo={req.repo} branch={req.branch}")
    try:
        results = await rag_engine.search_graph(
            query=req.query,
            repo=req.repo,
            branch=req.branch,
            limit=req.limit,
            graph_depth=req.graph_depth,
        )
        log.info(f"Búsqueda grafo completada: {len(results)} resultados")
        return {"results": results}
    except Exception as exc:
        log.error(f"Error en búsqueda grafo: {exc}")
        raise HTTPException(status_code=500, detail=f"Error en búsqueda híbrida: {exc}")


@app.get("/graph/entity/{name}", dependencies=[Depends(verify_api_key)])
async def graph_entity(name: str, repo: str | None = None, branch: str | None = None):
    """Busca una entidad por nombre exacto y devuelve sus relaciones directas."""
    try:
        result = await rag_engine.search_entity_in_graph(name, repo=repo, branch=branch)
        if not result:
            raise HTTPException(status_code=404, detail=f"Entidad no encontrada: {name}")
        return result
    except HTTPException:
        raise
    except Exception as exc:
        log.error(f"Error consultando grafo para {name}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/graph/related/{entity_id}", dependencies=[Depends(verify_api_key)])
async def graph_related(entity_id: str, depth: int = 2):
    """Devuelve entidades relacionadas en el grafo desde un ID dado."""
    try:
        related = graph_store.get_related_entities(entity_id, depth=depth)
        return {"entity_id": entity_id, "depth": depth, "related": related}
    except Exception as exc:
        log.error(f"Error obteniendo relaciones para {entity_id}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ─────────────────────────────────────────
# Fetch directo de archivo desde GitHub
# ─────────────────────────────────────────

class FetchFileRequest(BaseModel):
    repo: str
    file_path: str
    branch: str = "HEAD"


@app.post("/fetch-file", dependencies=[Depends(verify_api_key)])
async def fetch_file(req: FetchFileRequest):
    """Trae el contenido crudo de un archivo específico desde GitHub."""
    try:
        owner, repo_name = req.repo.split("/", 1)
    except ValueError:
        raise HTTPException(status_code=400, detail="repo debe tener formato 'org/repo'")

    token = get_installation_token()
    try:
        content = get_file_content(token, owner, repo_name, req.file_path, ref=req.branch)
    except GitHubTokenExpired:
        token = get_installation_token()
        content = get_file_content(token, owner, repo_name, req.file_path, ref=req.branch)
    except Exception as exc:
        log.error(f"Error fetching {req.file_path}: {exc}")
        raise HTTPException(status_code=502, detail=f"Error al obtener archivo de GitHub: {exc}")

    if content is None:
        raise HTTPException(status_code=404, detail=f"Archivo no encontrado: {req.file_path}")

    return {
        "repo": req.repo,
        "branch": req.branch,
        "file_path": req.file_path,
        "content": content,
    }


# ─────────────────────────────────────────
# Generación de PDF
# ─────────────────────────────────────────

_NOTO_REG_URL = "https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf"
_NOTO_BOLD_URL = "https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSans/NotoSans-Bold.ttf"
_NOTO_REG_PATH = "/tmp/NotoSans-Regular.ttf"
_NOTO_BOLD_PATH = "/tmp/NotoSans-Bold.ttf"

try:
    from fpdf import FPDF
    _HAS_FPDF = True
except ImportError:
    _HAS_FPDF = False
    log.warning("fpdf2 no está instalado. El endpoint /pdf no funcionará.")


def _ensure_noto_fonts():
    if not _HAS_FPDF:
        return
    try:
        if not os.path.exists(_NOTO_REG_PATH):
            urllib.request.urlretrieve(_NOTO_REG_URL, _NOTO_REG_PATH)
        if not os.path.exists(_NOTO_BOLD_PATH):
            urllib.request.urlretrieve(_NOTO_BOLD_URL, _NOTO_BOLD_PATH)
    except Exception as exc:
        log.debug(f"No se pudieron descargar fuentes Noto: {exc}")


class _PDF(FPDF):
    def __init__(self, title: str = "Documento"):
        super().__init__()
        self.doc_title = title

    def header(self):
        has_noto = os.path.exists(_NOTO_REG_PATH)
        self.set_font("NotoSans" if has_noto else "Helvetica", "B", 14)
        self.cell(0, 10, self.doc_title, ln=True, align="C")
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        has_noto = os.path.exists(_NOTO_REG_PATH)
        self.set_font("NotoSans" if has_noto else "Helvetica", "", 8)
        self.cell(0, 10, f"Pagina {self.page_no()}", align="C")


class PdfRequest(BaseModel):
    title: str = "Documento"
    content: str
    repo: str | None = None
    branch: str | None = None


@app.post("/pdf", dependencies=[Depends(verify_api_key)])
async def generate_pdf(req: PdfRequest):
    """Genera un PDF a partir de markdown/texto plano."""
    if not _HAS_FPDF:
        raise HTTPException(status_code=501, detail="fpdf2 no está instalado")

    _ensure_noto_fonts()

    pdf = _PDF(title=req.title)
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    has_noto = os.path.exists(_NOTO_REG_PATH)
    if has_noto:
        pdf.add_font("NotoSans", "", _NOTO_REG_PATH, uni=True)
        pdf.add_font("NotoSans", "B", _NOTO_BOLD_PATH, uni=True)

    for raw_line in req.content.split("\n"):
        line = raw_line.rstrip()
        if not line:
            pdf.ln(3)
            continue

        if line.startswith("# "):
            pdf.set_font("NotoSans" if has_noto else "Helvetica", "B", 16)
            pdf.multi_cell(0, 8, line[2:])
        elif line.startswith("## "):
            pdf.set_font("NotoSans" if has_noto else "Helvetica", "B", 13)
            pdf.multi_cell(0, 7, line[3:])
        elif line.startswith("### "):
            pdf.set_font("NotoSans" if has_noto else "Helvetica", "B", 11)
            pdf.multi_cell(0, 6, line[4:])
        else:
            pdf.set_font("NotoSans" if has_noto else "Helvetica", "", 10)
            pdf.multi_cell(0, 5, line)

    output = BytesIO()
    pdf.output(output)
    output.seek(0)

    filename = f"{req.title.replace(' ', '_')}.pdf"
    return StreamingResponse(
        output,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ─────────────────────────────────────────
# Generación de Markdown descargable
# ─────────────────────────────────────────

class MarkdownRequest(BaseModel):
    title: str = "Documento"
    content: str
    repo: str | None = None
    branch: str | None = None


@app.post("/markdown", dependencies=[Depends(verify_api_key)])
async def generate_markdown(req: MarkdownRequest):
    """Genera un archivo .md descargable a partir de contenido markdown."""
    filename = f"{req.title.replace(' ', '_')}.md"
    # Añadir metadatos YAML frontmatter si se proporciona repo/rama
    frontmatter = ""
    if req.repo or req.branch:
        frontmatter = "---\n"
        if req.repo:
            frontmatter += f"repo: {req.repo}\n"
        if req.branch:
            frontmatter += f"branch: {req.branch}\n"
        frontmatter += "---\n\n"

    full_content = frontmatter + req.content
    return StreamingResponse(
        BytesIO(full_content.encode("utf-8")),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ─────────────────────────────────────────
# Endpoints de diagnóstico
# ─────────────────────────────────────────

@app.get("/debug/files", dependencies=[Depends(verify_api_key)])
async def debug_files(repo: str, branch: str = "HEAD"):
    """Lista los archivos que serían indexados (sin indexar)."""
    try:
        owner, repo_name = repo.split("/", 1)
        token = get_installation_token()
        files = get_repo_files(token, owner, repo_name, ref=branch)
        return {"repo": repo, "branch": branch, "file_count": len(files), "files": [f["path"] for f in files]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/debug/files-indexed", dependencies=[Depends(verify_api_key)])
async def debug_files_indexed(repo: str, branch: str = "HEAD", language: str | None = None):
    """Lista los file_paths únicos que YA están indexados en Qdrant para un repo/rama."""
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        client = get_client()
        must = [
            FieldCondition(key="repo", match=MatchValue(value=repo)),
            FieldCondition(key="branch", match=MatchValue(value=branch)),
        ]
        if language:
            must.append(FieldCondition(key="language", match=MatchValue(value=language)))

        all_paths = set()
        offset = None
        while True:
            results = client.scroll(
                collection_name=QDRANT_COLLECTION,
                scroll_filter=Filter(must=must),
                limit=1000,
                offset=offset,
                with_payload=True,
            )
            # Compatibilidad: scroll puede retornar tupla o objeto con .points / .next_page_offset
            if hasattr(results, 'points'):
                points = results.points
                offset = results.next_page_offset
            else:
                points = results[0]
                offset = results[1]
            if not points:
                break
            for p in points:
                path = p.payload.get("file_path")
                if path:
                    all_paths.add(path)
            if offset is None:
                break

        return {
            "repo": repo,
            "branch": branch,
            "language_filter": language,
            "indexed_file_count": len(all_paths),
            "files": sorted(list(all_paths)),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/debug/chunks", dependencies=[Depends(verify_api_key)])
async def debug_chunks(repo: str, file_path: str, branch: str = "HEAD"):
    """Muestra los chunks guardados en Qdrant para un archivo específico."""
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        client = get_client()
        must = [
            FieldCondition(key="repo", match=MatchValue(value=repo)),
            FieldCondition(key="branch", match=MatchValue(value=branch)),
            FieldCondition(key="file_path", match=MatchValue(value=file_path)),
        ]
        results = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=Filter(must=must),
            limit=50,
            with_payload=True,
        )
        # Compatibilidad: scroll puede retornar tupla o objeto con .points
        if hasattr(results, 'points'):
            points = results.points
        else:
            points = results[0]
        return {
            "repo": repo,
            "branch": branch,
            "file_path": file_path,
            "chunk_count": len(points),
            "chunks": [
                {
                    "id": str(p.id),
                    "language": p.payload.get("language"),
                    "position": p.payload.get("position"),
                    "text_preview": p.payload.get("text", "")[:500],
                }
                for p in points
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ─────────────────────────────────────────
# Health check (con verificación de Qdrant)
# ─────────────────────────────────────────

@app.get("/health")
async def health():
    client = get_client()
    qdrant_ok = ping_client(client)
    neo4j_ok = graph_store.ping()
    javaparser_ok = False
    try:
        import httpx
        resp = httpx.get(f"{JAVAPARSER_URL}/health", timeout=5.0)
        javaparser_ok = resp.status_code == 200
    except Exception:
        pass

    if not qdrant_ok:
        raise HTTPException(status_code=503, detail="Qdrant no responde")

    status = {
        "status": "ok",
        "qdrant": "reachable" if qdrant_ok else "unreachable",
        "neo4j": "reachable" if neo4j_ok else "unreachable",
        "javaparser": "reachable" if javaparser_ok else "unreachable",
    }
    if not neo4j_ok or not javaparser_ok:
        status["status"] = "degraded"
    return status
