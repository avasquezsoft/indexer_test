import asyncio
import hmac
import hashlib
import logging
import os
import re
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, Depends, Header
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from io import BytesIO

from github_client import get_installation_token, get_installation_token_for_repo, get_repo_files, get_file_content, GitHubTokenExpired, list_repos, list_all_repos
from chunker import chunk_file
from embedder import get_embedding, get_embeddings_batch
from qdrant_store import get_client, ensure_collection, delete_repo_chunks, upsert_chunks, search_chunks, ping_client
from config import QDRANT_COLLECTION, WEBHOOK_SECRET, VECTOR_SIZE, JAVAPARSER_URL, INDEXER_API_KEY

import ast_parser
import code_links
import graph_store
import rag_engine
import repo_clone

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Regex simple para validar formato org/repo
_REPO_PATTERN = re.compile(r"^[\w.-]+/[\w.-]+$")



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
async def github_webhook(request: Request):
    body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256", "")
    delivery = request.headers.get("X-GitHub-Delivery", "unknown")
    event = request.headers.get("X-GitHub-Event", "unknown")

    log.info(f"Webhook recibido: delivery={delivery} event={event} ip={request.client.host if request.client else 'unknown'} size={len(body)} bytes")

    if not _verify_signature(body, signature):
        log.warning(f"Webhook firma inválida: delivery={delivery} event={event}")
        raise HTTPException(status_code=401, detail="Firma inválida")

    try:
        payload = await request.json()
    except Exception as exc:
        log.warning(f"Webhook JSON inválido: delivery={delivery} error={exc}")
        raise HTTPException(status_code=400, detail="Payload inválido")

    if event == "push":
        repo_name = payload.get("repository", {}).get("full_name", "")
        ref = payload.get("ref", "")
        pusher = payload.get("pusher", {}).get("name", "unknown")

        if ref.startswith("refs/heads/") and repo_name:
            branch = ref.replace("refs/heads/", "")
            log.info(f"Push detectado en {repo_name} @ {branch} (pusher={pusher}) — iniciando re-indexación")
            asyncio.create_task(asyncio.to_thread(_run_index_sync, repo_name, branch))
        else:
            log.info(f"Push ignorado: repo={repo_name} ref={ref}")

    elif event == "pull_request":
        action = payload.get("action", "")
        merged = payload.get("pull_request", {}).get("merged", False)
        base_ref = payload.get("pull_request", {}).get("base", {}).get("ref", "")
        head_ref = payload.get("pull_request", {}).get("head", {}).get("ref", "")
        repo_name = payload.get("repository", {}).get("full_name", "")
        pr_number = payload.get("number", "?")
        sender = payload.get("sender", {}).get("login", "unknown")

        log.info(f"Pull request event: action={action} merged={merged} repo={repo_name} pr=#{pr_number} base={base_ref} head={head_ref} sender={sender}")

        if action == "closed" and merged and repo_name and base_ref:
            log.info(f"PR mergeado detectado en {repo_name} @ {base_ref} — iniciando re-indexación")
            asyncio.create_task(asyncio.to_thread(_run_index_sync, repo_name, base_ref))
        else:
            log.info(f"PR event ignorado: action={action} merged={merged}")

    else:
        log.info(f"Evento GitHub no manejado: {event}")

    return JSONResponse({"status": "ok"})


# ─────────────────────────────────────────
# Indexación de un repo completo
# ─────────────────────────────────────────

def _run_index_sync(full_repo_name: str, branch: str = "HEAD"):
    """Wrapper síncrono para ejecutar index_repo en un thread separado."""
    asyncio.run(index_repo(full_repo_name, branch))


async def index_repo(full_repo_name: str, branch: str = "HEAD"):
    """
    Indexa todos los archivos de un repo (rama específica) con pipeline profesional:
    AST → Grafo (Neo4j) + Vectores (Qdrant). Corre en background.
    También mantiene un clon local actualizado en CLONE_BASE_DIR.
    """
    total_entities = 0
    total_chunks = 0
    total_files = 0
    success = False
    details = ""

    try:
        owner, repo = full_repo_name.split("/", 1)
        log.info(f"Indexando {full_repo_name} @ {branch}...")

        # Asegurar clon local actualizado: si existe, los archivos se leen de disco
        clone_ok = False
        try:
            if await repo_clone.clone_or_pull_repo(full_repo_name, branch):
                clone_ok = True
                log.info("Clon local actualizado para %s @ %s", full_repo_name, branch)
        except Exception as exc:
            log.warning("No se pudo actualizar clon local de %s: %s", full_repo_name, exc)

        token = get_installation_token_for_repo(owner, repo)
        read_counts = {"clon": 0, "api": 0}

        def read_file(path: str) -> str | None:
            """Lee del clon local; si no está ahí, de la API de GitHub."""
            if clone_ok:
                content = repo_clone.read_file_from_clone(full_repo_name, path, branch)
                if content is not None:
                    read_counts["clon"] += 1
                    return content
            read_counts["api"] += 1
            return get_file_content(token, owner, repo, path, ref=branch)

        # Un .sql lo referencian varios chunks y archivos Java: se lee una sola vez
        sql_cache: dict[str, str | None] = {}

        def read_sql(path: str) -> str | None:
            if path not in sql_cache:
                sql_cache[path] = read_file(path)
            return sql_cache[path]
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

        all_entities: list = []
        all_chunks: list = []
        # Relaciones y tablas se crean al final, con todos los nodos ya guardados
        pending_rels: list[dict] = []
        all_tables: set[str] = set()

        def resolve_sql(ref: str, source_path: str) -> str | None:
            path = _resolve_sql_path(source_path, ref, all_file_paths)
            return path if path in sql_files_map else None

        for file_info in files:
            attempt = 0
            max_token_retries = 2
            processed = False
            while attempt < max_token_retries and not processed:
                try:
                    content = read_file(file_info["path"])
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
                        if file_info["path"].lower().endswith(".sql"):
                            sql_entity, tables = code_links.sql_file_entity(
                                file_info["path"], content, full_repo_name, branch
                            )
                            entities = [sql_entity]
                            all_tables |= tables

                    if not chunks:
                        log.warning(f"Sin chunks para {file_info['path']} (tamaño {len(content)} chars)")
                        processed = True
                        continue

                    # Si es Java, resolver referencias a SQL inline
                    if file_info["path"].lower().endswith(".java") and sql_files_map:
                        chunks = _inline_sql_references_ast(
                            chunks, read_sql, all_file_paths, sql_files_map
                        )

                    if language:
                        all_tables |= code_links.add_sql_links(entities, resolve_sql)
                        if language == "java":
                            code_links.add_routes(entities)
                        _annotate_chunks(chunks, entities)

                    all_entities.extend(entities)
                    all_chunks.extend(chunks)
                    total_entities += len(entities)
                    total_files += 1

                    # Batch flush cada 500 entidades / 1000 chunks para no saturar memoria
                    if len(all_entities) >= 500:
                        pending_rels += await _flush_to_graph(client, all_entities, all_chunks)
                        total_chunks += len(all_chunks)
                        all_entities = []
                        all_chunks = []

                    log.info(f"Parseado {file_info['path']}: {len(content)} chars → {len(entities)} entidades, {len(chunks)} chunks")
                    processed = True

                except GitHubTokenExpired:
                    attempt += 1
                    if attempt < max_token_retries:
                        log.warning(f"Token expirado procesando {file_info['path']}, renovando token ({attempt}/{max_token_retries})...")
                        token = get_installation_token_for_repo(owner, repo)
                    else:
                        log.error(f"Token sigue expirado después de {max_token_retries} intentos. Saltando {file_info['path']}")
                        processed = True

                except Exception as exc:
                    log.warning(f"Error procesando {file_info['path']} en {full_repo_name}: {exc}")
                    processed = True

        # Flush final
        if all_entities or all_chunks:
            pending_rels += await _flush_to_graph(client, all_entities, all_chunks)
            total_chunks += len(all_chunks)

        # Relaciones al final: así ninguna se pierde por apuntar a un lote posterior
        try:
            if all_tables:
                graph_store.upsert_entities(code_links.table_entities(all_tables, full_repo_name, branch))
            graph_store.upsert_relations(pending_rels)
        except Exception as exc:
            log.error("Error creando relaciones en Neo4j: %s", exc)

        log.info(f"Indexación completa: {full_repo_name} @ {branch} — {total_files} archivos, {total_entities} entidades, {total_chunks} chunks guardados")
        log.info("Archivos leídos: %d desde el clon, %d desde la API de GitHub", read_counts["clon"], read_counts["api"])
        success = True
        details = f"Archivos procesados: {total_files}\nEntidades: {total_entities}\nChunks: {total_chunks}"

    except Exception as e:
        log.error(f"Error indexando {full_repo_name} @ {branch}: {e}")
        details = str(e)
    finally:
        try:
            from email_notifier import send_index_notification
            send_index_notification(full_repo_name, branch, success, details)
        except Exception as ne:
            log.error("Error al enviar notificación de indexación: %s", ne)


def _detect_language_from_path(file_path: str) -> str | None:
    ext = file_path.lower().split(".")[-1] if "." in file_path else ""
    mapping = {
        "java": "java", "py": "python", "js": "javascript", "ts": "typescript",
        "jsx": "javascript", "tsx": "typescript", "go": "go",
    }
    return mapping.get(ext)


async def _flush_to_graph(client, entities: list, chunks: list) -> list[dict]:
    """Persiste nodos en Neo4j y chunks en Qdrant. Devuelve las relaciones pendientes."""
    rels: list[dict] = []
    if entities:
        try:
            rels = graph_store.upsert_entities(entities)
        except Exception as exc:
            log.error("Error guardando entidades en Neo4j: %s", exc)
    if chunks:
        try:
            texts = [c["metadata"]["embed_text"] for c in chunks]
            embeddings = await get_embeddings_batch(texts)
            upsert_chunks(client, QDRANT_COLLECTION, chunks, embeddings)
        except Exception as exc:
            log.error("Error guardando chunks en Qdrant: %s", exc)
    return rels


def _annotate_chunks(chunks: list[dict], entities: list) -> None:
    """
    Suma al texto de embedding el endpoint, los .sql y las tablas de cada entidad,
    para que preguntas como "¿qué usa la tabla FACTURA?" o "endpoint /facturas"
    encuentren el método por búsqueda vectorial.
    """
    by_id = {ast_parser._make_entity_id(e): e for e in entities}
    for chunk in chunks:
        entity = by_id.get(chunk["metadata"].get("entity_id"))
        if entity is None:
            continue
        extra = []
        if entity.route:
            extra.append(f"Endpoint: {entity.route}")
            chunk["metadata"]["route"] = entity.route
        sql_files = [r.target_name for r in entity.relations if r.type == "USES_SQL"]
        reads = [r.target_name for r in entity.relations if r.type == "READS"]
        writes = [r.target_name for r in entity.relations if r.type == "WRITES"]
        if sql_files:
            extra.append(f"Uses SQL: {', '.join(sql_files)}")
        if reads:
            extra.append(f"Reads tables: {', '.join(reads)}")
        if writes:
            extra.append(f"Writes tables: {', '.join(writes)}")
        if extra:
            chunk["metadata"]["embed_text"] += "\n" + "\n".join(extra)


def _inline_sql_references_ast(
    chunks: list[dict],
    read_sql: Callable[[str], str | None],
    all_file_paths: set[str],
    sql_files_map: dict[str, dict],
) -> list[dict]:
    """Resuelve referencias a archivos .sql dentro de chunks y adjunta el contenido SQL.
    Si el SQL es muy grande se trunca para no romper los límites de embedding."""
    _MAX_INLINE_SQL_CHARS = 6000
    for chunk in chunks:
        for match in code_links.SQL_REF_RE.finditer(chunk["text"]):
            sql_ref = match.group(1)
            resolved = _resolve_sql_path(chunk["metadata"].get("file_path", ""), sql_ref, all_file_paths)
            if resolved and resolved in sql_files_map:
                try:
                    sql_content = read_sql(resolved)
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
async def manual_index(req: IndexRequest):
    """Dispara indexación manual de un repo (rama opcional) en un thread aparte."""
    asyncio.create_task(asyncio.to_thread(_run_index_sync, req.repo, req.branch))
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


@app.get("/graph/usages/{name}", dependencies=[Depends(verify_api_key)])
async def graph_usages(name: str, repo: str | None = None, branch: str | None = None, limit: int = 100):
    """Quién usa una clase/método/tabla: llamadas, inyecciones, herencia, SQL."""
    try:
        usages = await asyncio.to_thread(graph_store.find_usages, name, repo, branch, min(limit, 300))
        return {"name": name, "usages": usages}
    except Exception as exc:
        log.error("Error buscando usos de %s: %s", name, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/graph/flow/{name}", dependencies=[Depends(verify_api_key)])
async def graph_flow(name: str, repo: str | None = None, branch: str | None = None, depth: int = 4):
    """Flujo hacia abajo: qué llama, qué inyecta, qué SQL ejecuta y qué tablas toca."""
    try:
        edges = await asyncio.to_thread(graph_store.get_flow, name, repo, branch, depth)
        return {"name": name, "depth": depth, "edges": edges}
    except Exception as exc:
        log.error("Error obteniendo flujo de %s: %s", name, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/sql/table/{table}", dependencies=[Depends(verify_api_key)])
async def sql_table_usage(table: str, repo: str | None = None, branch: str | None = None):
    """Qué .sql y qué métodos leen o escriben una tabla."""
    try:
        usage = await asyncio.to_thread(graph_store.find_table_usage, table, repo, branch)
        return {"table": table.upper(), "usage": usage}
    except Exception as exc:
        log.error("Error buscando uso de la tabla %s: %s", table, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/sql/tables", dependencies=[Depends(verify_api_key)])
async def sql_tables(repo: str, branch: str | None = None):
    """Tablas que usa un repo, con cuántos lectores y escritores tiene cada una."""
    try:
        tables = await asyncio.to_thread(graph_store.list_tables, repo, branch)
        return {"repo": repo, "tables": tables}
    except Exception as exc:
        log.error("Error listando tablas de %s: %s", repo, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/endpoints", dependencies=[Depends(verify_api_key)])
async def list_endpoints(repo: str, branch: str | None = None):
    """Rutas HTTP (Spring MVC / JAX-RS) expuestas por un repo."""
    try:
        endpoints = await asyncio.to_thread(graph_store.list_endpoints, repo, branch)
        return {"repo": repo, "endpoints": endpoints}
    except Exception as exc:
        log.error("Error listando endpoints de %s: %s", repo, exc)
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
    """
    Trae el contenido crudo de un archivo específico.
    Primero intenta leer del clon local; si no existe, fallback a GitHub API.
    """
    # 1) Intentar leer del clon local
    local_content = repo_clone.read_file_from_clone(req.repo, req.file_path, req.branch)
    if local_content is not None:
        log.info("Fetch local: %s/%s (desde clon)", req.repo, req.file_path)
        return {
            "repo": req.repo,
            "branch": req.branch,
            "file_path": req.file_path,
            "content": local_content,
            "source": "clone",
        }

    # 2) Fallback a GitHub API
    try:
        owner, repo_name = req.repo.split("/", 1)
    except ValueError:
        raise HTTPException(status_code=400, detail="repo debe tener formato 'org/repo'")

    token = get_installation_token_for_repo(owner, repo_name)
    try:
        content = get_file_content(token, owner, repo_name, req.file_path, ref=req.branch)
    except GitHubTokenExpired:
        token = get_installation_token_for_repo(owner, repo_name)
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
        "source": "github",
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
# Búsqueda en clon local (fallback cuando falta contexto)
# ─────────────────────────────────────────

class SearchCloneRequest(BaseModel):
    repo: str
    branch: str = "HEAD"
    keywords: list[str] = []
    entities: list[str] = []   # clases mencionadas: su archivo va primero
    methods: list[str] = []    # métodos mencionados: su cuerpo se incluye completo
    max_files: int = 20
    max_chars_per_file: int = 40000


@app.post("/search-clone", dependencies=[Depends(verify_api_key)])
async def search_clone(req: SearchCloneRequest):
    """
    Busca en el clon local de repo@branch los archivos más relevantes para la
    pregunta. Si el clon no existe, lo crea en el momento.
    Útil cuando la búsqueda vectorial/grafo no trae suficiente contexto.
    """
    if not _REPO_PATTERN.match(req.repo):
        raise HTTPException(status_code=400, detail="repo debe tener formato 'org/repo'")
    if not (req.keywords or req.entities or req.methods):
        return {"results": []}

    clone_path = repo_clone._get_clone_path(req.repo, req.branch)
    if not os.path.isdir(os.path.join(clone_path, ".git")):
        log.info("Clon local no encontrado para %s @ %s: clonando bajo demanda", req.repo, req.branch)
        try:
            if not await repo_clone.clone_or_pull_repo(req.repo, req.branch):
                return {"results": []}
        except Exception as exc:
            log.warning("No se pudo clonar %s @ %s: %s", req.repo, req.branch, exc)
            return {"results": []}

    results = await asyncio.to_thread(
        repo_clone.search_clone,
        req.repo,
        req.branch,
        req.keywords,
        req.entities,
        req.methods,
        req.max_files,
        req.max_chars_per_file,
    )
    log.info(
        "Búsqueda en clon %s @ %s: %d archivos (exactos=%d, keywords=%s, entidades=%s)",
        req.repo, req.branch, len(results),
        sum(1 for r in results if r["match"] == "name"), req.keywords, req.entities,
    )
    return {"results": results}


# ─────────────────────────────────────────
# Endpoints de diagnóstico
# ─────────────────────────────────────────

@app.get("/debug/files", dependencies=[Depends(verify_api_key)])
async def debug_files(repo: str, branch: str = "HEAD"):
    """Lista los archivos que serían indexados (sin indexar)."""
    try:
        owner, repo_name = repo.split("/", 1)
        token = get_installation_token_for_repo(owner, repo_name)
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

@app.get("/repos", dependencies=[Depends(verify_api_key)])
async def list_indexed_repos():
    """Devuelve la lista de repos únicos indexados en Neo4j."""
    try:
        driver = graph_store.get_driver()
        with driver.session() as session:
            result = session.run("MATCH (e:CodeEntity) RETURN DISTINCT e.repo AS repo ORDER BY repo")
            repos = [record["repo"] for record in result if record["repo"]]
        return {"repos": repos}
    except Exception as exc:
        log.error(f"Error listando repos: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/repos-available", dependencies=[Depends(verify_api_key)])
async def list_available_repos():
    """Lista todos los repositorios a los que la GitHub App tiene acceso vía la instalación."""
    try:
        raw_repos = list_all_repos()
        repos = [
            {
                "full_name": r.get("full_name"),
                "name": r.get("name"),
                "owner": r.get("owner", {}).get("login"),
                "default_branch": r.get("default_branch", "main"),
                "private": r.get("private"),
                "html_url": r.get("html_url"),
                "description": r.get("description"),
            }
            for r in raw_repos
        ]
        return {"count": len(repos), "repos": repos}
    except Exception as exc:
        log.error(f"Error listando repos disponibles: {exc}")
        raise HTTPException(status_code=502, detail=f"Error al obtener repos de GitHub: {exc}")


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
