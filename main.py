import hmac
import hashlib
import logging
import os
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from io import BytesIO

from github_client import get_installation_token, get_repo_files, get_file_content, GitHubTokenExpired
from chunker import chunk_file
from embedder import get_embedding, get_embeddings_batch
from qdrant_store import get_client, ensure_collection, delete_repo_chunks, upsert_chunks, search_chunks, ping_client
from config import QDRANT_COLLECTION, WEBHOOK_SECRET

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
    """
    for chunk in chunks:
        for match in _SQL_REF_RE.finditer(chunk["text"]):
            sql_ref = match.group(1)
            resolved = _resolve_sql_path(chunk["metadata"]["file_path"], sql_ref, all_file_paths)
            if resolved and resolved in sql_files_map:
                try:
                    sql_content = get_file_content(token, owner, repo, resolved, ref=branch)
                    if sql_content and sql_content.strip():
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
    log.info(f"Qdrant collection '{QDRANT_COLLECTION}' lista")
    yield


app = FastAPI(title="Tennis Doc Indexer", lifespan=lifespan)

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
    """Indexa todos los archivos de un repo (rama específica). Corre en background."""
    try:
        owner, repo = full_repo_name.split("/", 1)
        log.info(f"Indexando {full_repo_name} @ {branch}...")

        token = get_installation_token()
        files = get_repo_files(token, owner, repo, ref=branch)
        log.info(f"Encontrados {len(files)} archivos en {full_repo_name} @ {branch}")

        client = get_client()
        # Borrar solo los chunks de esta rama para no afectar otras ramas indexadas
        delete_repo_chunks(client, QDRANT_COLLECTION, full_repo_name, branch=branch)

        all_file_paths = {f["path"] for f in files}
        sql_files_map = {f["path"]: f for f in files if f["path"].lower().endswith(".sql")}
        log.info(f"Archivos SQL detectados en el repo: {len(sql_files_map)}")

        total_chunks = 0
        total_files = 0
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

                    chunks = chunk_file(content, file_info["path"], full_repo_name, branch=branch)
                    if not chunks:
                        log.warning(f"Sin chunks para {file_info['path']} (tamaño {len(content)} chars)")
                        processed = True
                        continue

                    # Si es Java, resolver referencias a SQL inline
                    if file_info["path"].lower().endswith(".java") and sql_files_map:
                        chunks = await _inline_sql_references(
                            chunks, token, owner, repo, branch, all_file_paths, sql_files_map
                        )

                    texts = [c["metadata"]["embed_text"] for c in chunks]
                    embeddings = await get_embeddings_batch(texts)
                    upsert_chunks(client, QDRANT_COLLECTION, chunks, embeddings)
                    total_chunks += len(chunks)
                    total_files += 1
                    log.info(f"Indexado {file_info['path']}: {len(content)} chars → {len(chunks)} chunks")
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

        log.info(f"Indexación completa: {full_repo_name} @ {branch} — {total_files} archivos, {total_chunks} chunks guardados")

    except Exception as e:
        log.error(f"Error indexando {full_repo_name} @ {branch}: {e}")


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


@app.post("/index")
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


@app.post("/search")
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


@app.post("/pdf")
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
# Endpoints de diagnóstico
# ─────────────────────────────────────────

@app.get("/debug/files")
async def debug_files(repo: str, branch: str = "HEAD"):
    """Lista los archivos que serían indexados (sin indexar)."""
    try:
        owner, repo_name = repo.split("/", 1)
        token = get_installation_token()
        files = get_repo_files(token, owner, repo_name, ref=branch)
        return {"repo": repo, "branch": branch, "file_count": len(files), "files": [f["path"] for f in files]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/debug/chunks")
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
    if not qdrant_ok:
        raise HTTPException(status_code=503, detail="Qdrant no responde")
    return {"status": "ok", "qdrant": "reachable"}
