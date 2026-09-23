"""
title: Tennis Doc RAG
version: 3.1
requirements: requests
description: Filtro para Open WebUI que recupera contexto de código desde Tennis Doc IA.

Arquitectura
============

Open WebUI
    |
    v
inlet()  ── comandos (repos, indexar, grafo, pdf)  ──> resultado exacto vía outlet()
    |
    +── charla / seguimiento ("gracias", "hazlo más corto") ──> sin búsqueda
    |
    +── RAG (en paralelo)
    |       /search-augmented   archivos completos / contexto ampliado
    |       /search-graph       Qdrant + Neo4j + exact match
    |       /graph/entity/{n}   entidad completa + relaciones
    |       /search-clone       si se nombra una clase/método: su archivo real
    |                           desde el clon (condensado si es muy grande)
    |   luego, si aún falta contexto:
    |       /search-clone       fallback por keywords
    |
    +── fusión RRF + deduplicación + presupuesto de tokens
    |
    v
mensaje de sistema (unido al system prompt existente) + citas en la UI

Cambios v3
==========
- Valves editables desde la UI (pydantic).
- Búsquedas en paralelo, sin bloquear el event loop de Open WebUI.
- Fusión por posición (RRF) en vez de mezclar escalas de score.
- Dedupe por archivo/entidad/contenido.
- Repos validados contra /repos (caché); acepta el nombre corto del repo.
- Memoria de conversación: repo, rama y preguntas de seguimiento.
- Si no hay contexto, se le prohíbe al modelo inventar.
- Resultados exactos de comandos (grafo, índice, PDF) garantizados en outlet().
- Estado ("Buscando…") y fuentes clicables en la interfaz.
"""

import asyncio
import base64
import io
import os
import re
import threading
import time
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

import requests
from pydantic import BaseModel, Field


# ============================================================
# UTILIDADES
# ============================================================


def _normalize(text):
    """Minúsculas y sin tildes: 'Cuántos' -> 'cuantos'."""

    text = unicodedata.normalize("NFD", str(text).lower())

    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def _text_of(content):
    """Texto plano de un mensaje (str o lista multimodal)."""

    if isinstance(content, list):

        return " ".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("text")
        ).strip()

    return str(content or "").strip()


def _count_turns(messages):
    """Cantidad de mensajes user/assistant (ignora system)."""

    return sum(1 for m in messages or [] if m.get("role") in ("user", "assistant"))


# Resultados de comandos pendientes de mostrar en outlet().
# Nivel de módulo para sobrevivir entre inlet y outlet.
_PENDING = {}
_PENDING_TTL = 600


# ============================================================
# REGEX
# ============================================================

# ----------------------------------------------------------------
# Repositorio "org/repo"
# No acepta segmentos de ruta (src/main/java) ni URLs.
# ----------------------------------------------------------------

_REPO_RE = re.compile(r"(?<![\w./-])([\w.-]+)/([\w.-]+)(?![\w./-])")

_REPO_STOP = {
    "y/o",
    "i/o",
    "tcp/ip",
    "input/output",
    "crud/rest",
    "src/main",
    "src/test",
    "http/https",
    "get/post",
    "si/no",
    "true/false",
}

# ----------------------------------------------------------------
# Rama: "rama develop", "branch `feature/x`"
# ----------------------------------------------------------------

_BRANCH_RE = re.compile(
    r"\b(?:rama|branch)\s+[`'\"]?([A-Za-z0-9._/-]+)[`'\"]?",
    re.IGNORECASE,
)

_BRANCH_STOP = {
    "de",
    "del",
    "la",
    "el",
    "los",
    "las",
    "un",
    "una",
    "principal",
    "actual",
    "que",
    "en",
    "por",
    "con",
    "esta",
    "este",
    "mi",
    "su",
    "nueva",
    "otra",
    "correcta",
    "default",
    "y",
    "o",
}

# ----------------------------------------------------------------
# Exportar (se evalúa sobre texto normalizado)
# ----------------------------------------------------------------

_EXPORT_VERB_RE = re.compile(
    r"^\s*(?:por\s+favor\s+)?(?:genera\w*|crea\w*|descarga\w*|exporta\w*|"
    r"guarda\w*|saca\w*|dame|convierte\w*|pasa\w*|hazme|haz|prepara\w*|"
    r"arma\w*|quiero|necesito|manda\w*|envia\w*|lo\s+quiero|ponlo)\b"
)

_PDF_WORD_RE = re.compile(r"(?<!\.)\bpdf\b")

_PDF_EXPLICIT_RE = re.compile(r"\b(?:en|a|como|formato|al)\s+(?:un\s+)?pdf\b")

_MD_WORD_RE = re.compile(r"(?<!\.)\b(?:markdown|md)\b")

_MD_EXPLICIT_RE = re.compile(
    r"\b(?:en|a|como|formato|al)\s+(?:un\s+)?(?:markdown|md)\b"
)

# Preguntas SOBRE el código que genera PDF/MD, no peticiones de exportar.
_EXPORT_QUESTION_RE = re.compile(
    r"\b(?:como|donde|quien|que\s+clase|que\s+metodo|cual)\s+"
    r"(?:\w+\s+){0,3}?(?:genera|crea|arma|exporta|construye)\w*\b"
)

# ----------------------------------------------------------------
# Grafo: "grafo PaymentService", "grafo de la clase `PaymentService`"
# ----------------------------------------------------------------

_GRAPH_RE = re.compile(
    r"\b(?i:grafo)\s+(?i:de\s+)?(?i:la\s+)?(?i:clase\s+)?`?([A-Z][A-Za-z0-9_]*)`?"
)

# ----------------------------------------------------------------
# Indexar: "indexa org/repo [rama]"
# ----------------------------------------------------------------

_INDEX_CMD_RE = re.compile(
    r"^\s*(?:por\s+favor\s+)?(?:re)?index\w*\s+(?:el\s+)?(?:repo(?:sitorio)?\s+)?"
    r"`?([\w.-]+/[\w.-]+)`?(?:\s+(?:en\s+)?(?:la\s+)?(?:rama\s+|branch\s+)?`?([\w./-]+)`?)?\s*$",
    re.IGNORECASE,
)

# ----------------------------------------------------------------
# Intención: repositorios indexados (texto normalizado)
# ----------------------------------------------------------------

_RW = r"(?:repos?|repositorios?|proyectos?)"
_RW_PLURAL = r"(?:repos|repositorios|proyectos)"

_INDEX_WORDS = (
    r"(?:indexad[oa]s?|disponibles?|cargad[oa]s?|registrad[oa]s?|"
    r"indexed|available)"
)

_REPOS_INTENT_RES = [
    re.compile(
        r"^\s*(?:listar?\s+(?:de\s+)?(?:los\s+)?)?" + _RW_PLURAL + r"\s*[?!.]*\s*$"
    ),
    re.compile(r"\bcuant[oa]s\s+(?:\w+\s+){0,3}?" + _RW + r"\b"),
    re.compile(r"\b" + _RW + r"\s+(?:\w+\s+){0,3}?" + _INDEX_WORDS + r"\b"),
    re.compile(
        r"\b(?:que|cuales?)\s+(?:son\s+)?(?:los\s+)?"
        + _RW_PLURAL
        + r"\s+(?:\w+\s+){0,2}?(?:hay|tienes|tenes|existen|conoces|"
        r"manejas|tenemos|tengo|estan|son)\b"
    ),
    re.compile(
        r"\b(?:cuales\s+son|dime|dame|sabes)\s+(?:\w+\s+){0,2}?"
        + _RW_PLURAL
        + r"\s*[?!.]*\s*$"
    ),
    re.compile(
        r"\b(?:lista\w*|listar|muestra\w*|mostrar|mostra\w*|dame|ver|"
        r"ensena\w*|enumera\w*|show|list)\s+(?:\w+\s+){0,3}?"
        + _RW_PLURAL
        + r"\b"
    ),
    re.compile(
        r"\b(?:que|cuales?)\s+(?:\w+\s+){0,2}?(?:tienes|hay|tenes|esta|estan)"
        r"\s+(?:\w+\s+){0,1}?indexad\w*"
    ),
    re.compile(r"\b(?:how\s+many|which|what)\s+(?:\w+\s+){0,2}?repo(?:s|sitories)\b"),
]

_REPO_INDEXED_CHECK_RE = re.compile(
    r"\b(?:indexad[oa]s?|indexed|disponible|cargad[oa])\b"
)

_REPOS_EXCLUDE_RE = re.compile(
    r"\b(?:clase|metodo|funcion|servicio|controller|query|sql|flujo|"
    r"arquitectura|implementa\w*|explica\w*|codigo|grafo|donde|"
    r"usa|usan|llama|llaman)\b"
)

# ----------------------------------------------------------------
# Charla y seguimientos que NO necesitan búsqueda (normalizado)
# ----------------------------------------------------------------

_SMALLTALK_WORDS = {
    "hola",
    "holi",
    "buenas",
    "buenos",
    "dias",
    "tardes",
    "noches",
    "gracias",
    "muchas",
    "mil",
    "ok",
    "okay",
    "okey",
    "vale",
    "listo",
    "perfecto",
    "genial",
    "excelente",
    "chao",
    "adios",
    "hello",
    "hi",
    "thanks",
    "thank",
    "you",
    "dale",
    "bien",
    "super",
    "entendido",
    "ya",
    "si",
    "no",
    "claro",
    "de",
    "una",
    "muy",
    "bueno",
    "buena",
    "crack",
}

_REWRITE_RE = re.compile(
    r"^\s*(?:hazlo|haz\s*lo|resume\w*|resumelo|traduce\w*|traducelo|"
    r"reformula\w*|reescribe\w*|mas\s+corto|mas\s+largo|mas\s+detalle|"
    r"en\s+tabla|en\s+ingles|en\s+espanol|otra\s+vez|repite\w*|"
    r"continua\w*|sigue|simplifica\w*|acorta\w*|amplia\w*)\b"
)

# Preguntas que dependen de la anterior: "¿y eso dónde se usa?"
_FOLLOWUP_RE = re.compile(
    r"^\s*(?:y|e|pero|entonces|tambien|ademas|ahora)\b|"
    r"\b(?:eso|esto|ese|esa|este|esta|esos|esas|ahi|alli|dicho|dicha|"
    r"mismo|misma|anterior|lo\s+que\s+dijiste)\b"
)

# ----------------------------------------------------------------
# Tipo de consulta (texto original)
# ----------------------------------------------------------------

_SQL_RE = re.compile(
    r"\b(select|insert|update|delete|from|join|where|"
    r"group\s+by|stored\s+procedure|procedure|query|queries|"
    r"sql|jpql|tabla|tablas)\b",
    re.IGNORECASE,
)

_ARCHITECTURE_RE = re.compile(
    r"\b(arquitectura|flujo|flujo completo|dependencias|"
    r"cómo se conecta|como se conecta|end.?to.?end|"
    r"end to end|recorrido|pipeline|trazabilidad|"
    r"llama a|llamado desde|quién llama|quien llama)\b",
    re.IGNORECASE,
)

_CODE_RE = re.compile(
    r"\b(implementación|implementacion|implementation|código|codigo|code|clase|class|"
    r"método|metodo|method|service|servicio|controller|dao|repository|"
    r"repositorio|mapper|entity|entidad|bean|funciona|"
    r"cómo funciona|como funciona|explica)\b",
    re.IGNORECASE,
)

# ----------------------------------------------------------------
# Entidades y métodos
# ----------------------------------------------------------------

_CLASS_RE = re.compile(r"\b[A-Z][a-zA-Z0-9_]{2,}\b")

_CAMEL_HUMP_RE = re.compile(r"[a-z0-9][A-Z]")

_BACKTICK_RE = re.compile(r"`([A-Za-z_][\w.]*)(?:\(\))?`")

# save(  obj.save(  — sin espacio antes del paréntesis
_METHOD_CALL_RE = re.compile(r"(?<![\w])([a-zA-Z_][a-zA-Z0-9_]*)\(")

# findByCliente, getUserById — camelCase que empieza en minúscula
_METHOD_CAMEL_RE = re.compile(r"\b([a-z][a-z0-9]*[A-Z][a-zA-Z0-9]*)\b")

_METHOD_STOP = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "return",
    "print",
    "println",
    "new",
}

_ENTITY_IGNORED = {
    "Qué",
    "Que",
    "Cómo",
    "Como",
    "Dónde",
    "Donde",
    "Cuál",
    "Cual",
    "Explícame",
    "Explicame",
    "Quiero",
    "Necesito",
    "Puedes",
    "Podrías",
    "Podrias",
    "Dame",
    "Muéstrame",
    "Muestrame",
    "JavaScript",
    "TypeScript",
    "GitHub",
    "GitLab",
    "OpenWebUI",
    "PostgreSQL",
    "MySQL",
    "SpringBoot",
}

_ENTITY_SUFFIXES = (
    "Impl",
    "Service",
    "Repository",
    "Repo",
    "Dao",
    "DAO",
    "Mapper",
    "Controller",
    "Dto",
    "DTO",
    "Entity",
    "Config",
    "Util",
    "Utils",
    "Factory",
    "Handler",
    "Listener",
    "Task",
    "Job",
    "Processor",
    "Writer",
    "Reader",
    "Interceptor",
    "Filter",
    "Endpoint",
    "Client",
    "Provider",
    "Adapter",
    "Facade",
    "Builder",
    "Validator",
    "Converter",
    "Parser",
    "Scheduler",
    "Resolver",
    "Registry",
    "Cache",
    "Indexer",
    "Extractor",
    "Loader",
    "Saver",
    "Retriever",
    "Updater",
    "Creator",
    "Initializer",
    "Dispatcher",
    "Router",
    "Producer",
    "Consumer",
    "Exception",
    "Request",
    "Response",
)

# ----------------------------------------------------------------
# Lenguaje por extensión (para los bloques de código)
# ----------------------------------------------------------------

_LANG_BY_EXT = {
    ".java": "java",
    ".kt": "kotlin",
    ".py": "python",
    ".sql": "sql",
    ".xml": "xml",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "jsx",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".properties": "properties",
    ".json": "json",
    ".md": "markdown",
    ".go": "go",
    ".cs": "csharp",
    ".html": "html",
    ".sh": "bash",
    ".gradle": "groovy",
}

# Peso de cada fuente en la fusión RRF.
_SOURCE_WEIGHTS = {
    "entity": 1.3,
    "graph": 1.0,
    "augmented": 1.0,
    "relation": 0.6,
    # Archivo cuyo nombre es la clase preguntada, leído del clon
    # (además se fuerza al primer lugar en _fetch_context).
    "clone_exact": 1.5,
    "clone": 0.4,
}

_STRATEGY_HINTS = {
    "architecture": (
        "Explica el flujo paso a paso (quién llama a quién) usando las "
        "relaciones y llamadas que aparecen en el contexto."
    ),
    "entity": (
        "Describe la clase o método: responsabilidad, métodos clave, "
        "dependencias, jerarquía (extends/implements) y queries si las hay."
    ),
    "sql": (
        "Identifica la query, tablas, DAO/Repository, procedimientos "
        "almacenados y el método Java que los usa."
    ),
    "code": "Explica qué hace el código relevante y dónde está.",
    "semantic": "Explica qué hace el código relevante y dónde está.",
}


# ============================================================
# FILTER
# ============================================================


class Filter:

    # ========================================================
    # CONFIGURACIÓN (editable desde Open WebUI)
    # ========================================================

    class Valves(BaseModel):

        priority: int = Field(default=0, description="Orden del filtro.")

        indexer_url: str = Field(
            default_factory=lambda: os.environ.get(
                "TENNIS_DOC_INDEXER_URL", "http://indexer:8001"
            ),
            description="URL base del indexer.",
        )

        api_key: str = Field(
            default_factory=lambda: os.environ.get("INDEXER_API_KEY", ""),
            description="API key del indexer (Bearer).",
        )

        default_branch: str = Field(
            default_factory=lambda: os.environ.get("TENNIS_DOC_DEFAULT_BRANCH", "prod"),
            description="Rama por defecto.",
        )

        # --- Comandos ------------------------------------------------

        enable_commands: bool = Field(
            default=True,
            description=(
                "Comandos por texto (repos, indexa, grafo, pdf). Se desactivan "
                "solos si el chat tiene activa la Tool indicada abajo."
            ),
        )

        tools_id: str = Field(
            default="tennis_doc_tools",
            description="ID de la Tool Tennis Doc Tools en Open WebUI.",
        )

        # --- Búsqueda ------------------------------------------------

        graph_limit: int = 20
        graph_depth: int = 2
        augmented_vector_limit: int = 50
        max_files: int = 10
        max_results: int = 20
        max_entity_relations: int = 5
        clone_max_files: int = 15
        clone_max_chars: int = Field(
            default=40000,
            description=(
                "Máximo de caracteres por archivo del clon. Los archivos Java más "
                "grandes se condensan (firmas + métodos relevantes completos)."
            ),
        )
        request_timeout: int = Field(default=45, description="Segundos por llamada.")
        repos_cache_seconds: int = 60

        # --- Contexto ------------------------------------------------

        max_chars_per_chunk: int = 12000

        max_context_tokens: int = Field(
            default=24000,
            description=(
                "Presupuesto de tokens para el contexto. Debe ser menor que el "
                "num_ctx / ventana de contexto del modelo."
            ),
        )

        chars_per_token: float = Field(
            default=3.5, description="Estimación de caracteres por token (código)."
        )

        # --- Conversación --------------------------------------------

        use_history: bool = Field(
            default=True,
            description="Recordar repo/rama y completar preguntas de seguimiento.",
        )

        followup_max_words: int = 10

        # --- Interfaz ------------------------------------------------

        emit_status: bool = True
        emit_citations: bool = True
        debug: bool = True

    def __init__(self):

        self.name = "Tennis Doc RAG"

        self.valves = self.Valves()

        self._local = threading.local()

        self._repos_cache = None

        self._repos_cache_at = 0.0

        self._repos_lock = threading.Lock()

        self._log(f"Filter cargado | indexer={self.valves.indexer_url}")

    # ========================================================
    # LOG / EVENTOS
    # ========================================================

    def _log(self, message):

        if self.valves.debug:

            print(f"[TennisDoc RAG] {message}")

    def _emit(self, event):

        emit = getattr(self._local, "emit", None)

        if emit:

            emit(event)

    def _status(self, description, done=False):

        if self.valves.emit_status:

            self._emit(
                {
                    "type": "status",
                    "data": {"description": description, "done": done},
                }
            )

    # ========================================================
    # HTTP (una sesión por hilo → reutiliza conexiones)
    # ========================================================

    def _session(self):

        session = getattr(self._local, "session", None)

        if session is None:

            session = requests.Session()

            self._local.session = session

        return session

    def _headers(self):

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if self.valves.api_key:

            headers["Authorization"] = f"Bearer {self.valves.api_key}"

        return headers

    def _url(self, endpoint):

        return f"{self.valves.indexer_url.rstrip('/')}{endpoint}"

    def _get(self, endpoint, **kwargs):

        kwargs.setdefault("headers", self._headers())

        kwargs.setdefault("timeout", self.valves.request_timeout)

        return self._session().get(self._url(endpoint), **kwargs)

    def _post(self, endpoint, **kwargs):

        kwargs.setdefault("headers", self._headers())

        kwargs.setdefault("timeout", self.valves.request_timeout)

        return self._session().post(self._url(endpoint), **kwargs)

    # ========================================================
    # REPOS INDEXADOS (con caché)
    # ========================================================

    def _get_repos(self, force=False):
        """
        Lista de {"name": "org/repo", "branch": "prod"|None}.

        Acepta {"repos": ["org/a"]} y
        {"repos": [{"repo": "org/a", "branch": "prod"}]}.
        """

        with self._repos_lock:

            fresh = time.time() - self._repos_cache_at < self.valves.repos_cache_seconds

            if self._repos_cache is not None and fresh and not force:

                return self._repos_cache

        try:

            response = self._get("/repos", timeout=10)

            response.raise_for_status()

            data = response.json()

        except Exception:

            if self._repos_cache is not None:

                return self._repos_cache

            raise

        raw = data.get("repos", []) if isinstance(data, dict) else data

        repos = []

        seen = set()

        for item in raw or []:

            if isinstance(item, dict):

                name = item.get("repo") or item.get("name") or item.get("full_name")

                branch = item.get("branch") or item.get("branches")

                if isinstance(branch, list):

                    branch = ", ".join(str(b) for b in branch)

            else:

                name, branch = item, None

            if not name:

                continue

            key = (str(name), str(branch or ""))

            if key in seen:

                continue

            seen.add(key)

            repos.append({"name": str(name), "branch": branch})

        repos.sort(key=lambda r: r["name"].lower())

        with self._repos_lock:

            self._repos_cache = repos

            self._repos_cache_at = time.time()

        return repos

    def _known_repo_names(self):

        try:

            return sorted({r["name"] for r in self._get_repos()})

        except Exception as exc:

            self._log(f"No pude leer /repos: {exc}")

            return None

    # ========================================================
    # REPO / RAMA
    # ========================================================

    def _repo_mentions(self, query):
        """Candidatos 'org/repo' con filtro heurístico."""

        mentions = []

        for owner, name in _REPO_RE.findall(query):

            candidate = f"{owner}/{name}"

            low = candidate.lower()

            if low in _REPO_STOP:

                continue

            if owner.lower() in {"http:", "https:", "http", "https", "localhost"}:

                continue

            if len(owner) < 2 or len(name) < 2:

                continue

            if owner.isupper() and name.isupper():

                continue

            if re.search(r"\.(java|py|xml|sql|js|ts|kt|md|json|ya?ml)$", low):

                continue

            mentions.append(candidate)

        return mentions

    def _resolve_repo(self, query):
        """
        Devuelve (repo, no_indexado).

        - Si /repos responde, valida contra la lista y acepta
          también el nombre corto ("pagos-api" → "miorg/pagos-api").
        - Si no responde, usa la heurística.
        """

        mentions = self._repo_mentions(query)

        known = self._known_repo_names()

        if not known:

            return (mentions[0] if mentions else None), False

        by_lower = {k.lower(): k for k in known}

        for mention in mentions:

            if mention.lower() in by_lower:

                return by_lower[mention.lower()], False

        lower_q = query.lower()

        for full in known:

            short = full.split("/")[-1].lower()

            # Solo nombres distintivos ("pagos-api", "core_v2") o entre
            # backticks, para no confundir palabras como "backend".
            distinctive = re.search(r"[-_.\d]", short) and len(short) >= 4

            if not distinctive and f"`{short}`" not in lower_q:

                continue

            if re.search(r"(?<![\w.-])" + re.escape(short) + r"(?![\w.-])", lower_q):

                return full, False

        if mentions:

            return mentions[0], True

        return None, False

    def _extract_branch(self, query):

        for match in _BRANCH_RE.finditer(query):

            value = match.group(1).strip(".,;:?!")

            if len(value) >= 2 and value.lower() not in _BRANCH_STOP:

                return value

        return None

    # ========================================================
    # ENTIDADES / MÉTODOS
    # ========================================================

    def _extract_entities(self, query):

        result = []

        for token in _BACKTICK_RE.findall(query):

            name = token.split(".")[0]

            if name[:1].isupper():

                result.append(name)

        for candidate in _CLASS_RE.findall(query):

            if candidate in _ENTITY_IGNORED:

                continue

            if (
                candidate.endswith(_ENTITY_SUFFIXES)
                or "_" in candidate
                or _CAMEL_HUMP_RE.search(candidate)
            ):

                result.append(candidate)

        return list(dict.fromkeys(result))[:5]

    def _extract_methods(self, query):

        result = []

        for token in _BACKTICK_RE.findall(query):

            last = token.split(".")[-1]

            if last[:1].islower():

                result.append(last)

        for name in _METHOD_CALL_RE.findall(query):

            if name.lower() not in _METHOD_STOP and not name[:1].isupper():

                result.append(name)

        for name in _METHOD_CAMEL_RE.findall(query):

            result.append(name)

        return list(dict.fromkeys(result))[:5]

    # ========================================================
    # ANÁLISIS DE LA CONSULTA
    # ========================================================

    def _analyze_query(self, query):

        entities = self._extract_entities(query)

        methods = self._extract_methods(query)

        is_sql = bool(_SQL_RE.search(query))

        is_architecture = bool(_ARCHITECTURE_RE.search(query))

        is_code = bool(_CODE_RE.search(query))

        if is_architecture:

            strategy = "architecture"

        elif entities or methods:

            strategy = "entity"

        elif is_sql:

            strategy = "sql"

        elif is_code:

            strategy = "code"

        else:

            strategy = "semantic"

        return {
            "strategy": strategy,
            "entities": entities,
            "methods": methods,
            "is_sql": is_sql,
            "is_architecture": is_architecture,
            "is_code": is_code,
        }

    def _enrich_query(self, query, analysis):

        parts = [query]

        parts.extend(analysis.get("entities", []))

        parts.extend(analysis.get("methods", []))

        if analysis.get("is_sql"):

            parts.extend(["SQL", "JPQL", "database", "query", "DAO", "repository"])

        if analysis.get("is_architecture"):

            parts.extend(["dependencies", "relationships", "call flow", "architecture"])

        elif analysis.get("strategy") == "entity":

            parts.extend(["class", "implementation", "methods", "dependencies"])

        return " ".join(dict.fromkeys(parts))

    def _extract_keywords(self, query):

        stopwords = {
            "este",
            "esta",
            "estos",
            "estas",
            "para",
            "como",
            "donde",
            "cuando",
            "quien",
            "cual",
            "cuales",
            "desde",
            "hasta",
            "sobre",
            "entre",
            "tambien",
            "también",
            "porque",
            "quiero",
            "puedes",
            "podrias",
            "podrías",
            "explicame",
            "explícame",
            "mostrar",
            "muestra",
            "hacer",
            "hacerme",
            "dame",
            "genera",
            "generar",
            "codigo",
            "código",
            "archivo",
            "archivos",
            "funciona",
            "funcion",
            "función",
            "hace",
            "tiene",
            "clase",
            "metodo",
            "método",
        }

        keywords = []

        for value in re.findall(r"\b[A-Z][a-zA-Z0-9_]{3,}\b", query):

            if value not in keywords:

                keywords.append(value)

        for value in re.findall(r"\b[a-zA-ZáéíóúñÁÉÍÓÚÑ_]{4,}\b", query):

            normalized = value.lower()

            if normalized in stopwords:

                continue

            if any(normalized == x.lower() for x in keywords):

                continue

            keywords.append(value)

        return keywords[:12]

    # ========================================================
    # DETECCIÓN DE INTENCIONES
    # ========================================================

    def _is_smalltalk(self, query):

        norm = _normalize(query)

        if _REWRITE_RE.search(norm) and len(norm.split()) <= 8:

            return True

        words = re.findall(r"[a-z]+", norm)

        return bool(words) and len(words) <= 6 and all(w in _SMALLTALK_WORDS for w in words)

    def _wants_export(self, query, word_re, explicit_re, allow_entities=False):

        norm = _normalize(query)

        if not word_re.search(norm):

            return False

        if _EXPORT_QUESTION_RE.search(norm):

            return False

        if not allow_entities and self._extract_entities(query):

            return False

        if explicit_re.search(norm):

            return True

        return bool(_EXPORT_VERB_RE.search(norm)) and len(norm.split()) <= 10

    def _detect_repos_intent(self, query):
        """
        None si no es sobre el catálogo de repos. Si lo es:
            {"mode": "count" | "list"} o {"mode": "check", "repo": x}
        """

        norm = _normalize(query)

        if len(norm) > 200 or _REPOS_EXCLUDE_RE.search(norm):

            return None

        mentions = self._repo_mentions(query)

        if mentions:

            if _REPO_INDEXED_CHECK_RE.search(norm):

                return {"mode": "check", "repo": mentions[0]}

            return None

        if any(regex.search(norm) for regex in _REPOS_INTENT_RES):

            if re.search(r"\bcuant[oa]s\b|\bhow\s+many\b", norm):

                return {"mode": "count"}

            return {"mode": "list"}

        return None

    def _tools_active(self, body, metadata):

        tool_ids = list(body.get("tool_ids") or [])

        tool_ids += list((metadata or {}).get("tool_ids") or [])

        return self.valves.tools_id in tool_ids

    # ========================================================
    # HISTORIAL
    # ========================================================

    def _history_context(self, messages):
        """Repo, rama y última pregunta de los mensajes previos del usuario."""

        repo = None

        branch = None

        previous_query = None

        for message in reversed(messages[:-1]):

            if message.get("role") != "user":

                continue

            text = _text_of(message.get("content"))

            if not text:

                continue

            if previous_query is None and not self._is_smalltalk(text):

                previous_query = text

            if repo is None:

                found, not_indexed = self._resolve_repo(text)

                if found and not not_indexed:

                    repo = found

            if branch is None:

                branch = self._extract_branch(text)

            if repo and branch and previous_query:

                break

        return repo, branch, previous_query

    # ========================================================
    # BÚSQUEDAS (cada una devuelve su propia lista)
    # ========================================================

    def _search_augmented(self, repo, branch, search_query, errors):

        try:

            response = self._post(
                "/search-augmented",
                json={
                    "query": search_query,
                    "repo": repo,
                    "branch": branch,
                    "max_files": self.valves.max_files,
                    "vector_limit": self.valves.augmented_vector_limit,
                },
            )

            response.raise_for_status()

            data = response.json()

            results = data.get("results", [])

            self._log(
                f"/search-augmented → {len(results)} | files={data.get('files_fetched', 0)}"
            )

            return results

        except Exception as exc:

            errors.append(f"/search-augmented: {exc}")

            self._log(f"Error /search-augmented: {exc}")

            return []

    def _search_graph(self, repo, branch, search_query, errors):

        try:

            response = self._post(
                "/search-graph",
                json={
                    "query": search_query,
                    "repo": repo,
                    "branch": branch,
                    "limit": self.valves.graph_limit,
                    "graph_depth": self.valves.graph_depth,
                },
            )

            response.raise_for_status()

            results = response.json().get("results", [])

            self._log(f"/search-graph → {len(results)}")

            return results

        except Exception as exc:

            errors.append(f"/search-graph: {exc}")

            self._log(f"Error /search-graph: {exc}")

            return []

    def _get_entity(self, name, repo, branch):

        params = {k: v for k, v in {"repo": repo, "branch": branch}.items() if v}

        try:

            response = self._get(f"/graph/entity/{name}", params=params, timeout=15)

            if response.status_code != 200:

                return None

            return response.json()

        except Exception as exc:

            self._log(f"Error entidad {name}: {exc}")

            return None

    def _entity_to_result(self, entity, repo, branch, score, source):

        code = entity.get("code", "")

        if not code:

            return None

        return {
            "score": score,
            "repo": entity.get("repo") or repo or "",
            "branch": entity.get("branch") or branch or "",
            "file_path": entity.get("file_path", ""),
            "language": entity.get("language") or "",
            "text": code,
            "entity_id": entity.get("id"),
            "ast_name": entity.get("name", ""),
            "ast_type": entity.get("type", ""),
            "ast_signature": entity.get("signature", ""),
            "source": source,
        }

    def _search_entities(self, repo, branch, names):
        """Entidades exactas y sus relaciones, en paralelo."""

        names = names[:3]

        if not names:

            return [], []

        with ThreadPoolExecutor(max_workers=len(names)) as pool:

            entities = list(pool.map(lambda n: self._get_entity(n, repo, branch), names))

        entity_results = []

        targets = []

        for name, entity in zip(names, entities):

            if not entity:

                continue

            result = self._entity_to_result(entity, repo, branch, 1.0, "entity")

            if result:

                entity_results.append(result)

                self._log(f"Entidad {name} | chars={len(result['text'])}")

            for relation in (entity.get("relations") or [])[
                : self.valves.max_entity_relations
            ]:

                target = relation.get("target_name")

                if target and target not in names and target not in targets:

                    targets.append(target)

        relation_results = []

        if targets:

            with ThreadPoolExecutor(max_workers=min(8, len(targets))) as pool:

                fetched = list(pool.map(lambda n: self._get_entity(n, repo, branch), targets))

            for entity in fetched:

                if entity:

                    result = self._entity_to_result(entity, repo, branch, 0.95, "graph_relation")

                    if result:

                        relation_results.append(result)

        return entity_results, relation_results

    def _search_clone(self, repo, branch, query, analysis, errors):

        keywords = self._extract_keywords(query)

        if not repo or not (keywords or analysis["entities"] or analysis["methods"]):

            return []

        try:

            response = self._post(
                "/search-clone",
                json={
                    "repo": repo,
                    "branch": branch,
                    "keywords": keywords,
                    "entities": analysis["entities"],
                    "methods": analysis["methods"],
                    "max_files": self.valves.clone_max_files,
                    "max_chars_per_file": self.valves.clone_max_chars,
                },
                timeout=max(60, self.valves.request_timeout),
            )

            response.raise_for_status()

            results = []

            for item in response.json().get("results", []):

                results.append(
                    {
                        "score": 1.0 if item.get("match") == "name" else item.get("keyword_hits", 0) / 100,
                        "repo": repo,
                        "branch": branch,
                        "file_path": item.get("file_path", ""),
                        "language": item.get("language", ""),
                        "text": item.get("content", ""),
                        "source": "clone_exact" if item.get("match") == "name" else "clone",
                        "full_file": True,
                    }
                )

            self._log(
                f"/search-clone → {len(results)} "
                f"| exactos={sum(1 for r in results if r['source'] == 'clone_exact')}"
            )

            return results

        except Exception as exc:

            errors.append(f"/search-clone: {exc}")

            self._log(f"Error /search-clone: {exc}")

            return []

    # ========================================================
    # FUSIÓN (RRF) + DEDUPE
    # ========================================================

    @staticmethod
    def _same_chunk(a, b):

        if a.get("repo") and b.get("repo") and a["repo"] != b["repo"]:

            return False

        if (a.get("file_path") or "") != (b.get("file_path") or ""):

            return False

        if a.get("ast_name") and a.get("ast_name") == b.get("ast_name"):

            return True

        text_a = (a.get("text") or "").strip()

        text_b = (b.get("text") or "").strip()

        short, long_ = sorted((text_a, text_b), key=len)

        if not short:

            return False

        return short[:300] in long_

    def _fuse(self, lists, k=60):
        """
        Reciprocal Rank Fusion: cada fuente aporta
        peso / (k + posición). Los scores crudos de
        cada backend no se comparan entre sí.
        """

        merged = []

        for source, results in lists.items():

            weight = _SOURCE_WEIGHTS.get(source, 1.0)

            ordered = sorted(
                (r for r in results if r.get("text")),
                key=lambda r: float(r.get("score") or 0),
                reverse=True,
            )

            for rank, result in enumerate(ordered):

                contribution = weight / (k + rank + 1)

                existing = next((m for m in merged if self._same_chunk(m, result)), None)

                if existing:

                    existing["_rrf"] += contribution

                    existing["_sources"].add(source)

                    if len(result["text"]) > len(existing["text"]):

                        existing["text"] = result["text"]

                    for key in ("ast_name", "ast_type", "ast_signature", "language"):

                        if not existing.get(key) and result.get(key):

                            existing[key] = result[key]

                    continue

                item = dict(result)

                item["_rrf"] = contribution

                item["_sources"] = {source}

                merged.append(item)

        merged.sort(key=lambda r: r["_rrf"], reverse=True)

        return merged

    # ========================================================
    # ORQUESTADOR
    # ========================================================

    def _fetch_context(self, repo, branch, query, analysis, retrieval_query=None):

        errors = []

        search_query = self._enrich_query(retrieval_query or query, analysis)

        use_graph = analysis["strategy"] in {"architecture", "entity", "code", "sql"}

        entity_names = analysis["entities"]

        self._log(
            f"strategy={analysis['strategy']} | repo={repo} | branch={branch} "
            f"| entities={entity_names} | methods={analysis['methods']}"
        )

        self._status("🔎 Buscando en el código indexado…")

        # Si se nombra una clase o método, el archivo real se busca en el clon
        # en paralelo: el índice puede tener la clase partida o truncada.
        named = bool(repo and (entity_names or analysis["methods"]))

        with ThreadPoolExecutor(max_workers=4) as pool:

            fut_aug = pool.submit(self._search_augmented, repo, branch, search_query, errors)

            fut_graph = (
                pool.submit(self._search_graph, repo, branch, search_query, errors)
                if use_graph
                else None
            )

            fut_entities = (
                pool.submit(self._search_entities, repo, branch, entity_names)
                if entity_names
                else None
            )

            fut_clone = (
                pool.submit(self._search_clone, repo, branch, retrieval_query or query, analysis, errors)
                if named
                else None
            )

            lists = {"augmented": fut_aug.result()}

            if fut_graph:

                lists["graph"] = fut_graph.result()

            if fut_entities:

                lists["entity"], lists["relation"] = fut_entities.result()

            if fut_clone:

                clone_results = fut_clone.result()

                lists["clone_exact"] = [r for r in clone_results if r["source"] == "clone_exact"]

                lists["clone"] = [r for r in clone_results if r["source"] == "clone"]

        # El archivo del clon (completo o condensado) reemplaza a los
        # fragmentos del mismo archivo que trajo el índice.
        exact_files = {r["file_path"] for r in lists.get("clone_exact", [])}

        if exact_files:

            for source in ("augmented", "graph", "entity", "relation"):

                if source in lists:

                    lists[source] = [
                        r for r in lists[source] if r.get("file_path") not in exact_files
                    ]

        total = sum(len(v) for v in lists.values())

        total_chars = sum(len(r.get("text") or "") for v in lists.values() for r in v)

        if repo and not fut_clone and (total < 6 or total_chars < 12000):

            self._status("📂 Buscando por palabras clave en el repositorio…")

            lists["clone"] = self._search_clone(repo, branch, retrieval_query or query, analysis, errors)

        fused = self._fuse(lists)

        # El archivo de la clase preguntada siempre primero, para que el
        # presupuesto de contexto nunca lo deje afuera.
        fused.sort(key=lambda r: "clone_exact" not in r["_sources"])

        fused = fused[: self.valves.max_results]

        self._log(
            f"Contexto final: {len(fused)} fragmentos "
            f"| {sum(len(r['text']) for r in fused)} chars"
        )

        return fused, errors

    # ========================================================
    # CONSTRUIR CONTEXTO
    # ========================================================

    @staticmethod
    def _language_for(result):

        path = (result.get("file_path") or "").lower()

        for ext, lang in _LANG_BY_EXT.items():

            if path.endswith(ext):

                return lang

        return (result.get("language") or "").lower()

    def _build_context(self, results):
        """Devuelve (texto_contexto, resultados_incluidos)."""

        budget = int(self.valves.max_context_tokens * self.valves.chars_per_token)

        parts = []

        used = []

        total = 0

        for result in results:

            text = result.get("text") or ""

            if not text:

                continue

            # Los archivos del clon ya vienen recortados/condensados por el indexer.
            limit = (
                self.valves.clone_max_chars
                if result.get("full_file")
                else self.valves.max_chars_per_chunk
            )

            truncated = len(text) > limit

            text = text[:limit]

            repo = result.get("repo") or "?"

            branch = result.get("branch") or ""

            location = f"{repo}@{branch}" if branch else repo

            header = f"### [{len(parts) + 1}] {location} · {result.get('file_path') or '?'}\n"

            if result.get("ast_name"):

                header += f"Entidad: {result.get('ast_type', '')} {result['ast_name']}".strip()

                header += "\n"

            if result.get("ast_signature"):

                header += f"Firma: {result['ast_signature']}\n"

            fence = "```"

            while fence in text:

                fence += "`"

            remaining = budget - total - len(header) - 40

            if remaining < 1500:

                self._log(f"Presupuesto de contexto agotado ({budget} chars)")

                break

            if len(text) > remaining:

                text = text[:remaining]

                truncated = True

            if truncated:

                text += "\n… (fragmento truncado)"

            block = f"{header}{fence}{self._language_for(result)}\n{text}\n{fence}\n"

            parts.append(block)

            used.append(result)

            total += len(block)

        return "\n".join(parts), used

    # ========================================================
    # MENSAJES DE SISTEMA
    # ========================================================

    @staticmethod
    def _inject_system(body, content):
        """
        Une el contenido al system prompt existente (posición 0).
        Varias plantillas (Mistral, Llama en Ollama) ignoran o
        rechazan un mensaje 'system' que no esté al inicio.
        """

        messages = body.setdefault("messages", [])

        if messages and messages[0].get("role") == "system":

            existing = _text_of(messages[0].get("content"))

            messages[0]["content"] = f"{existing}\n\n{content}".strip()

        else:

            messages.insert(0, {"role": "system", "content": content.strip()})

        return body

    def _rag_prompt(self, scope, analysis, context, markdown_mode, repo_note):

        hint = _STRATEGY_HINTS.get(analysis["strategy"], _STRATEGY_HINTS["code"])

        output_rule = (
            "5. Entrega un documento Markdown completo y bien estructurado "
            "(títulos, tablas cuando aporten, bloques de código para snippets)."
            if markdown_mode
            else "5. Cita archivos como `ruta/Archivo.java` y usa snippets cortos "
            "del contexto cuando ayuden."
        )

        return f"""
Eres Tennis Doc IA, asistente que analiza el código fuente de la organización.

Ámbito: {scope}
Tipo de consulta: {analysis["strategy"]}{repo_note}

Reglas:
1. Responde solo con base en los fragmentos de CONTEXTO. Revísalos todos antes
   de concluir que algo no existe; una clase grande puede venir partida en
   varios fragmentos.
2. No inventes clases, métodos, endpoints, tablas ni comportamientos. Si
   infieres algo, márcalo como inferencia.
3. Si el contexto no alcanza para responder, dilo claramente.
4. {hint}
{output_rule}
6. Termina con "### Fuentes": los archivos que usaste, sin inventar ninguno.

CONTEXTO
========

{context}
"""

    def _no_context_prompt(self, repo, errors, repo_note):

        reason = " porque el indexer no respondió" if errors else ""

        where = f" en `{repo}`" if repo else " en los repositorios indexados"

        return f"""
Eres Tennis Doc IA. Para esta pregunta NO se recuperó código{reason}.{repo_note}

No respondas con conocimiento general como si fuera el código de la
organización. Di en una o dos frases que no encontraste información{where}
y sugiere reformular mencionando la clase, el método o el repositorio
(por ejemplo: "Explícame PagoService en org/repo").
"""

    # ========================================================
    # RESPUESTAS EXACTAS (comandos)
    # ========================================================

    def _command_reply(self, body, metadata, content, echo=True):
        """
        Guarda el resultado para que outlet() lo muestre tal cual.
        Además le pide al modelo que lo repita, por si outlet no
        encuentra la respuesta (doble seguro).
        """

        metadata = metadata or {}

        key = metadata.get("chat_id") or body.get("chat_id") or "_default"

        now = time.time()

        for k in [k for k, v in _PENDING.items() if now - v["at"] > _PENDING_TTL]:

            _PENDING.pop(k, None)

        _PENDING[key] = {
            "content": content,
            "at": now,
            "message_id": metadata.get("message_id"),
            "turns": _count_turns(body.get("messages")),
        }

        if echo:

            instruction = (
                "Responde EXACTAMENTE con el siguiente contenido, sin agregar ni "
                f"quitar nada:\n\n{content}"
            )

        else:

            instruction = "Responde solamente: 'Listo.'"

        return self._inject_system(body, instruction)

    # ========================================================
    # COMANDO: INDEXAR
    # ========================================================

    def _trigger_index(self, body, metadata, repo, branch):

        self._status(f"🗂️ Iniciando indexación de {repo}…")

        try:

            response = self._post("/index", json={"repo": repo, "branch": branch}, timeout=10)

            response.raise_for_status()

            data = response.json()

            content = (
                "🗂️ **Indexación iniciada**\n\n"
                f"**Repositorio:** `{repo}`\n"
                f"**Rama:** `{branch}`\n"
                f"**Estado:** `{data.get('status', 'ok')}`\n\n"
                "Cuando finalice puedes hacer preguntas sobre el repositorio."
            )

            with self._repos_lock:

                self._repos_cache_at = 0.0

        except Exception as exc:

            content = f"❌ Error iniciando la indexación de `{repo}` @ `{branch}`:\n\n`{exc}`"

        return self._command_reply(body, metadata, content)

    # ========================================================
    # COMANDO: GRAFO
    # ========================================================

    def _show_graph(self, body, metadata, class_name, repo, branch):

        self._status(f"🕸️ Consultando el grafo de {class_name}…")

        entity = self._get_entity(class_name, repo, branch)

        if not entity:

            content = f"No encontré `{class_name}` en el grafo" + (
                f" de `{repo}`." if repo else "."
            )

            return self._command_reply(body, metadata, content)

        lines = [
            f"## Grafo: `{class_name}`",
            "",
            f"**Tipo:** `{entity.get('type', 'Unknown')}`  ",
            f"**Archivo:** `{entity.get('file_path', 'N/A')}`",
        ]

        if entity.get("signature"):

            lines.append(f"**Firma:** `{entity['signature']}`")

        lines.append("")

        relations = entity.get("relations") or []

        if relations:

            lines.append("### Relaciones")

            for relation in relations:

                lines.append(
                    f"- **{relation.get('rel_type', 'REL')}** → "
                    f"`{relation.get('target_name', '?')}` "
                    f"({relation.get('target_type', '?')})"
                )

        else:

            lines.append("No se encontraron relaciones directas.")

        return self._command_reply(body, metadata, "\n".join(lines))

    # ========================================================
    # COMANDO: PDF
    # ========================================================

    @staticmethod
    def _store_file(data, filename, content_type, user_id):
        """
        Guarda el archivo en el almacenamiento de Open WebUI y
        devuelve su URL. Devuelve None si la API interna no está
        disponible (cambia entre versiones).
        """

        try:

            from open_webui.models.files import FileForm, Files
            from open_webui.storage.provider import Storage

            file_id = str(uuid.uuid4())

            stored_name = f"{file_id}_{filename}"

            try:

                _, path = Storage.upload_file(io.BytesIO(data), stored_name, {})

            except TypeError:

                _, path = Storage.upload_file(io.BytesIO(data), stored_name)

            Files.insert_new_file(
                user_id,
                FileForm(
                    id=file_id,
                    filename=filename,
                    path=path,
                    meta={
                        "name": filename,
                        "content_type": content_type,
                        "size": len(data),
                    },
                ),
            )

            return f"/api/v1/files/{file_id}/content"

        except Exception as exc:

            print(f"[TennisDoc RAG] No pude guardar el archivo en Open WebUI: {exc}")

            return None

    def _generate_pdf(self, body, metadata, user, repo, branch):

        messages = body.get("messages", [])

        previous = next(
            (m for m in reversed(messages[:-1]) if m.get("role") == "assistant"),
            None,
        )

        if not previous:

            return self._command_reply(
                body, metadata, "No hay una respuesta anterior para convertir a PDF."
            )

        self._status("📄 Generando PDF…")

        title = f"Respuesta_{repo.replace('/', '_')}" if repo else "Respuesta"

        try:

            response = self._post(
                "/pdf",
                json={
                    "title": title,
                    "content": _text_of(previous.get("content")),
                    "repo": repo,
                    "branch": branch,
                },
                timeout=60,
            )

            response.raise_for_status()

            pdf = response.content

        except Exception as exc:

            return self._command_reply(body, metadata, f"❌ Error generando PDF: `{exc}`")

        url = self._store_file(pdf, f"{title}.pdf", "application/pdf", (user or {}).get("id"))

        if url:

            content = f"📄 PDF generado: [Descargar {title}.pdf]({url})"

            return self._command_reply(body, metadata, content)

        # Fallback: enlace embebido (solo lo agrega outlet, es muy largo
        # para que el modelo lo repita).
        b64 = base64.b64encode(pdf).decode("utf-8")

        content = (
            "📄 PDF generado.\n\n"
            f'<a href="data:application/pdf;base64,{b64}" download="{title}.pdf">'
            "Descargar PDF</a>"
        )

        return self._command_reply(body, metadata, content, echo=False)

    # ========================================================
    # COMANDO: REPOS INDEXADOS
    # ========================================================

    def _answer_repos(self, body, intent):
        """
        Le pasa al LLM los datos reales de /repos. La pregunta del
        usuario se conserva para que responda con naturalidad.
        """

        self._status("📚 Consultando repositorios indexados…")

        try:

            repos = self._get_repos(force=True)

        except Exception as exc:

            self._log(f"Error /repos: {exc}")

            return self._inject_system(
                body,
                "Eres Tennis Doc IA. El usuario preguntó por los repositorios "
                f"indexados, pero el indexer no respondió (error: {exc}). Díselo "
                "en una frase y no inventes ningún repositorio.",
            )

        names = sorted({r["name"] for r in repos})

        listing = (
            "\n".join(
                f"- {r['name']}" + (f" (rama: {r['branch']})" if r.get("branch") else "")
                for r in repos
            )
            or "(ninguno)"
        )

        check = ""

        if intent.get("mode") == "check":

            target = intent["repo"].lower()

            found = next(
                (n for n in names if n.lower() == target or n.lower().split("/")[-1] == target),
                None,
            )

            check = f"\nVERIFICACIÓN: `{intent['repo']}` {'SÍ' if found else 'NO'} está indexado.\n"

            if not found:

                check += f"Sugiere indexarlo con: `indexa {intent['repo']}`\n"

        self._log(f"/repos → {len(names)} | modo={intent.get('mode')}")

        return self._inject_system(
            body,
            f"""
Eres Tennis Doc IA. El usuario pregunta por los repositorios indexados.
Datos REALES obtenidos ahora de /repos:

TOTAL DE REPOSITORIOS: {len(names)}

{listing}
{check}
Reglas: responde exactamente lo que pregunta (cantidad, lista o si un repo
está indexado), usa solo estos datos, muestra cada repo en `código`, y si no
hay ninguno sugiere `indexa org/repositorio`. Sé breve.
""",
        )

    # ========================================================
    # CITAS EN LA INTERFAZ
    # ========================================================

    def _emit_citations(self, results):

        if not self.valves.emit_citations:

            return

        seen = set()

        for result in results:

            name = f"{result.get('repo') or '?'} · {result.get('file_path') or '?'}"

            if name in seen:

                continue

            seen.add(name)

            self._emit(
                {
                    "type": "citation",
                    "data": {
                        "document": [(result.get("text") or "")[:4000]],
                        "metadata": [
                            {
                                "source": name,
                                "repo": result.get("repo"),
                                "branch": result.get("branch"),
                                "file_path": result.get("file_path"),
                            }
                        ],
                        "source": {"name": name},
                    },
                }
            )

    # ========================================================
    # PROCESO PRINCIPAL (síncrono, corre en un hilo aparte)
    # ========================================================

    def _process(self, body, user, metadata, emit=None):

        # Emisor por hilo: evita mezclar eventos entre usuarios concurrentes.
        self._local.emit = emit

        messages = body.get("messages") or []

        if not messages or messages[-1].get("role") != "user":

            return body

        query = _text_of(messages[-1].get("content"))

        if not query:

            return body

        self._log(f"Query: {query[:150]}")

        commands = self.valves.enable_commands and not self._tools_active(body, metadata)

        # -------------------------------------------------
        # COMANDOS
        # -------------------------------------------------

        if commands:

            index_match = _INDEX_CMD_RE.match(query)

            if index_match:

                repo = index_match.group(1)

                branch = index_match.group(2) or self.valves.default_branch

                return self._trigger_index(body, metadata, repo, branch)

            if self._wants_export(query, _PDF_WORD_RE, _PDF_EXPLICIT_RE):

                repo, _ = self._resolve_repo(query)

                branch = self._extract_branch(query) or self.valves.default_branch

                return self._generate_pdf(body, metadata, user, repo, branch)

            graph_match = _GRAPH_RE.search(query)

            if graph_match:

                repo, _ = self._resolve_repo(query)

                return self._show_graph(
                    body,
                    metadata,
                    graph_match.group(1),
                    repo,
                    self._extract_branch(query) or self.valves.default_branch,
                )

            repos_intent = self._detect_repos_intent(query)

            if repos_intent:

                self._log(f"Intención repos: {repos_intent}")

                return self._answer_repos(body, repos_intent)

        # -------------------------------------------------
        # CHARLA / REESCRITURA → sin búsqueda
        # -------------------------------------------------

        if self._is_smalltalk(query):

            self._log("Charla o seguimiento de formato: sin búsqueda")

            return body

        markdown_mode = self._wants_export(
            query, _MD_WORD_RE, _MD_EXPLICIT_RE, allow_entities=True
        )

        # -------------------------------------------------
        # REPO / RAMA / SEGUIMIENTO
        # -------------------------------------------------

        analysis = self._analyze_query(query)

        repo, not_indexed = self._resolve_repo(query)

        branch = self._extract_branch(query)

        retrieval_query = None

        if self.valves.use_history:

            hist_repo, hist_branch, previous_query = self._history_context(messages)

            if not repo and hist_repo:

                repo = hist_repo

                self._log(f"Repo tomado del historial: {repo}")

            if not branch and hist_branch:

                branch = hist_branch

            is_followup = (
                previous_query
                and not analysis["entities"]
                and not analysis["methods"]
                and len(query.split()) <= self.valves.followup_max_words
                and (
                    _FOLLOWUP_RE.search(_normalize(query))
                    or len(query.split()) <= 4
                )
            )

            if is_followup:

                retrieval_query = f"{previous_query} {query}"

                prev_analysis = self._analyze_query(previous_query)

                analysis["entities"] = prev_analysis["entities"]

                analysis["methods"] = prev_analysis["methods"]

                if analysis["strategy"] == "semantic":

                    analysis["strategy"] = prev_analysis["strategy"]

                self._log("Pregunta de seguimiento: se combina con la anterior")

        branch = branch or self.valves.default_branch

        repo_note = (
            f"\nNota: `{repo}` no aparece entre los repositorios indexados."
            if not_indexed
            else ""
        )

        # -------------------------------------------------
        # RAG
        # -------------------------------------------------

        results, errors = self._fetch_context(repo, branch, query, analysis, retrieval_query)

        if not results:

            self._log("No se recuperó contexto")

            return self._inject_system(body, self._no_context_prompt(repo, errors, repo_note))

        context, used = self._build_context(results)

        scope = f"repositorio `{repo}`, rama `{branch}`" if repo else f"todos los repos, rama `{branch}`"

        self._inject_system(
            body,
            self._rag_prompt(scope, analysis, context, markdown_mode, repo_note),
        )

        self._emit_citations(used)

        self._log(f"Contexto inyectado: {len(used)} fragmentos | {len(context)} chars")

        return body

    # ========================================================
    # INLET / OUTLET (Open WebUI)
    # ========================================================

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
        __metadata__: Optional[dict] = None,
    ) -> dict:

        loop = asyncio.get_running_loop()

        emit = None

        if __event_emitter__:

            def emit(event):

                try:

                    asyncio.run_coroutine_threadsafe(__event_emitter__(event), loop)

                except Exception:

                    pass

        try:

            # Las llamadas HTTP son bloqueantes: se hacen en otro hilo
            # para no congelar Open WebUI mientras se busca.
            body = await asyncio.to_thread(
                self._process, body, __user__, __metadata__, emit
            )

        except Exception as exc:

            print(f"[TennisDoc RAG] ERROR inlet: {exc}")

        if __event_emitter__ and self.valves.emit_status:

            await __event_emitter__({"type": "status", "data": {"description": "", "done": True, "hidden": True}})

        return body

    async def outlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __metadata__: Optional[dict] = None,
    ) -> dict:
        """Muestra tal cual el resultado de un comando (grafo, índice, PDF)."""

        try:

            key = body.get("chat_id") or (__metadata__ or {}).get("chat_id") or "_default"

            pending = _PENDING.get(key)

            if not pending:

                return body

            # Solo la respuesta de ESTE comando, nunca una posterior.
            message_id = body.get("id")

            if pending.get("message_id") and message_id:

                matches = pending["message_id"] == message_id

            else:

                matches = _count_turns(body.get("messages")) == pending["turns"] + 1

            if not matches:

                if time.time() - pending["at"] > 120:

                    _PENDING.pop(key, None)

                return body

            _PENDING.pop(key, None)

            for message in reversed(body.get("messages") or []):

                if message.get("role") == "assistant":

                    message["content"] = pending["content"]

                    break

        except Exception as exc:

            print(f"[TennisDoc RAG] ERROR outlet: {exc}")

        return body
