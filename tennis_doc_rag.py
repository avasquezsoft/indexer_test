"""
Tennis Doc RAG
Version: 2.0

Filtro para Open WebUI que recupera contexto desde Tennis Doc IA.

Arquitectura:

Open WebUI
    |
    v
tennis-doc-rag.py
    |
    +-- /search-augmented
    |       |
    |       +-- recuperación semántica
    |       +-- archivos completos/contexto ampliado
    |
    +-- /search-graph
    |       |
    |       +-- Qdrant
    |       +-- Neo4j
    |       +-- exact match
    |       +-- ranking híbrido
    |
    +-- /graph/entity/{name}
    |       |
    |       +-- entidad completa
    |       +-- relaciones
    |
    +-- /search-clone
            |
            +-- fallback por keywords

                    |
                    v
              Contexto final
                    |
                    v
                   LLM
"""

import base64
import os
import re
import requests


# ============================================================
# REGEX
# ============================================================

_REPO_RE = re.compile(
    r"(?<![\w.-])([\w.-]+)/([\w.-]+)(?![\w.-])"
)

_BRANCH_RE = re.compile(
    r"(?:rama|branch)\s+([A-Za-z0-9._/-]+)",
    re.IGNORECASE,
)

_PDF_RE = re.compile(
    r"(?:\b(?:genera?(?:r|me)?|crea?(?:r|me)?|descarga?(?:r|me)?|"
    r"exporta?(?:r|me)?|guarda?(?:r|me)?|sacar|dame|mostra?(?:r|me)?|"
    r"hazme|preparame|armame)\b).*?\bpdf\b",
    re.IGNORECASE,
)

_MD_RE = re.compile(
    r"(?:\b(?:genera?(?:r|me)?|crea?(?:r|me)?|descarga?(?:r|me)?|"
    r"exporta?(?:r|me)?|guarda?(?:r|me)?|sacar|dame|mostra?(?:r|me)?|"
    r"hazme|preparame|armame)\b).*?\b(?:markdown|md|\.md)\b",
    re.IGNORECASE,
)

_GRAPH_RE = re.compile(
    r"^\s*grafo\s+([A-Z][a-zA-Z0-9_]*)\s*$",
    re.IGNORECASE,
)

_REPOS_LIST_RE = re.compile(
    r"^\s*(repos|repositorios|listar\s+repos?)\s*$",
    re.IGNORECASE,
)

_SQL_RE = re.compile(
    r"\b(select|insert|update|delete|from|join|where|"
    r"group\s+by|stored\s+procedure|procedure|query|queries|"
    r"sql|jpql)\b",
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
    r"\b(implementación|implementation|código|code|clase|class|"
    r"método|method|service|servicio|controller|dao|repository|"
    r"repositorio|mapper|entity|entidad|bean|funciona|"
    r"cómo funciona|como funciona|explica)\b",
    re.IGNORECASE,
)

_CLASS_RE = re.compile(
    r"\b[A-Z][a-zA-Z0-9_]{2,}\b"
)

_METHOD_RE = re.compile(
    r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
)


# ============================================================
# FILTER
# ============================================================

class Filter:

    def __init__(self):

        self.name = "Tennis Doc RAG"

        self.valves = self.Valves()

        print(
            "[TennisDoc RAG] "
            "Filter cargado correctamente "
            f"| indexer={self.valves.indexer_url}"
        )

    # ========================================================
    # CONFIGURACIÓN
    # ========================================================

    class Valves:

        def __init__(self):

            self.indexer_url = os.environ.get(
                "TENNIS_DOC_INDEXER_URL",
                "http://indexer:8001",
            )

            self.api_key = os.environ.get(
                "INDEXER_API_KEY",
                "",
            )

            self.default_branch = os.environ.get(
                "TENNIS_DOC_DEFAULT_BRANCH",
                "prod",
            )

            # ------------------------------------------------
            # SEARCH GRAPH
            # ------------------------------------------------

            # Top final que devuelve rag_engine.py.
            self.graph_limit = 20

            self.graph_depth = 2

            # ------------------------------------------------
            # SEARCH AUGMENTED
            # ------------------------------------------------

            # Mantener los valores originales.
            self.augmented_vector_limit = 50

            self.max_files = 10

            # ------------------------------------------------
            # RESULTADOS
            # ------------------------------------------------

            self.max_results = 20

            # ------------------------------------------------
            # CONTEXTO
            # ------------------------------------------------

            # Importante para clases Java grandes.
            self.max_chars_per_chunk = 12000

            # Límite global razonablemente grande.
            self.max_context_chars = 90000

            # ------------------------------------------------
            # GRAPH ENTITY
            # ------------------------------------------------

            self.max_entity_relations = 5

            # ------------------------------------------------
            # CLONE FALLBACK
            # ------------------------------------------------

            self.clone_max_files = 15

            self.clone_max_chars = 12000

    # ========================================================
    # HTTP
    # ========================================================

    def _headers(self):

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if self.valves.api_key:

            headers["Authorization"] = (
                f"Bearer {self.valves.api_key}"
            )

        return headers

    def _get(
        self,
        endpoint,
        **kwargs,
    ):

        kwargs.setdefault(
            "headers",
            self._headers(),
        )

        return requests.get(
            f"{self.valves.indexer_url}{endpoint}",
            **kwargs,
        )

    def _post(
        self,
        endpoint,
        **kwargs,
    ):

        kwargs.setdefault(
            "headers",
            self._headers(),
        )

        return requests.post(
            f"{self.valves.indexer_url}{endpoint}",
            **kwargs,
        )

    # ========================================================
    # REPO / BRANCH
    # ========================================================

    def _extract_repo_branch(
        self,
        query,
    ):

        repo = None

        branch = (
            self.valves.default_branch
        )

        match = _REPO_RE.search(
            query
        )

        if match:

            candidate = (
                f"{match.group(1)}/"
                f"{match.group(2)}"
            )

            if not candidate.startswith(
                (
                    "http/",
                    "https/",
                    "localhost/",
                )
            ):

                repo = candidate

        branch_match = _BRANCH_RE.search(
            query
        )

        if branch_match:

            branch = (
                branch_match.group(1)
            )

        return repo, branch

    # ========================================================
    # ENTIDADES
    # ========================================================

    def _extract_entities(
        self,
        query,
    ):

        candidates = _CLASS_RE.findall(
            query
        )

        ignored = {
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
        }

        suffixes = (
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
        )

        result = []

        for candidate in candidates:

            if candidate in ignored:
                continue

            if (
                candidate.endswith(
                    suffixes
                )
                or "_" in candidate
            ):

                result.append(
                    candidate
                )

        return list(
            dict.fromkeys(result)
        )[:5]

    # ========================================================
    # MÉTODOS
    # ========================================================

    def _extract_methods(
        self,
        query,
    ):

        methods = _METHOD_RE.findall(
            query
        )

        return list(
            dict.fromkeys(methods)
        )[:5]

    # ========================================================
    # ANALIZAR CONSULTA
    # ========================================================

    def _analyze_query(
        self,
        query,
    ):

        entities = (
            self._extract_entities(
                query
            )
        )

        methods = (
            self._extract_methods(
                query
            )
        )

        is_sql = bool(
            _SQL_RE.search(query)
        )

        is_architecture = bool(
            _ARCHITECTURE_RE.search(
                query
            )
        )

        is_code = bool(
            _CODE_RE.search(query)
        )

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

    # ========================================================
    # ENRIQUECER QUERY
    # ========================================================

    def _enrich_query(
        self,
        query,
        analysis,
    ):

        parts = [query]

        parts.extend(
            analysis.get(
                "entities",
                [],
            )
        )

        parts.extend(
            analysis.get(
                "methods",
                [],
            )
        )

        if analysis.get(
            "is_sql"
        ):

            parts.extend(
                [
                    "SQL",
                    "JPQL",
                    "database",
                    "query",
                    "DAO",
                    "repository",
                ]
            )

        if analysis.get(
            "is_architecture"
        ):

            parts.extend(
                [
                    "dependencies",
                    "relationships",
                    "call flow",
                    "architecture",
                ]
            )

        elif (
            analysis.get(
                "strategy"
            )
            == "entity"
        ):

            parts.extend(
                [
                    "class",
                    "implementation",
                    "methods",
                    "dependencies",
                ]
            )

        return " ".join(
            dict.fromkeys(parts)
        )

    # ========================================================
    # AGREGAR RESULTADO
    # ========================================================

    def _add_result(
        self,
        results,
        seen,
        result,
        source,
    ):

        text = result.get(
            "text",
            "",
        )

        if not text:

            return

        key = (
            result.get(
                "repo",
                "",
            ),
            result.get(
                "branch",
                "",
            ),
            result.get(
                "file_path",
                "",
            ),
            text[:160],
        )

        # ----------------------------------------------------
        # DUPLICADO
        # ----------------------------------------------------

        if key in seen:

            for existing in results:

                existing_key = (
                    existing.get(
                        "repo",
                        "",
                    ),
                    existing.get(
                        "branch",
                        "",
                    ),
                    existing.get(
                        "file_path",
                        "",
                    ),
                    existing.get(
                        "text",
                        "",
                    )[:160],
                )

                if (
                    existing_key
                    == key
                ):

                    sources = set(
                        existing.get(
                            "_sources",
                            [],
                        )
                    )

                    sources.add(
                        source
                    )

                    existing[
                        "_sources"
                    ] = list(
                        sources
                    )

                    # Mantener el score más alto.
                    existing[
                        "_final_score"
                    ] = max(
                        float(
                            existing.get(
                                "_final_score",
                                0,
                            )
                        ),
                        float(
                            result.get(
                                "score",
                                0,
                            )
                        ),
                    )

                    break

            return

        # ----------------------------------------------------
        # NUEVO
        # ----------------------------------------------------

        new_result = dict(
            result
        )

        new_result[
            "_sources"
        ] = [source]

        new_result[
            "_final_score"
        ] = float(
            result.get(
                "score",
                0,
            )
        )

        results.append(
            new_result
        )

        seen.add(
            key
        )

    # ========================================================
    # SEARCH GRAPH
    #
    # IMPORTANTE:
    #
    # rag_engine.py es responsable del ranking híbrido.
    #
    # Aquí NO hacemos un segundo reranking.
    # ========================================================

    def _search_graph(
        self,
        repo,
        branch,
        query,
        analysis,
        results,
        seen,
    ):

        should_use_graph = (
            analysis["strategy"]
            in {
                "architecture",
                "entity",
                "code",
                "sql",
            }
            or bool(
                analysis["entities"]
            )
        )

        if not should_use_graph:

            return

        search_query = (
            self._enrich_query(
                query,
                analysis,
            )
        )

        try:

            response = self._post(
                "/search-graph",
                json={
                    "query": search_query,
                    "repo": repo,
                    "branch": branch,
                    "limit": (
                        self.valves.graph_limit
                    ),
                    "graph_depth": (
                        self.valves.graph_depth
                    ),
                },
                timeout=45,
            )

            response.raise_for_status()

            data = response.json()

            graph_results = (
                data.get(
                    "results",
                    [],
                )
            )

            for result in graph_results:

                self._add_result(
                    results,
                    seen,
                    result,
                    "graph",
                )

            print(
                "[TennisDoc RAG] "
                f"/search-graph → "
                f"{len(graph_results)} resultados"
            )

        except Exception as exc:

            print(
                "[TennisDoc RAG] "
                f"Error /search-graph: "
                f"{exc}"
            )

    # ========================================================
    # SEARCH AUGMENTED
    #
    # Esta búsqueda es importante para clases grandes.
    #
    # Recupera contexto de archivos relevantes.
    # ========================================================

    def _search_augmented(
        self,
        repo,
        branch,
        query,
        analysis,
        results,
        seen,
    ):

        search_query = (
            self._enrich_query(
                query,
                analysis,
            )
        )

        try:

            response = self._post(
                "/search-augmented",
                json={
                    "query": search_query,
                    "repo": repo,
                    "branch": branch,
                    "max_files": (
                        self.valves.max_files
                    ),
                    "vector_limit": (
                        self.valves.augmented_vector_limit
                    ),
                },
                timeout=45,
            )

            response.raise_for_status()

            data = response.json()

            augmented_results = (
                data.get(
                    "results",
                    [],
                )
            )

            for result in augmented_results:

                self._add_result(
                    results,
                    seen,
                    result,
                    "augmented",
                )

            print(
                "[TennisDoc RAG] "
                f"/search-augmented → "
                f"{len(augmented_results)} resultados "
                f"| files="
                f"{data.get('files_fetched', 0)}"
            )

        except Exception as exc:

            print(
                "[TennisDoc RAG] "
                f"Error /search-augmented: "
                f"{exc}"
            )

    # ========================================================
    # ENTITY EXACTA
    #
    # Si el usuario menciona una clase explícitamente,
    # recuperamos la entidad completa del grafo.
    # ========================================================

    def _search_entities(
        self,
        repo,
        branch,
        analysis,
        results,
        seen,
    ):

        if not repo:

            return

        entities = analysis.get(
            "entities",
            [],
        )

        for class_name in entities[:3]:

            try:

                response = self._get(
                    f"/graph/entity/{class_name}",
                    params={
                        "repo": repo,
                        "branch": branch,
                    },
                    timeout=15,
                )

                if response.status_code != 200:

                    continue

                entity = response.json()

                code = entity.get(
                    "code",
                    "",
                )

                # ------------------------------------------------
                # ENTIDAD PRINCIPAL
                # ------------------------------------------------

                if code:

                    result = {
                        "score": 1.0,
                        "repo": repo,
                        "branch": branch,
                        "file_path": entity.get(
                            "file_path",
                            "",
                        ),
                        "language": "java",
                        "text": code,
                        "entity_id": entity.get(
                            "id"
                        ),
                        "ast_name": entity.get(
                            "name",
                            "",
                        ),
                        "ast_type": entity.get(
                            "type",
                            "",
                        ),
                        "ast_signature": entity.get(
                            "signature",
                            "",
                        ),
                        "source": "entity",
                    }

                    self._add_result(
                        results,
                        seen,
                        result,
                        "entity",
                    )

                    print(
                        "[TennisDoc RAG] "
                        f"Entidad encontrada: "
                        f"{class_name} "
                        f"| chars={len(code)}"
                    )

                # ------------------------------------------------
                # RELACIONES DIRECTAS
                # ------------------------------------------------

                relations = entity.get(
                    "relations",
                    [],
                )

                for relation in relations[
                    : self.valves.max_entity_relations
                ]:

                    target_name = (
                        relation.get(
                            "target_name"
                        )
                    )

                    if not target_name:

                        continue

                    try:

                        target_response = self._get(
                            f"/graph/entity/{target_name}",
                            params={
                                "repo": repo,
                                "branch": branch,
                            },
                            timeout=15,
                        )

                        if (
                            target_response.status_code
                            != 200
                        ):

                            continue

                        target = (
                            target_response.json()
                        )

                        target_code = (
                            target.get(
                                "code",
                                "",
                            )
                        )

                        if not target_code:

                            continue

                        result = {
                            "score": 0.95,
                            "repo": repo,
                            "branch": branch,
                            "file_path": target.get(
                                "file_path",
                                "",
                            ),
                            "language": "java",
                            "text": target_code,
                            "entity_id": target.get(
                                "id"
                            ),
                            "ast_name": target.get(
                                "name",
                                "",
                            ),
                            "ast_type": target.get(
                                "type",
                                "",
                            ),
                            "ast_signature": target.get(
                                "signature",
                                "",
                            ),
                            "source": "graph_relation",
                        }

                        self._add_result(
                            results,
                            seen,
                            result,
                            "relation",
                        )

                    except Exception as relation_exc:

                        print(
                            "[TennisDoc RAG] "
                            f"Error relación "
                            f"{target_name}: "
                            f"{relation_exc}"
                        )

            except Exception as exc:

                print(
                    "[TennisDoc RAG] "
                    f"Error buscando entidad "
                    f"{class_name}: "
                    f"{exc}"
                )

    # ========================================================
    # FALLBACK CLONE
    # ========================================================

    def _search_clone_fallback(
        self,
        repo,
        branch,
        query,
        results,
        seen,
    ):

        if not repo:

            return

        total_chars = sum(
            len(
                r.get(
                    "text",
                    "",
                )
            )
            for r in results
        )

        # Solo utilizar fallback si realmente falta contexto.
        if (
            len(results) >= 6
            and total_chars >= 12000
        ):

            return

        keywords = (
            self._extract_keywords(
                query
            )
        )

        if not keywords:

            return

        try:

            response = self._post(
                "/search-clone",
                json={
                    "repo": repo,
                    "branch": branch,
                    "keywords": keywords,
                    "max_files": (
                        self.valves.clone_max_files
                    ),
                    "max_chars_per_file": (
                        self.valves.clone_max_chars
                    ),
                },
                timeout=60,
            )

            response.raise_for_status()

            data = response.json()

            clone_results = (
                data.get(
                    "results",
                    [],
                )
            )

            for result in clone_results:

                normalized = {
                    "score": 0.35,
                    "repo": repo,
                    "branch": branch,
                    "file_path": result.get(
                        "file_path",
                        "",
                    ),
                    "language": "java",
                    "text": result.get(
                        "content",
                        "",
                    ),
                    "source": "clone",
                }

                self._add_result(
                    results,
                    seen,
                    normalized,
                    "clone",
                )

            print(
                "[TennisDoc RAG] "
                f"Fallback clone → "
                f"{len(clone_results)} resultados"
            )

        except Exception as exc:

            print(
                "[TennisDoc RAG] "
                f"Error /search-clone: "
                f"{exc}"
            )

    # ========================================================
    # ORQUESTADOR PRINCIPAL
    # ========================================================

    def _fetch_context(
        self,
        repo,
        branch,
        query,
    ):

        analysis = (
            self._analyze_query(
                query
            )
        )

        results = []

        seen = set()

        print(
            "[TennisDoc RAG] "
            f"strategy={analysis['strategy']} "
            f"| repo={repo} "
            f"| branch={branch} "
            f"| entities={analysis['entities']} "
            f"| methods={analysis['methods']}"
        )

        # ====================================================
        # 1. SEARCH AUGMENTED
        #
        # Siempre se utiliza.
        #
        # Es especialmente importante para clases grandes.
        # ====================================================

        self._search_augmented(
            repo,
            branch,
            query,
            analysis,
            results,
            seen,
        )

        # ====================================================
        # 2. SEARCH GRAPH
        #
        # Para consultas donde las relaciones importan.
        # ====================================================

        self._search_graph(
            repo,
            branch,
            query,
            analysis,
            results,
            seen,
        )

        # ====================================================
        # 3. ENTITY EXACTA
        #
        # Para preguntas sobre una clase concreta.
        # ====================================================

        if analysis["entities"]:

            self._search_entities(
                repo,
                branch,
                analysis,
                results,
                seen,
            )

        # ====================================================
        # 4. FALLBACK
        # ====================================================

        self._search_clone_fallback(
            repo,
            branch,
            query,
            results,
            seen,
        )

        # ====================================================
        # ORDEN
        #
        # NO hacemos un reranking artificial.
        #
        # Conservamos el score que viene del backend.
        # ====================================================

        for result in results:

            result["_final_score"] = float(
                result.get(
                    "score",
                    0,
                )
            )

        results.sort(
            key=lambda result: result.get(
                "_final_score",
                0,
            ),
            reverse=True,
        )

        final_results = results[
            : self.valves.max_results
        ]

        total_chars = sum(
            len(
                r.get(
                    "text",
                    "",
                )
            )
            for r in final_results
        )

        print(
            "[TennisDoc RAG] "
            f"Contexto final: "
            f"{len(final_results)} resultados "
            f"| {total_chars} chars"
        )

        return final_results

    # ========================================================
    # BUILD CONTEXT
    #
    # IMPORTANTE:
    # 12.000 caracteres por fragmento.
    # 90.000 caracteres máximo global.
    # ========================================================

    def _build_context(
        self,
        results,
    ):

        parts = []

        total_chars = 0

        for index, result in enumerate(
            results,
            1,
        ):

            text = result.get(
                "text",
                "",
            )

            if not text:

                continue

            text = text[
                : self.valves.max_chars_per_chunk
            ]

            repo = result.get(
                "repo",
                "unknown",
            )

            branch = result.get(
                "branch",
                "",
            )

            file_path = result.get(
                "file_path",
                "unknown",
            )

            language = result.get(
                "language",
                "",
            )

            ast_type = result.get(
                "ast_type",
                "",
            )

            ast_name = result.get(
                "ast_name",
                "",
            )

            ast_signature = result.get(
                "ast_signature",
                "",
            )

            score = float(
                result.get(
                    "score",
                    0,
                )
            )

            vector_score = result.get(
                "vector_score"
            )

            graph_score = result.get(
                "graph_score"
            )

            keyword_score = result.get(
                "keyword_score"
            )

            sources = ", ".join(
                result.get(
                    "_sources",
                    [],
                )
            )

            # ------------------------------------------------
            # METADATA
            # ------------------------------------------------

            block = (
                f"[FRAGMENTO {index}]\n"
                f"Repositorio: {repo}\n"
                f"Rama: {branch}\n"
                f"Archivo: {file_path}\n"
                f"Lenguaje: {language}\n"
                f"Score: {score:.4f}\n"
            )

            if vector_score is not None:

                block += (
                    f"Vector score: "
                    f"{float(vector_score):.4f}\n"
                )

            if graph_score is not None:

                block += (
                    f"Graph score: "
                    f"{float(graph_score):.4f}\n"
                )

            if keyword_score is not None:

                block += (
                    f"Keyword score: "
                    f"{float(keyword_score):.4f}\n"
                )

            if sources:

                block += (
                    f"Fuentes retrieval: "
                    f"{sources}\n"
                )

            if ast_name:

                block += (
                    f"Entidad: "
                    f"{ast_type} "
                    f"{ast_name}\n"
                )

            if ast_signature:

                block += (
                    f"Firma: "
                    f"{ast_signature}\n"
                )

            block += (
                "\n"
                "```text\n"
                f"{text}\n"
                "```\n"
            )

            # ------------------------------------------------
            # LÍMITE GLOBAL
            # ------------------------------------------------

            if (
                total_chars
                + len(block)
                > self.valves.max_context_chars
            ):

                print(
                    "[TennisDoc RAG] "
                    "Límite de contexto alcanzado "
                    f"({self.valves.max_context_chars} chars)"
                )

                break

            parts.append(
                block
            )

            total_chars += len(
                block
            )

        return "\n\n---\n\n".join(
            parts
        )

    # ========================================================
    # KEYWORDS
    # ========================================================

    def _extract_keywords(
        self,
        query,
    ):

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
        }

        keywords = []

        # PascalCase / CamelCase
        for value in re.findall(
            r"\b[A-Z][a-zA-Z0-9_]{3,}\b",
            query,
        ):

            if value not in keywords:

                keywords.append(
                    value
                )

        # Palabras técnicas normales
        for value in re.findall(
            r"\b[a-zA-ZáéíóúñÁÉÍÓÚÑ_]{4,}\b",
            query,
        ):

            normalized = value.lower()

            if normalized in stopwords:

                continue

            if any(
                normalized
                == x.lower()
                for x in keywords
            ):

                continue

            keywords.append(
                value
            )

        return keywords[:12]

    # ========================================================
    # INLET
    # ========================================================

    def inlet(
        self,
        body: dict,
        user: dict = None,
    ):

        try:

            messages = body.get(
                "messages",
                [],
            )

            if not messages:

                return body

            last_msg = messages[-1]

            if last_msg.get(
                "role"
            ) != "user":

                return body

            raw_content = (
                last_msg.get(
                    "content",
                    "",
                )
            )

            # ------------------------------------------------
            # CONTENIDO NORMAL
            # ------------------------------------------------

            if isinstance(
                raw_content,
                list,
            ):

                query_parts = []

                for item in raw_content:

                    if isinstance(
                        item,
                        dict,
                    ):

                        text = item.get(
                            "text",
                            "",
                        )

                        if text:

                            query_parts.append(
                                str(text)
                            )

                query = " ".join(
                    query_parts
                ).strip()

            else:

                query = str(
                    raw_content
                ).strip()

            if not query:

                return body

            print(
                "[TennisDoc RAG] "
                f"Query recibida: "
                f"{query[:150]}"
            )

            lower_q = query.lower()

            # =================================================
            # INDEXAR
            # =================================================

            index_commands = (
                "indexa ",
                "indexar ",
                "reindexa ",
                "reindexar ",
                "index ",
            )

            if lower_q.startswith(
                index_commands
            ):

                args = (
                    query.split(
                        " ",
                        1,
                    )[1].strip()
                    if " " in query
                    else ""
                )

                parts = args.split()

                repo = (
                    parts[0]
                    if parts
                    else ""
                )

                branch = (
                    parts[1]
                    if len(parts) > 1
                    else self.valves.default_branch
                )

                if repo:

                    return self._trigger_index(
                        body,
                        repo,
                        branch,
                    )

            # =================================================
            # MARKDOWN
            # =================================================

            if _MD_RE.search(
                query
            ):

                repo, branch = (
                    self._extract_repo_branch(
                        query
                    )
                )

                return self._generate_markdown(
                    body,
                    repo,
                    branch,
                    query,
                )

            # =================================================
            # PDF
            # =================================================

            if _PDF_RE.search(
                query
            ):

                repo, branch = (
                    self._extract_repo_branch(
                        query
                    )
                )

                return self._generate_pdf(
                    body,
                    repo,
                    branch,
                )

            # =================================================
            # GRAFO
            # =================================================

            graph_match = (
                _GRAPH_RE.search(
                    query
                )
            )

            if graph_match:

                class_name = (
                    graph_match.group(
                        1
                    )
                )

                repo, branch = (
                    self._extract_repo_branch(
                        query
                    )
                )

                return self._show_graph(
                    body,
                    class_name,
                    repo,
                    branch,
                )

            # =================================================
            # REPOS
            # =================================================

            if _REPOS_LIST_RE.search(
                query
            ):

                return self._list_repos(
                    body
                )

            # =================================================
            # RAG
            # =================================================

            repo, branch = (
                self._extract_repo_branch(
                    query
                )
            )

            results = (
                self._fetch_context(
                    repo,
                    branch,
                    query,
                )
            )

            if not results:

                print(
                    "[TennisDoc RAG] "
                    "No se recuperó contexto"
                )

                return body

            context = (
                self._build_context(
                    results
                )
            )

            analysis = (
                self._analyze_query(
                    query
                )
            )

            # =================================================
            # SYSTEM PROMPT
            # =================================================

            if repo:

                scope = (
                    f"Repositorio: `{repo}`\n"
                    f"Rama: `{branch}`"
                )

            else:

                scope = (
                    f"Repositorio no especificado.\n"
                    f"Filtro de rama: `{branch}`"
                )

            system_content = f"""
Eres Tennis Doc IA, un asistente especializado
en analizar código fuente de la organización.

ÁMBITO DE LA CONSULTA
=====================

{scope}

TIPO DE CONSULTA
================

{analysis["strategy"]}

REGLAS IMPORTANTES
==================

1. Basa la respuesta principalmente en el código
   recuperado en el contexto.

2. NO inventes clases, métodos, endpoints,
   tablas, relaciones ni comportamientos.

3. Si haces una inferencia, indícalo claramente.

4. Si la información no está disponible en el
   contexto, dilo explícitamente.

5. Busca información en TODOS los fragmentos
   recuperados antes de concluir que algo no existe.

6. Las clases Java pueden ser grandes y estar
   divididas en múltiples fragmentos. No asumas
   que un solo fragmento representa toda la clase.

7. Cuando expliques una clase identifica, cuando
   exista evidencia:

   - repositorio
   - rama
   - archivo
   - clase
   - métodos
   - dependencias
   - interfaces
   - clases padre
   - queries SQL/JPQL
   - servicios utilizados

8. Para preguntas de arquitectura explica el flujo
   utilizando las relaciones encontradas en el grafo.

9. Para preguntas sobre métodos, busca también
   las llamadas que realizan y los servicios,
   repositorios o componentes involucrados.

10. Para preguntas SQL/JPQL identifica cuando exista:

    - tabla
    - query
    - DAO
    - Repository
    - procedimiento almacenado
    - entidad relacionada

11. Si una clase implementa o extiende otra,
    explica la jerarquía encontrada.

12. No asumas que una clase es utilizada en producción
    solamente porque existe en el repositorio.

13. Si existen varias implementaciones posibles,
    menciona las diferencias encontradas.

14. Cuando sea relevante, incluye los archivos
    involucrados.

15. Puedes utilizar fragmentos de código del contexto
    para explicar el comportamiento.

16. Si el contexto contiene una referencia SQL
    asociada a código Java, relaciónala con el método
    correspondiente cuando exista evidencia.

17. Si no hay suficiente información para responder
    con seguridad, dilo en lugar de inventar.

FUENTES
=======

Al final agrega una sección:

### Fuentes

Lista los archivos más relevantes utilizados
para construir la respuesta.

No inventes archivos.

CONTEXTO RECUPERADO
===================

{context}
"""

            system_msg = {
                "role": "system",
                "content": system_content,
            }

            # Insertar antes del último mensaje del usuario.
            messages.insert(
                len(messages) - 1,
                system_msg,
            )

            body["messages"] = (
                messages
            )

            print(
                "[TennisDoc RAG] "
                f"Contexto inyectado: "
                f"{len(results)} resultados "
                f"| {len(context)} caracteres"
            )

        except Exception as exc:

            print(
                "[TennisDoc RAG] "
                f"ERROR inlet: {exc}"
            )

        return body

    # ========================================================
    # INDEX
    # ========================================================

    def _trigger_index(
        self,
        body,
        repo,
        branch,
    ):

        try:

            response = self._post(
                "/index",
                json={
                    "repo": repo,
                    "branch": branch,
                },
                timeout=5,
            )

            response.raise_for_status()

            data = response.json()

            body["messages"][-1][
                "content"
            ] = (
                "🗂️ **Indexación iniciada**\n\n"
                f"**Repositorio:** `{repo}`\n"
                f"**Rama:** `{branch}`\n"
                f"**Estado:** "
                f"`{data.get('status', 'ok')}`\n\n"
                "Cuando finalice puedes hacer "
                "preguntas sobre el repositorio."
            )

        except Exception as exc:

            body["messages"][-1][
                "content"
            ] = (
                f"❌ Error iniciando indexación "
                f"de `{repo}` @ `{branch}`:\n\n"
                f"`{exc}`"
            )

        return body

    # ========================================================
    # MARKDOWN
    # ========================================================

    def _generate_markdown(
        self,
        body,
        repo,
        branch,
        query,
    ):

        try:

            results = (
                self._fetch_context(
                    repo,
                    branch,
                    query,
                )
            )

            if not results:

                body["messages"][-1][
                    "content"
                ] = (
                    "No encontré suficiente "
                    "contexto para generar "
                    "el Markdown."
                )

                return body

            context = (
                self._build_context(
                    results
                )
            )

            system_msg = {
                "role": "system",
                "content": (
                    "Eres Tennis Doc IA.\n\n"
                    "Genera la respuesta "
                    "exclusivamente utilizando "
                    "el contexto del código "
                    "proporcionado.\n\n"
                    "No inventes información.\n"
                    "Incluye archivos y clases "
                    "relevantes.\n"
                    "Utiliza tablas cuando aporten "
                    "claridad.\n"
                    "Utiliza bloques de código "
                    "para snippets.\n\n"
                    f"CONTEXTO:\n{context}"
                ),
            }

            body["messages"].insert(
                len(body["messages"]) - 1,
                system_msg,
            )

        except Exception as exc:

            print(
                "[TennisDoc RAG] "
                f"ERROR Markdown: {exc}"
            )

        return body

    # ========================================================
    # PDF
    # ========================================================

    def _generate_pdf(
        self,
        body,
        repo,
        branch,
    ):

        try:

            messages = body.get(
                "messages",
                [],
            )

            assistant_msg = None

            for message in reversed(
                messages[:-1]
            ):

                if (
                    message.get(
                        "role"
                    )
                    == "assistant"
                ):

                    assistant_msg = message

                    break

            if not assistant_msg:

                body["messages"][-1][
                    "content"
                ] = (
                    "No hay una respuesta "
                    "anterior para convertir "
                    "a PDF."
                )

                return body

            content = (
                assistant_msg.get(
                    "content",
                    "",
                )
            )

            title = (
                f"Respuesta_"
                f"{repo.replace('/', '_')}"
                if repo
                else "Respuesta"
            )

            response = self._post(
                "/pdf",
                json={
                    "title": title,
                    "content": content,
                    "repo": repo,
                    "branch": branch,
                },
                timeout=30,
            )

            response.raise_for_status()

            pdf_b64 = (
                base64.b64encode(
                    response.content
                ).decode(
                    "utf-8"
                )
            )

            download_link = (
                f'<a href="data:application/pdf;base64,'
                f'{pdf_b64}" '
                f'download="{title}.pdf">'
                f'📄 Descargar PDF'
                f'</a>'
            )

            body["messages"][-1][
                "content"
            ] = (
                "PDF generado correctamente.\n\n"
                f"{download_link}"
            )

        except Exception as exc:

            body["messages"][-1][
                "content"
            ] = (
                f"Error generando PDF: "
                f"`{exc}`"
            )

        return body

    # ========================================================
    # GRAFO
    # ========================================================

    def _show_graph(
        self,
        body,
        class_name,
        repo,
        branch,
    ):

        try:

            response = self._get(
                f"/graph/entity/{class_name}",
                params={
                    k: v
                    for k, v in {
                        "repo": repo,
                        "branch": branch,
                    }.items()
                    if v
                },
                timeout=15,
            )

            response.raise_for_status()

            data = response.json()

            lines = [
                f"## Grafo: `{class_name}`",
                "",
                f"**Tipo:** "
                f"`{data.get('type', 'Unknown')}`",
                f"**Archivo:** "
                f"`{data.get('file_path', 'N/A')}`",
            ]

            if data.get(
                "signature"
            ):

                lines.append(
                    f"**Firma:** "
                    f"`{data['signature']}`"
                )

            lines.append("")

            relations = data.get(
                "relations",
                [],
            )

            if relations:

                lines.append(
                    "### Relaciones"
                )

                for relation in relations:

                    lines.append(
                        f"- **"
                        f"{relation.get('rel_type', 'REL')}"
                        f"** → "
                        f"`{relation.get('target_name', '?')}` "
                        f"("
                        f"{relation.get('target_type', '?')}"
                        f")"
                    )

            else:

                lines.append(
                    "No se encontraron "
                    "relaciones directas."
                )

            body["messages"][-1][
                "content"
            ] = "\n".join(
                lines
            )

        except Exception as exc:

            body["messages"][-1][
                "content"
            ] = (
                f"No pude obtener el "
                f"grafo de `{class_name}`.\n\n"
                f"Error: `{exc}`"
            )

        return body

    # ========================================================
    # REPOS
    # ========================================================

    def _list_repos(
        self,
        body,
    ):

        try:

            response = self._get(
                "/repos",
                timeout=10,
            )

            response.raise_for_status()

            data = response.json()

            repos = data.get(
                "repos",
                [],
            )

            if not repos:

                body["messages"][-1][
                    "content"
                ] = (
                    "No hay repositorios "
                    "indexados.\n\n"
                    "Puedes iniciar uno con:\n\n"
                    "`indexa org/repositorio`"
                )

                return body

            lines = [
                "## Repositorios indexados",
                "",
            ]

            for repo_name in sorted(
                repos
            ):

                lines.append(
                    f"- `{repo_name}`"
                )

            lines.extend(
                [
                    "",
                    "Para consultar uno:",
                    "",
                    "`Explícame el flujo de org/repositorio`",
                ]
            )

            body["messages"][-1][
                "content"
            ] = "\n".join(
                lines
            )

        except Exception as exc:

            body["messages"][-1][
                "content"
            ] = (
                "Error consultando "
                "repositorios:\n\n"
                f"`{exc}`"
            )

        return body
