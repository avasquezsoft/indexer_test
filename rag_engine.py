"""
rag_engine.py — Motor de retrieval híbrido para código.

Combina:
1. Qdrant       → búsqueda semántica
2. Neo4j        → relaciones y dependencias
3. Exact match  → búsqueda de entidades/clases
4. Reranking    → combinación de señales
5. Deduplicación → evita enviar el mismo código varias veces
"""

import logging
import re
from dataclasses import dataclass, field

import embedder
import graph_store

from qdrant_store import (
    get_client,
    search_chunks,
    QDRANT_COLLECTION,
)

logger = logging.getLogger(__name__)


# ============================================================
# MODELOS
# ============================================================

@dataclass
class TextNode:
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class NodeWithScore:
    node: TextNode

    # Score original de la fuente
    score: float = 0.0

    # Scores normalizados
    vector_score: float = 0.0
    graph_score: float = 0.0
    keyword_score: float = 0.0

    # Score final
    final_score: float = 0.0


# ============================================================
# CONSTANTES DE RANKING
# ============================================================

VECTOR_WEIGHT = 0.50
GRAPH_WEIGHT = 0.20
KEYWORD_WEIGHT = 0.20
ENTITY_WEIGHT = 0.10


# ============================================================
# PATRONES
# ============================================================

JAVA_SUFFIXES = (
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
    "Connection",
    "Transaction",
    "Context",
    "Event",
    "Message",
    "Command",
    "Query",
    "Request",
    "Response",
    "Result",
    "Wrapper",
    "Proxy",
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


# ============================================================
# HELPERS
# ============================================================

def _normalize_score(score: float) -> float:
    """
    Normaliza scores externos a un rango aproximado 0..1.
    """
    try:
        value = float(score)
    except (TypeError, ValueError):
        return 0.0

    return max(0.0, min(1.0, value))


def _extract_identifiers(query: str) -> list[str]:
    """
    Extrae posibles nombres de clases, interfaces y entidades Java.
    """

    candidates = re.findall(
        r"\b[A-Z][a-zA-Z0-9_]{2,}\b",
        query,
    )

    result = []

    for candidate in candidates:

        if candidate in {
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
        }:
            continue

        # Priorizar nombres que parecen Java
        if (
            candidate.endswith(JAVA_SUFFIXES)
            or "_" in candidate
        ):
            result.append(candidate)

    return list(dict.fromkeys(result))


def _extract_methods(query: str) -> list[str]:
    """
    Detecta nombres con formato método(...).
    """

    matches = re.findall(
        r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",
        query,
    )

    return list(dict.fromkeys(matches))


# ============================================================
# RETRIEVER
# ============================================================

class CodeGraphRetriever:

    def __init__(
        self,
        repo: str | None = None,
        branch: str | None = None,
        vector_limit: int = 50,
        graph_depth: int = 2,
        name_search_limit: int = 5,
    ):
        self.repo = repo
        self.branch = branch
        self.vector_limit = vector_limit
        self.graph_depth = graph_depth
        self.name_search_limit = name_search_limit

    # ========================================================
    # MAIN RETRIEVAL
    # ========================================================

    async def retrieve(
        self,
        query: str,
    ) -> list[NodeWithScore]:

        nodes: dict[str, NodeWithScore] = {}

        identifiers = _extract_identifiers(query)
        methods = _extract_methods(query)

        logger.info(
            "RAG query repo=%s branch=%s identifiers=%s methods=%s",
            self.repo,
            self.branch,
            identifiers,
            methods,
        )

        # ====================================================
        # 1. VECTOR SEARCH
        # ====================================================

        await self._vector_search(
            query,
            nodes,
        )

        # ====================================================
        # 2. EXACT NAME SEARCH
        # ====================================================

        await self._keyword_search(
            identifiers,
            nodes,
        )

        # ====================================================
        # 3. GRAPH EXPANSION
        # ====================================================

        await self._graph_expansion(
            nodes,
        )

        # ====================================================
        # 4. ENTITY / METHOD BOOST
        # ====================================================

        self._apply_entity_boost(
            nodes,
            identifiers,
            methods,
        )

        # ====================================================
        # 5. FINAL RANKING
        # ====================================================

        results = self._rerank(
            nodes,
        )

        logger.info(
            "RAG final results=%d",
            len(results),
        )

        return results

    # ========================================================
    # VECTOR
    # ========================================================

    async def _vector_search(
        self,
        query: str,
        nodes: dict[str, NodeWithScore],
    ):

        try:

            client = get_client()

            embedding = await embedder.get_embedding(
                query
            )

            vector_results = search_chunks(
                client,
                QDRANT_COLLECTION,
                embedding,
                self.repo,
                self.branch,
                self.vector_limit,
            )

            for result in vector_results:

                entity_id = result.get(
                    "entity_id"
                )

                file_path = result.get(
                    "file_path",
                    "",
                )

                position = result.get(
                    "position",
                    0,
                )

                # Entity ID es la mejor clave.
                # Si no existe, usamos archivo + posición.
                key = (
                    str(entity_id)
                    if entity_id
                    else f"{file_path}:{position}"
                )

                vector_score = _normalize_score(
                    result.get(
                        "score",
                        0.0,
                    )
                )

                node = TextNode(
                    text=result.get(
                        "text",
                        "",
                    ),
                    metadata={
                        "repo": result.get(
                            "repo"
                        ),
                        "branch": result.get(
                            "branch"
                        ),
                        "file_path": file_path,
                        "language": result.get(
                            "language"
                        ),
                        "entity_id": entity_id,
                        "ast_type": result.get(
                            "ast_type"
                        ),
                        "ast_name": result.get(
                            "ast_name"
                        ),
                        "ast_signature": result.get(
                            "ast_signature"
                        ),
                        "source": "vector",
                    },
                )

                nodes[key] = NodeWithScore(
                    node=node,
                    score=vector_score,
                    vector_score=vector_score,
                )

        except Exception as exc:

            logger.warning(
                "Error en búsqueda vectorial: %s",
                exc,
            )

    # ========================================================
    # KEYWORD / EXACT NAME
    # ========================================================

    async def _keyword_search(
        self,
        identifiers: list[str],
        nodes: dict[str, NodeWithScore],
    ):

        for identifier in identifiers[
            : self.name_search_limit
        ]:

            try:

                keyword_results = (
                    graph_store.search_by_name(
                        identifier,
                        repo=self.repo,
                        branch=self.branch,
                    )
                )

                for result in keyword_results[:3]:

                    key = str(
                        result["id"]
                    )

                    keyword_score = 1.0

                    if key in nodes:

                        nodes[key].keyword_score = max(
                            nodes[key].keyword_score,
                            keyword_score,
                        )

                        continue

                    node = TextNode(
                        text=result.get(
                            "code",
                            "",
                        ),
                        metadata={
                            "repo": self.repo,
                            "branch": self.branch,
                            "file_path": result.get(
                                "file_path",
                                "",
                            ),
                            "entity_id": result["id"],
                            "ast_name": result.get(
                                "name",
                                "",
                            ),
                            "ast_type": result.get(
                                "type",
                                "",
                            ),
                            "ast_signature": result.get(
                                "signature",
                                "",
                            ),
                            "source": "keyword",
                            "matched_identifier": identifier,
                        },
                    )

                    nodes[key] = NodeWithScore(
                        node=node,
                        score=keyword_score,
                        keyword_score=keyword_score,
                    )

            except Exception as exc:

                logger.warning(
                    "Error en búsqueda exacta %s: %s",
                    identifier,
                    exc,
                )

    # ========================================================
    # GRAPH EXPANSION
    # ========================================================

    async def _graph_expansion(
        self,
        nodes: dict[str, NodeWithScore],
    ):

        entity_ids = {
            node.node.metadata.get(
                "entity_id"
            )
            for node in nodes.values()
            if node.node.metadata.get(
                "entity_id"
            )
        }

        for entity_id in entity_ids:

            try:

                related = (
                    graph_store.get_related_entities(
                        entity_id,
                        depth=self.graph_depth,
                    )
                )

                for relation in related:

                    relation_id = str(
                        relation["id"]
                    )

                    distance = max(
                        int(
                            relation.get(
                                "distance",
                                1,
                            )
                        ),
                        1,
                    )

                    # Cuanto más cerca,
                    # mayor relevancia.
                    graph_score = 1.0 / distance

                    if relation_id in nodes:

                        nodes[
                            relation_id
                        ].graph_score = max(
                            nodes[
                                relation_id
                            ].graph_score,
                            graph_score,
                        )

                        continue

                    node = TextNode(
                        text=relation.get(
                            "code",
                            "",
                        ),
                        metadata={
                            "repo": self.repo,
                            "branch": self.branch,
                            "file_path": relation.get(
                                "file_path",
                                "",
                            ),
                            "entity_id": relation_id,
                            "ast_name": relation.get(
                                "name",
                                "",
                            ),
                            "ast_type": relation.get(
                                "type",
                                "",
                            ),
                            "ast_signature": relation.get(
                                "signature",
                                "",
                            ),
                            "source": "graph",
                            "distance": distance,
                        },
                    )

                    nodes[relation_id] = NodeWithScore(
                        node=node,
                        score=graph_score,
                        graph_score=graph_score,
                    )

            except Exception as exc:

                logger.warning(
                    "Error en expansión de grafo %s: %s",
                    entity_id,
                    exc,
                )

    # ========================================================
    # ENTITY / METHOD BOOST
    # ========================================================

    def _apply_entity_boost(
        self,
        nodes: dict[str, NodeWithScore],
        identifiers: list[str],
        methods: list[str],
    ):

        identifiers_lower = [
            x.lower()
            for x in identifiers
        ]

        methods_lower = [
            x.lower()
            for x in methods
        ]

        for result in nodes.values():

            metadata = result.node.metadata

            ast_name = str(
                metadata.get(
                    "ast_name",
                    "",
                )
            ).lower()

            file_path = str(
                metadata.get(
                    "file_path",
                    "",
                )
            ).lower()

            text = result.node.text.lower()

            # ----------------------------------------------
            # Exact class/entity
            # ----------------------------------------------

            for identifier in identifiers_lower:

                if ast_name == identifier:

                    result.keyword_score = max(
                        result.keyword_score,
                        1.0,
                    )

                elif identifier in file_path:

                    result.keyword_score = max(
                        result.keyword_score,
                        0.90,
                    )

                elif identifier in text:

                    result.keyword_score = max(
                        result.keyword_score,
                        0.70,
                    )

            # ----------------------------------------------
            # Method match
            # ----------------------------------------------

            for method in methods_lower:

                if re.search(
                    rf"\b{re.escape(method)}\s*\(",
                    text,
                ):

                    result.keyword_score = max(
                        result.keyword_score,
                        0.95,
                    )

    # ========================================================
    # RERANK
    # ========================================================

    def _rerank(
        self,
        nodes: dict[str, NodeWithScore],
    ) -> list[NodeWithScore]:

        results = list(
            nodes.values()
        )

        for result in results:

            vector = _normalize_score(
                result.vector_score
            )

            graph = _normalize_score(
                result.graph_score
            )

            keyword = _normalize_score(
                result.keyword_score
            )

            # ----------------------------------------------
            # Fusion
            # ----------------------------------------------

            final_score = (
                vector * VECTOR_WEIGHT
                + graph * GRAPH_WEIGHT
                + keyword * KEYWORD_WEIGHT
            )

            # ----------------------------------------------
            # Entity bonus
            # ----------------------------------------------

            metadata = result.node.metadata

            if metadata.get(
                "ast_name"
            ):

                final_score += (
                    ENTITY_WEIGHT
                    * min(
                        keyword,
                        1.0,
                    )
                )

            # ----------------------------------------------
            # Multi-source bonus
            # ----------------------------------------------

            source = metadata.get(
                "source",
                "",
            )

            if (
                vector > 0
                and graph > 0
            ):
                final_score += 0.08

            if (
                keyword > 0
                and vector > 0
            ):
                final_score += 0.08

            # ----------------------------------------------
            # Cap
            # ----------------------------------------------

            result.final_score = min(
                final_score,
                1.20,
            )

            result.score = result.final_score

        results.sort(
            key=lambda result: result.final_score,
            reverse=True,
        )

        return results


# ============================================================
# PUBLIC API
# ============================================================

async def search_graph(
    query: str,
    repo: str | None = None,
    branch: str | None = None,
    limit: int = 12,
    graph_depth: int = 2,
) -> list[dict]:
    """
    Búsqueda híbrida:

    Qdrant + Neo4j + exact match + reranking.

    Mantiene el formato de respuesta compatible
    con el endpoint /search-graph.
    """

    retriever = CodeGraphRetriever(
        repo=repo,
        branch=branch,
        vector_limit=max(
            limit * 4,
            30,
        ),
        graph_depth=graph_depth,
        name_search_limit=5,
    )

    results = await retriever.retrieve(
        query
    )

    output = []

    for result in results[:limit]:

        metadata = result.node.metadata

        output.append(
            {
                "score": float(
                    result.final_score
                ),

                "vector_score": float(
                    result.vector_score
                ),

                "graph_score": float(
                    result.graph_score
                ),

                "keyword_score": float(
                    result.keyword_score
                ),

                "repo": metadata.get(
                    "repo",
                    repo or "",
                ),

                "branch": metadata.get(
                    "branch",
                    branch or "",
                ),

                "file_path": metadata.get(
                    "file_path",
                    "",
                ),

                "language": metadata.get(
                    "language",
                    "",
                ),

                "text": result.node.text,

                "entity_id": metadata.get(
                    "entity_id"
                ),

                "ast_type": metadata.get(
                    "ast_type"
                ),

                "ast_name": metadata.get(
                    "ast_name"
                ),

                "ast_signature": metadata.get(
                    "ast_signature"
                ),

                "source": metadata.get(
                    "source",
                    "unknown",
                ),
            }
        )

    return output


# ============================================================
# ENTITY SEARCH
# ============================================================

async def search_entity_in_graph(
    name: str,
    repo: str | None = None,
    branch: str | None = None,
) -> dict | None:
    """
    Busca una entidad por nombre exacto
    y devuelve contexto + relaciones directas.
    """

    try:

        results = graph_store.search_by_name(
            name,
            repo=repo,
            branch=branch,
        )

        if not results:
            return None

        best = results[0]

        entity_id = best["id"]

        full = (
            graph_store.get_entity_with_direct_relations(
                entity_id
            )
        )

        return full or best

    except Exception as exc:

        logger.warning(
            "Error buscando entidad %s: %s",
            name,
            exc,
        )

        return None
