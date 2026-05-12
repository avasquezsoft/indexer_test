"""
rag_engine.py — Motor de retrieval híbrido: vectorial (Qdrant) + grafo (Neo4j).

Usa LlamaIndex como framework de orquestación con un retriever custom.
"""

import logging
from typing import Any

from llama_index.core import QueryBundle
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode

import embedder
import graph_store
from qdrant_store import get_client, search_chunks, QDRANT_COLLECTION

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# Retriever híbrido custom
# ═══════════════════════════════════════════════════════════════

class CodeGraphRetriever(BaseRetriever):
    """
    Retriever que combina:
      1. Búsqueda vectorial en Qdrant (similitud semántica)
      2. Expansión por grafo en Neo4j (vecinos de entidades encontradas)
      3. Búsqueda por nombre exacto (si la query menciona un identificador)
    """

    def __init__(
        self,
        repo: str | None = None,
        branch: str | None = None,
        vector_limit: int = 30,
        graph_depth: int = 2,
        name_search_limit: int = 5,
        **kwargs: Any,
    ):
        self.repo = repo
        self.branch = branch
        self.vector_limit = vector_limit
        self.graph_depth = graph_depth
        self.name_search_limit = name_search_limit
        super().__init__(**kwargs)

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query = query_bundle.query_str
        nodes: dict[str, NodeWithScore] = {}

        # ── 1. Búsqueda vectorial ──
        try:
            client = get_client()
            vector_results = search_chunks(
                client,
                QDRANT_COLLECTION,
                await embedder.get_embedding(query),
                self.repo,
                self.branch,
                self.vector_limit,
            )
            for r in vector_results:
                entity_id = r.get("entity_id")
                key = entity_id or f"{r['file_path']}:{r.get('position', 0)}"
                node = TextNode(
                    text=r.get("text", ""),
                    metadata={
                        "repo": r.get("repo"),
                        "branch": r.get("branch"),
                        "file_path": r.get("file_path"),
                        "language": r.get("language"),
                        "entity_id": entity_id,
                        "ast_type": r.get("ast_type"),
                        "ast_name": r.get("ast_name"),
                        "ast_signature": r.get("ast_signature"),
                        "source": "vector",
                    },
                )
                nodes[key] = NodeWithScore(node=node, score=r.get("score", 0.0))
        except Exception as exc:
            logger.warning("Error en búsqueda vectorial: %s", exc)

        # ── 2. Expansión por grafo ──
        entity_ids = [n.metadata["entity_id"] for n in nodes.values() if n.metadata.get("entity_id")]
        for eid in set(entity_ids):
            try:
                related = graph_store.get_related_entities(eid, depth=self.graph_depth)
                for rel in related:
                    key = rel["id"]
                    if key in nodes:
                        # Boost score si ya existe
                        nodes[key].score = max(nodes[key].score, 0.5 / rel.get("distance", 1))
                        continue
                    node = TextNode(
                        text=rel.get("code", ""),
                        metadata={
                            "repo": self.repo,
                            "branch": self.branch,
                            "file_path": rel.get("file_path", ""),
                            "entity_id": rel["id"],
                            "ast_name": rel.get("name", ""),
                            "ast_type": rel.get("type", ""),
                            "ast_signature": rel.get("signature", ""),
                            "source": "graph",
                            "distance": rel.get("distance", 1),
                        },
                    )
                    # Score decrece con la distancia en el grafo
                    distance = rel.get("distance", 1)
                    score = 0.6 / distance
                    nodes[key] = NodeWithScore(node=node, score=score)
            except Exception as exc:
                logger.warning("Error en expansión de grafo para %s: %s", eid, exc)

        # ── 3. Búsqueda por nombre exacto (keyword) ──
        # Extraer posibles identificadores CamelCase de la query
        import re
        # Palabras comunes de patrones enterprise Java/Spring
        _SUFFIX_PATTERN = r"(?:Impl|Dao|Service|Repository|Mapper|Controller|Dto|Entity|Config|Util|Factory|Handler|Listener|Task|Job|Processor|Writer|Reader|Interceptor|Filter|Endpoint|Client|Provider|Adapter|Facade|Builder|Validator|Converter|Parser|Renderer|Generator|Scheduler|Resolver|Registry|Cache|Pool|Queue|Map|Tree|Node|Connection|Transaction|Context|Event|Message|Command|Query|Request|Response|Result|Source|Target|Reference|Wrapper|Proxy|Mock|Spy|Checker|Tester|Inspector|Finder|Searcher|Indexer|Extractor|Loader|Saver|Retriever|Updater|Creator|Initializer|Activator|Dispatcher|Router|Balancer|Distributor|Assigner|Configurer|Setter|Getter|Accessor|Builder|Producer|Consumer|Subscriber|Publisher|Emitter|Receiver|Sender|Transmitter|Host|Client|Server|Helper|Utility|Tool|Api|Sdk|Cli|Ui|Web|Rest|Soap|Grpc|Graphql|Websocket|Socket|Port|Channel|Pipe|Stream|Flow|Pipeline|Chain|Sequence|Batch|Bundle|Package|Module|Component|Part|Section|Segment|Fragment|Chunk|Block|Unit|Item|Element|Member|Field|Property|Attribute|Parameter|Argument|Option|Setting|Configuration|Policy|Rule|Strategy|Pattern|Template|Schema|Model|Blueprint|Plan|Design|Layout|Structure|Framework|Platform|System|Engine|Kernel|Core|Base|Root|Foundation|Layer|Tier|Level|Stage|Phase|Step|Action|Operation|Process|Procedure|Routine|Function|Method|Subroutine|Macro|Script|Program|Application|App)"
        candidates = re.findall(rf"\b([A-Z][a-zA-Z0-9]*(?:{_SUFFIX_PATTERN})?)\b", query)
        candidates = [c for c in candidates if len(c) > 2]
        for cand in set(candidates[:self.name_search_limit]):
            try:
                keyword_results = graph_store.search_by_name(cand, repo=self.repo, branch=self.branch)
                for kr in keyword_results[:3]:
                    key = kr["id"]
                    if key in nodes:
                        nodes[key].score = max(nodes[key].score, 0.9)
                        continue
                    node = TextNode(
                        text=kr.get("code", ""),
                        metadata={
                            "repo": self.repo,
                            "branch": self.branch,
                            "file_path": kr.get("file_path", ""),
                            "entity_id": kr["id"],
                            "ast_name": kr.get("name", ""),
                            "ast_type": kr.get("type", ""),
                            "ast_signature": kr.get("signature", ""),
                            "source": "keyword",
                        },
                    )
                    nodes[key] = NodeWithScore(node=node, score=0.9)
            except Exception as exc:
                logger.warning("Error en búsqueda por nombre %s: %s", cand, exc)

        # Ordenar por score descendente y devolver
        sorted_nodes = sorted(nodes.values(), key=lambda n: n.score, reverse=True)
        return sorted_nodes

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        # LlamaIndex requiere ambos métodos; el síncrono delega al async
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self._aretrieve(query_bundle))


# ═══════════════════════════════════════════════════════════════
# API pública del RAG engine
# ═══════════════════════════════════════════════════════════════

async def search_graph(
    query: str,
    repo: str | None = None,
    branch: str | None = None,
    limit: int = 12,
    graph_depth: int = 2,
) -> list[dict]:
    """
    Búsqueda híbrida vector + grafo. Devuelve lista de dicts con el mismo
    formato que el endpoint /search actual para compatibilidad.
    """
    retriever = CodeGraphRetriever(
        repo=repo,
        branch=branch,
        vector_limit=limit * 3,
        graph_depth=graph_depth,
    )
    bundle = QueryBundle(query_str=query)
    results = await retriever.aretrieve(bundle)

    output = []
    for r in results[:limit]:
        meta = r.node.metadata
        output.append({
            "score": float(r.score),
            "repo": meta.get("repo", repo or ""),
            "branch": meta.get("branch", branch or ""),
            "file_path": meta.get("file_path", ""),
            "language": meta.get("language", ""),
            "text": r.node.text,
            "entity_id": meta.get("entity_id"),
            "ast_type": meta.get("ast_type"),
            "ast_name": meta.get("ast_name"),
            "ast_signature": meta.get("ast_signature"),
            "source": meta.get("source", "unknown"),
        })
    return output


async def search_entity_in_graph(name: str, repo: str | None = None, branch: str | None = None) -> dict | None:
    """Busca una entidad por nombre exacto y devuelve su contexto de grafo."""
    results = graph_store.search_by_name(name, repo=repo, branch=branch)
    if not results:
        return None
    best = results[0]
    entity_id = best["id"]
    full = graph_store.get_entity_with_direct_relations(entity_id)
    return full or best
