import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from config import QDRANT_URL, QDRANT_API_KEY, VECTOR_SIZE, QDRANT_COLLECTION


def get_client() -> QdrantClient:
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )


def ping_client(client: QdrantClient) -> bool:
    """Verifica que Qdrant responde."""
    try:
        client.get_collections()
        return True
    except Exception:
        return False


def ensure_collection(client: QdrantClient, collection: str):
    """Crea la colección si no existe. Si existe pero con dimensiones distintas, la recrea."""
    import logging
    log = logging.getLogger(__name__)

    existing_names = [c.name for c in client.get_collections().collections]

    if collection not in existing_names:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        log.info(f"Colección '{collection}' creada con dimensión {VECTOR_SIZE}")
        return

    # Validar que la dimensión coincida con el modelo actual
    info = client.get_collection(collection)
    current_size = None
    vectors_config = info.config.params.vectors

    if hasattr(vectors_config, "size"):
        current_size = vectors_config.size
    elif isinstance(vectors_config, dict) and "size" in vectors_config:
        current_size = vectors_config["size"]

    if current_size is not None and current_size != VECTOR_SIZE:
        log.warning(
            f"Colección '{collection}' existe con dimensión {current_size}, "
            f"pero el modelo actual requiere {VECTOR_SIZE}. Se eliminará y recreará."
        )
        client.delete_collection(collection_name=collection)
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        log.info(f"Colección '{collection}' recreada con dimensión {VECTOR_SIZE}")



def delete_repo_chunks(client: QdrantClient, collection: str, repo: str, branch: str | None = None):
    """Elimina todos los chunks de un repo (opcionalmente filtrado por rama)."""
    conditions = [FieldCondition(key="repo", match=MatchValue(value=repo))]
    if branch:
        conditions.append(FieldCondition(key="branch", match=MatchValue(value=branch)))
    client.delete(
        collection_name=collection,
        points_selector=Filter(must=conditions),
    )


def upsert_chunks(client: QdrantClient, collection: str, chunks: list[dict], embeddings: list[list[float]]):
    """Guarda los chunks con sus embeddings en Qdrant."""
    points = []
    for chunk, embedding in zip(chunks, embeddings):
        # Usar entity_id como ID si está disponible (idempotencia), sino UUID
        chunk_id = chunk["metadata"].get("entity_id") or str(uuid.uuid4())
        points.append(
            PointStruct(
                id=chunk_id,
                vector=embedding,
                payload={
                    "repo": chunk["metadata"]["repo"],
                    "branch": chunk["metadata"]["branch"],
                    "file_path": chunk["metadata"]["file_path"],
                    "language": chunk["metadata"]["language"],
                    "position": chunk["metadata"]["position"],
                    "text": chunk["text"],
                    "entity_id": chunk["metadata"].get("entity_id"),
                    "ast_type": chunk["metadata"].get("ast_type"),
                    "ast_name": chunk["metadata"].get("ast_name"),
                    "ast_signature": chunk["metadata"].get("ast_signature"),
                },
            )
        )

    # Insertar en lotes de 100
    batch_size = 100
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=collection,
            points=points[i : i + batch_size],
        )


def search_chunks(client: QdrantClient, collection: str, query_vector: list[float], repo_filter: str | None = None, branch_filter: str | None = None, limit: int = 6) -> list[dict]:
    """Busca los chunks más relevantes para un vector de consulta."""
    import logging
    log = logging.getLogger(__name__)

    conditions = []
    if repo_filter:
        conditions.append(FieldCondition(key="repo", match=MatchValue(value=repo_filter)))
    if branch_filter:
        conditions.append(FieldCondition(key="branch", match=MatchValue(value=branch_filter)))

    query_filter = Filter(must=conditions) if conditions else None

    try:
        response = client.query_points(
            collection_name=collection,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        results = response.points
    except Exception as exc:
        log.error(f"Error en Qdrant query_points: {exc}")
        raise

    output = []
    for r in results:
        try:
            output.append({
                "score": r.score,
                "repo": r.payload.get("repo", "unknown"),
                "branch": r.payload.get("branch", ""),
                "file_path": r.payload.get("file_path", "unknown"),
                "language": r.payload.get("language", "unknown"),
                "text": r.payload.get("text", ""),
                "entity_id": r.payload.get("entity_id"),
                "ast_type": r.payload.get("ast_type"),
                "ast_name": r.payload.get("ast_name"),
                "ast_signature": r.payload.get("ast_signature"),
            })
        except Exception as exc:
            log.warning(f"Chunk con payload incompleto descartado: {exc}")
            continue

    return output
