import os
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

# Dimensión del modelo nomic-embed-text
VECTOR_SIZE = 768


def get_client() -> QdrantClient:
    return QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ.get("QDRANT_API_KEY"),
    )


def ensure_collection(client: QdrantClient, collection: str):
    """Crea la colección si no existe."""
    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def delete_repo_chunks(client: QdrantClient, collection: str, repo: str):
    """Elimina todos los chunks de un repo antes de re-indexar."""
    client.delete(
        collection_name=collection,
        points_selector=Filter(
            must=[FieldCondition(key="repo", match=MatchValue(value=repo))]
        ),
    )


def upsert_chunks(client: QdrantClient, collection: str, chunks: list[dict], embeddings: list[list[float]]):
    """Guarda los chunks con sus embeddings en Qdrant."""
    points = []
    for chunk, embedding in zip(chunks, embeddings):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "repo": chunk["metadata"]["repo"],
                    "file_path": chunk["metadata"]["file_path"],
                    "language": chunk["metadata"]["language"],
                    "position": chunk["metadata"]["position"],
                    "text": chunk["text"],
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


def search_chunks(client: QdrantClient, collection: str, query_vector: list[float], repo_filter: str | None = None, limit: int = 6) -> list[dict]:
    """Busca los chunks más relevantes para un vector de consulta."""
    query_filter = None
    if repo_filter:
        query_filter = Filter(
            must=[FieldCondition(key="repo", match=MatchValue(value=repo_filter))]
        )

    results = client.search(
        collection_name=collection,
        query_vector=query_vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    )

    return [
        {
            "score": r.score,
            "repo": r.payload["repo"],
            "file_path": r.payload["file_path"],
            "language": r.payload["language"],
            "text": r.payload["text"],
        }
        for r in results
    ]
