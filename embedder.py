import os
import httpx


async def get_embedding(text: str) -> list[float]:
    """Genera embedding de un texto usando Ollama API."""
    api_base = os.environ["OLLAMA_API_BASE"].rstrip("/")
    api_key = os.environ["OLLAMA_API_KEY"]
    model = os.environ["OLLAMA_EMBED_MODEL"]

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{api_base}/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"model": model, "input": text},
        )
        response.raise_for_status()
        data = response.json()

        # La API de Ollama compatible con OpenAI devuelve data[0].embedding
        return data["data"][0]["embedding"]


async def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Genera embeddings para una lista de textos."""
    embeddings = []
    for text in texts:
        embedding = await get_embedding(text)
        embeddings.append(embedding)
    return embeddings
