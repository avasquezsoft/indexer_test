import os
import logging
import httpx

log = logging.getLogger(__name__)


async def get_embedding(text: str) -> list[float]:
    """Genera el embedding para un único texto usando OpenRouter."""
    embeddings = await get_embeddings_batch([text])
    return embeddings[0]


async def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Genera embeddings para una lista de textos vía OpenRouter."""
    api_base = os.environ["OPENROUTER_API_BASE"].rstrip("/")
    api_key = os.environ["OPENROUTER_API_KEY"]
    model = os.environ["OPENROUTER_EMBED_MODEL"]

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{api_base}/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://tennis-doc.tritechprime.com",
                "X-Title": "tennis-doc-app",
            },
            json={
                "model": model,
                "input": texts,
            },
        )
        response.raise_for_status()
        data = response.json()

        if "data" not in data:
            log.error(f"Respuesta inesperada de OpenRouter: {data}")
            raise RuntimeError("La respuesta de embeddings no contiene 'data'")

        return [item["embedding"] for item in data["data"]]
