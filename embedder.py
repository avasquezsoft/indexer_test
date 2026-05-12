import asyncio
import logging
import httpx

from config import OPENROUTER_API_BASE, OPENROUTER_API_KEY, OPENROUTER_EMBED_MODEL, VECTOR_SIZE

log = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BACKOFF_BASE = 2  # segundos


async def get_embedding(text: str) -> list[float]:
    """Genera el embedding para un único texto usando OpenRouter."""
    embeddings = await get_embeddings_batch([text])
    return embeddings[0]


async def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Genera embeddings para una lista de textos vía OpenRouter con reintentos."""
    api_base = OPENROUTER_API_BASE.rstrip("/")

    last_error: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    f"{api_base}/embeddings",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://tennis-doc.tritechprime.com",
                        "X-Title": "tennis-doc-app",
                    },
                    json={
                        "model": OPENROUTER_EMBED_MODEL,
                        "input": texts,
                    },
                )
                response.raise_for_status()
                data = response.json()

                if "data" not in data:
                    log.error(f"Respuesta inesperada de OpenRouter: {data}")
                    raise RuntimeError("La respuesta de embeddings no contiene 'data'")

                embeddings = [item["embedding"] for item in data["data"]]
                # Validar: no NaN/Inf, dimensión correcta
                for i, emb in enumerate(embeddings):
                    if len(emb) != VECTOR_SIZE:
                        log.error(f"Embedding {i} tiene dimensión {len(emb)}, se esperaba {VECTOR_SIZE}")
                        raise RuntimeError(f"Dimensión de embedding incorrecta: {len(emb)}")
                    if any(v != v or v == float("inf") or v == float("-inf") for v in emb):
                        log.error(f"Embedding {i} contiene NaN o Inf")
                        raise RuntimeError("Embedding contiene valores inválidos (NaN/Inf)")
                return embeddings

        except (httpx.HTTPStatusError, httpx.NetworkError, httpx.TimeoutException) as exc:
            last_error = exc
            if attempt == _MAX_RETRIES:
                break
            wait = _BACKOFF_BASE ** attempt
            log.warning(f"Error en embedding (intento {attempt}/{_MAX_RETRIES}): {exc}. Reintentando en {wait}s...")
            await asyncio.sleep(wait)

    raise RuntimeError(f"Fallo definitivo al generar embeddings tras {_MAX_RETRIES} intentos: {last_error}")
