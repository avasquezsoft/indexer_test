import asyncio
import logging
import httpx

from config import OPENROUTER_API_BASE, OPENROUTER_API_KEY, OPENROUTER_EMBED_MODEL, VECTOR_SIZE

log = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BACKOFF_BASE = 2  # segundos
_BATCH_SIZE = 50  # máximo de textos por request a OpenRouter
_MAX_EMBED_CHARS = 30000  # ~7500 tokens para código (estimación conservadora < 8192)


async def get_embedding(text: str) -> list[float]:
    """Genera el embedding para un único texto usando OpenRouter."""
    embeddings = await get_embeddings_batch([text])
    return embeddings[0]


async def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Genera embeddings para una lista de textos vía OpenRouter con reintentos.

    Divide automáticamente en sub-batches para evitar 'Too many tokens',
    y trunca textos individuales que excedan el límite de tokens del modelo.
    """
    api_base = OPENROUTER_API_BASE.rstrip("/")

    if not texts:
        return []

    # Truncar textos excesivamente largos para que ninguno exceda el max de tokens
    truncated_texts = []
    for i, t in enumerate(texts):
        if len(t) > _MAX_EMBED_CHARS:
            log.warning("Texto %s excede %s chars (%s), truncando para embedding", i, _MAX_EMBED_CHARS, len(t))
            truncated_texts.append(t[:_MAX_EMBED_CHARS])
        else:
            truncated_texts.append(t)

    all_embeddings: list[list[float]] = []

    for batch_start in range(0, len(truncated_texts), _BATCH_SIZE):
        batch = truncated_texts[batch_start : batch_start + _BATCH_SIZE]
        batch_embeddings = await _embed_single_batch(batch, api_base)
        all_embeddings.extend(batch_embeddings)
        log.info("Embedding batch %s-%s/%s OK", batch_start + 1, batch_start + len(batch), len(truncated_texts))

    return all_embeddings


async def _embed_single_batch(texts: list[str], api_base: str) -> list[list[float]]:
    """Envía un sub-batch a OpenRouter con reintentos."""
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
                # Validar: no NaN/Inf, dimensión correcta (truncar si es mayor, modelo Matryoshka)
                for i, emb in enumerate(embeddings):
                    if len(emb) > VECTOR_SIZE:
                        # Modelo Matryoshka (ej: Codestral Embed): dimensiones ordenadas por relevancia
                        log.debug(f"Embedding {i} tiene {len(emb)} dims, truncando a {VECTOR_SIZE}")
                        emb = emb[:VECTOR_SIZE]
                        embeddings[i] = emb
                    if len(emb) != VECTOR_SIZE:
                        log.error(f"Embedding {i} tiene dimensión {len(emb)}, se esperaba {VECTOR_SIZE}")
                        raise RuntimeError(f"Dimensión de embedding incorrecta: {len(emb)}. Verificá que OPENROUTER_EMBED_MODEL y VECTOR_SIZE coincidan.")
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
