import asyncio
import logging
import re
import httpx

from config import OPENROUTER_API_BASE, OPENROUTER_API_KEY, OPENROUTER_EMBED_MODEL, VECTOR_SIZE

log = logging.getLogger(__name__)

_MAX_RETRIES = 5
_BACKOFF_BASE = 2  # segundos
_BATCH_SIZE = 25  # máximo de textos por request a OpenRouter (reducido por rate limits)
_BATCH_DELAY = 0.5  # segundos entre batches para no saturar tokens/seg
_MAX_EMBED_CHARS = 20000  # ~6500 tokens para código (margen seguro < 8192)

_RETRY_AFTER_RE = re.compile(r"after\s+(\d+(?:\.\d+)?)\s*sec", re.IGNORECASE)


class _RateLimitError(Exception):
    """Excepción interna para rate limits devueltos en body JSON (HTTP 200 con error 429)."""
    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


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
        # Pausa entre batches para respetar la cuota de tokens/segundo de OpenRouter
        if batch_start + _BATCH_SIZE < len(truncated_texts):
            await asyncio.sleep(_BATCH_DELAY)

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

                # OpenRouter a veces responde HTTP 200 con error 429 dentro del body
                if "error" in data and data["error"].get("code") == 429:
                    msg = data["error"].get("message", "")
                    match = _RETRY_AFTER_RE.search(msg)
                    retry_after = float(match.group(1)) if match else None
                    log.warning("Rate limit 429 en body JSON: %s", msg)
                    raise _RateLimitError(f"OpenRouter rate limit: {msg}", retry_after=retry_after)

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

        except (httpx.HTTPStatusError, httpx.NetworkError, httpx.TimeoutException, _RateLimitError) as exc:
            last_error = exc
            if attempt == _MAX_RETRIES:
                break
            # Si el error trae un retry_after exacto (del body JSON de OpenRouter), usarlo + margen
            if isinstance(exc, _RateLimitError) and exc.retry_after is not None:
                wait = exc.retry_after + 0.5
            else:
                wait = _BACKOFF_BASE ** attempt
            log.warning(f"Error en embedding (intento {attempt}/{_MAX_RETRIES}): {exc}. Reintentando en {wait}s...")
            await asyncio.sleep(wait)

    raise RuntimeError(f"Fallo definitivo al generar embeddings tras {_MAX_RETRIES} intentos: {last_error}")
