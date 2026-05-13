# Tennis Doc IA — Indexador de Código con RAG

> Sistema RAG (Retrieval-Augmented Generation) especializado en indexar repositorios GitHub y responder preguntas técnicas mediante chat con IA.

## 🚀 Stack Tecnológico

| Servicio | Tecnología | Rol |
|----------|-----------|-----|
| **Indexador** | Python 3.13 + FastAPI | Orquesta indexación, búsqueda y generación |
| **JavaParser** | Java 21 + Javalin | Parseo AST de Java/Spring Boot |
| **Vector DB** | Qdrant | Embeddings de código (1024 dims, coseno) |
| **Graph DB** | Neo4j 5.26 Community + APOC | Grafo de entidades y dependencias |
| **LLM Gateway** | LiteLLM | Proxy unificado hacia OpenRouter |
| **Chat UI** | Open WebUI | Interfaz web de conversación |
| **Base de datos** | PostgreSQL 17 | Persistencia de LiteLLM |

## 📁 Estructura del Repositorio

```
.
├── main.py              # FastAPI: endpoints, webhook, lifespan
├── config.py            # Variables de entorno centralizadas
├── github_client.py     # Auth GitHub App (JWT) + fetch de repos
├── ast_parser.py        # AST y entidades (Tree-sitter + JavaParser HTTP)
├── chunker.py           # División inteligente de archivos en chunks
├── embedder.py          # Embeddings vía OpenRouter (batch + retries)
├── qdrant_store.py      # Cliente Qdrant
├── graph_store.py       # Cliente Neo4j
├── rag_engine.py        # Motor híbrido: vectorial + grafo + keyword
├── tennis_doc_rag.py    # Filtro/plugin de Open WebUI
├── docker-compose.yml   # Orquestación completa del stack
├── .env.example         # Plantilla de variables de entorno
├── arquitectura.html    # Documentación visual interactiva
└── parser_java/         # Microservicio Java (Maven + JavaParser)
```

## 🛠️ Configuración Rápida

1. Copiar `.env.example` a `.env` y completar valores.
2. Levantar el stack:
   ```bash
   docker compose up -d
   ```
3. Acceder a Open WebUI en el dominio configurado.

## 🔑 Variables de Entorno Principales

- `INDEXER_API_KEY` — Protege endpoints públicos del indexer (recomendado en producción).
- `WEBHOOK_SECRET` — Valida firmas HMAC de webhooks de GitHub.
- `GITHUB_APP_*` — Credenciales de la GitHub App.
- `OPENROUTER_API_KEY` / `OPENROUTER_EMBED_MODEL` — Embeddings y chat vía OpenRouter.
- `QDRANT_*`, `NEO4J_*` — Bases de datos vectorial y grafo.
- `LITELLM_MASTER_KEY` — API key maestra de LiteLLM.

Ver `.env.example` para la lista completa y descripciones.

## 📡 Endpoints del Indexador

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `POST` | `/index` | Indexación manual de un repo/rama |
| `POST` | `/webhook` | Webhook de GitHub (push events) |
| `POST` | `/search` | Búsqueda semántica pura en Qdrant |
| `POST` | `/search-augmented` | Vectorial + todos los chunks de archivos top |
| `POST` | `/search-graph` | Híbrida: vector + grafo + keyword |
| `GET`  | `/graph/entity/{name}` | Entidad por nombre y relaciones directas |
| `GET`  | `/graph/related/{entity_id}` | Vecinos en el grafo |
| `GET`  | `/repos` | Lista repos únicos indexados en Neo4j |
| `POST` | `/fetch-file` | Contenido crudo de archivo desde GitHub |
| `POST` | `/pdf` | Genera PDF descargable |
| `POST` | `/markdown` | Genera Markdown descargable |
| `GET`  | `/health` | Health check de Qdrant, Neo4j y JavaParser |

> **Nota:** todos los endpoints (excepto `/webhook` y `/health`) requieren el header `Authorization: Bearer <INDEXER_API_KEY>` cuando la variable está configurada.

## 💬 Comandos del Chat (Open WebUI)

| Comando | Acción |
|---------|--------|
| `indexa org/repo [rama]` | Dispara re-indexación manual |
| `grafo NombreClase` | Muestra relaciones del grafo para una clase |
| `repos` | Lista todos los repositorios indexados |
| `"generar markdown"` / `"guardar como md"` | Exporta contexto recuperado a `.md` |
| `"generar pdf"` | Convierte la última respuesta a PDF |
| `del repo org/repo ...` | Enfoca la búsqueda en un repo específico |

## 🏗️ Arquitectura

Ver `arquitectura.html` para un diagrama visual interactivo completo del flujo de datos, servicios y endpoints.

## 📝 Licencia

GNU General Public License v3.0 (GPL-3.0). Ver `LICENSE`.

---

**Tritech Prime** · [tennis-doc.tritechprime.com](https://tennis-doc.tritechprime.com)
