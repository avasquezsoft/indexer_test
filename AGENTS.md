# Tennis Doc — Guía para Agentes de Código

> Este documento está dirigido a agentes de IA que necesiten entender, modificar o extender el proyecto. El lector se asume sin conocimiento previo del sistema.

---

## 1. Visión general del proyecto

**Tennis Doc** (también referido como *Tennis Doc IA*) es un sistema RAG (*Retrieval-Augmented Generation*) especializado en indexar repositorios de código fuente y responder preguntas técnicas sobre ellos mediante un chat con inteligencia artificial.

El stack completo incluye:

- **Indexador** (Python + FastAPI): orquesta la indexación, búsqueda y generación de documentos.
- **JavaParser** (Java 21 + Javalin): microservicio REST que parsea archivos Java con *JavaParser* y extrae entidades/relaciones de forma precisa.
- **Qdrant**: base de datos vectorial para embeddings de código.
- **Neo4j**: grafo de entidades de código (clases, métodos, campos, relaciones de herencia, inyección, llamadas).
- **PostgreSQL**: persistencia de LiteLLM.
- **LiteLLM**: proxy unificado hacia proveedores de LLM (en producción apunta a OpenRouter).
- **Open WebUI**: interfaz de chat web donde los usuarios interactúan con el sistema.

Todo el despliegue se realiza mediante **Docker Compose** (`docker-compose.yml`).

---

## 2. Estructura del repositorio

```
.
├── main.py              # FastAPI: endpoints, webhook, lifespan, lógica de indexación
├── config.py            # Variables de entorno centralizadas (obligatorias vs opcionales)
├── github_client.py     # Autenticación GitHub App (JWT) + fetch de repos y archivos
├── ast_parser.py        # Extracción de AST y entidades (Tree-sitter + JavaParser HTTP)
├── chunker.py           # División de archivos en chunks con lógica por lenguaje
├── embedder.py          # Generación de embeddings vía OpenRouter (batch + reintentos)
├── qdrant_store.py      # Cliente Qdrant: colecciones, upsert, búsqueda, scroll
├── graph_store.py       # Cliente Neo4j: schema, upsert de entidades, navegación de grafo
├── rag_engine.py        # Motor híbrido de retrieval: vectorial + grafo + keyword
├── tennis_doc_rag.py    # Filtro/plugin de Open WebUI (inyección de contexto en el chat)
├── requirements.txt     # Dependencias Python
├── docker-compose.yml   # Orquestación de todo el stack
├── .env.example         # Plantilla de variables de entorno
├── parser_java/         # Microservicio Java (Maven, Javalin, JavaParser)
│   ├── pom.xml
│   └── src/main/java/com/tritech/javaparser/App.java
└── AGENTS.md            # Este archivo
```

---

## 3. Tecnologías y dependencias principales

### Python (indexador)
- `fastapi==0.136.1` + `uvicorn==0.34.0`
- `qdrant-client==1.17.1`
- `neo4j==5.27.0`
- `tree-sitter==0.24.0` + bindings para Java, Python, JavaScript, TypeScript, Go
- `httpx==0.28.1`
- `pydantic==2.10.4`
- `fpdf2==2.8.3` (generación de PDFs)
- `PyJWT==2.10.1` + `cryptography==44.0.0` (GitHub App)
- `tenacity==9.0.0`

### Java (parser)
- Java 21
- Maven 3.9
- Javalin 7.1.0
- JavaParser 3.26.3 (symbol solver core)
- Jackson 2.18.2

### Infraestructura
- Qdrant (imagen oficial `latest`)
- Neo4j 5.26 Community (con plugin APOC)
- PostgreSQL 17 Alpine
- LiteLLM (`ghcr.io/berriai/litellm:main-latest`)
- Open WebUI (`ghcr.io/open-webui/open-webui:main`)

---

## 4. Flujo de datos y arquitectura

### Indexación (pipeline)
1. Se dispara vía webhook de GitHub (`/webhook`) o manualmente (`POST /index`).
2. `github_client.py` obtiene token de instalación y lista todos los archivos del repo/rama.
3. Para cada archivo:
   - Se detecta el lenguaje por extensión.
   - Se parsea con `ast_parser.py` (Tree-sitter o JavaParser).
   - Se generan **entidades** (clases, métodos, campos, funciones) y **chunks**.
   - Los chunks se enriquecen con texto para embedding (`embed_text`).
   - Si es Java, se resuelven referencias a archivos `.sql` y se adjunta el contenido SQL inline.
4. Flush cada 500 entidades / 1000 chunks:
   - Entidades → Neo4j (`graph_store.upsert_entities`).
   - Chunks + embeddings → Qdrant (`qdrant_store.upsert_chunks`).

### Búsqueda (retrieval)
El sistema ofrece tres estrategias de búsqueda:

1. **`/search`** — puramente vectorial en Qdrant.
2. **`/search-augmented`** — vectorial para identificar archivos relevantes, luego trae **todos** los chunks de esos archivos (máx 10 archivos, 50 candidatos vectoriales).
3. **`/search-graph`** — híbrida (vector + grafo + keyword) mediante `rag_engine.py`:
   - Búsqueda vectorial en Qdrant.
   - Expansión por grafo en Neo4j (vecinos hasta profundidad configurable).
   - Búsqueda por nombre exacto si la query menciona identificadores en CamelCase.

### Chat (Open WebUI)
- `tennis_doc_rag.py` es un **filtro** de Open WebUI que intercepta mensajes del usuario.
- **Orden de retrieval actual:** primero `/search-augmented` (archivos completos), luego `/search-graph` (relaciones), y finalmente fetch directo de entidades del grafo si la query menciona nombres de clase.
- Detecta automáticamente formato `org/repo` y ramas.
- Enriquece la query con keywords técnicas según el contexto.
- Inyecta el contexto recuperado como mensaje `system` antes de enviar al LLM.
- Soporta comandos especiales:
  - `indexa org/repo [rama]` → reindexación manual.
  - `grafo NombreClase` → muestra relaciones del grafo.
  - `repos` → lista todos los repositorios indexados.
  - frases como "generar pdf" → convierte la última respuesta del asistente a PDF.
  - frases como "generar markdown", "guardar como md", "exportar a markdown" → exporta chunks recuperados a `.md` descargable o lo muestra en el chat si es pequeño.

### Chat (Open WebUI)
- `tennis_doc_rag.py` es un **filtro** de Open WebUI que intercepta mensajes del usuario.
- Detecta automáticamente formato `org/repo` y ramas.
- Enriquece la query con keywords técnicas según el contexto.
- Llama al endpoint `/search-graph` del indexador.
- Inyecta el contexto recuperado como mensaje `system` antes de enviar al LLM.
- Soporta comandos especiales:
  - `indexa org/repo [rama]` → reindexación manual.
  - `grafo NombreClase` → muestra relaciones del grafo.
  - `repos` → lista todos los repositorios indexados.
  - frases como "generar pdf" → convierte la última respuesta del asistente a PDF.
  - frases como "generar markdown", "guardar como md", "exportar a markdown" → exporta chunks recuperados a `.md` descargable o lo muestra en el chat si es pequeño.

---

## 5. Configuración y entorno

Toda la configuración pasa por **variables de entorno**. No hay archivos de config estáticos.

### Variables obligatorias
- `INDEXER_REPO_URL` — URL del repo del indexador (para clonar dentro del contenedor).
- `GITHUB_APP_ID`, `GITHUB_APP_CLIENT_ID`, `GITHUB_APP_CLIENT_SECRET`, `GITHUB_APP_INSTALLATION_ID`, `GITHUB_APP_PRIVATE_KEY` — credenciales de la GitHub App. `GITHUB_APP_INSTALLATION_ID` soporta múltiples IDs separados por coma para instalaciones en distintas organizaciones.
- `OPENROUTER_API_BASE`, `OPENROUTER_API_KEY`, `OPENROUTER_EMBED_MODEL` — embeddings vía OpenRouter.
- `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION` — vector DB.
- `NEO4J_URL`, `NEO4J_USER`, `NEO4J_PASSWORD` — graph DB.
- `WEBHOOK_SECRET` — para validar firmas HMAC de webhooks de GitHub.
- `DATABASE_URL` — PostgreSQL (usada por LiteLLM).
- `LITELLM_MASTER_KEY` — API key maestra de LiteLLM.

### Variables opcionales (con defaults)
- `VECTOR_SIZE` — dimensión de embeddings (default: `1024`).
- `JAVAPARSER_URL` — default: `http://javaparser:8080`.
- `PARSER_JAVA_REPO_URL` — repo del parser Java (para clonar en contenedor).
- `INDEXER_API_KEY` — API key para proteger endpoints públicos del indexer (recomendado en producción). Si no se define, los endpoints son públicos.

### Archivo de entorno
- Copiar `.env.example` a `.env` y completar los valores.
- **Nunca commitear `.env`** — ya está en `.gitignore` por convención.
- La private key de GitHub debe estar en **una sola línea** con `\n` como separadores (no saltos de línea reales).

---

## 6. Cómo levantar el proyecto

### Desarrollo local (solo indexador)
```bash
# 1. Crear entorno virtual (recomendado)
python -m venv .venv
source .venv/bin/activate  # o .venv\Scripts\activate en Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Exportar variables de entorno mínimas (ver .env.example)
export QDRANT_URL=http://localhost:6333
export QDRANT_COLLECTION=code_chunks
# ... etc

# 4. Levantar Uvicorn
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

### Producción / stack completo
```bash
docker compose up -d
```

Esto levanta: Qdrant, PostgreSQL, Neo4j, JavaParser, LiteLLM, Indexer y Open WebUI.

Los servicios internos se exponen entre sí por nombre de red Docker (ej: `http://qdrant:6333`, `http://neo4j:7474`, `http://javaparser:8080`, `http://indexer:8001`).

---

## 7. Convenciones de código

### Idioma
- Todo el código, comentarios, docstrings, logs y mensajes de error están en **español**.
- Los nombres de variables y funciones usan snake_case en Python, camelCase en Java.

### Estilo Python
- Imports agrupados: stdlib → terceros → locales.
- Usar `logging` con formato `"%(asctime)s %(levelname)s %(message)s"`.
- Tipado progresivo (`str | None`, `list[dict]`, etc.) usando sintaxis de Python 3.10+.
- Pydantic `BaseModel` para validación de requests en FastAPI.
- Funciones async para I/O (llamadas a APIs, embedding).

### Estilo Java
- Código con sangrado estilo K&R (llave en la misma línea).
- Uso extensivo de `var` (Java 10+).
- Formato vertical con líneas en blanco entre bloques lógicos.

### Manejo de errores
- Los errores de servicios externos (Qdrant, Neo4j, OpenRouter, GitHub) se capturan, loguean y **nunca rompen el flujo principal del chat**.
- En el filtro de Open WebUI (`tennis_doc_rag.py`), cualquier excepción en `inlet` devuelve el `body` sin modificar para no interrumpir la conversación.
- Reintentos con backoff exponencial en `embedder.py` (hasta 3 intentos).
- Reintentos de token de GitHub en `main.py` (hasta 2 intentos por archivo).

---

## 8. Endpoints principales del indexador

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `POST` | `/index` | Indexación manual de un repo/rama (async en background) |
| `POST` | `/webhook` | Webhook de GitHub para reindexar automáticamente en push |
| `POST` | `/search` | Búsqueda semántica pura en Qdrant |
| `POST` | `/search-augmented` | Búsqueda híbrida: vectorial + archivos completos |
| `POST` | `/search-graph` | Búsqueda híbrida vector + grafo + keyword |
| `GET`  | `/graph/entity/{name}` | Busca entidad por nombre y devuelve relaciones |
| `GET`  | `/graph/related/{entity_id}` | Vecinos en el grafo hasta cierta profundidad |
| `POST` | `/fetch-file` | Trae contenido crudo de un archivo desde GitHub |
| `POST` | `/pdf` | Genera PDF a partir de texto/markdown |
| `POST` | `/markdown` | Genera archivo `.md` descargable |
| `GET`  | `/health` | Health check con estado de Qdrant, Neo4j y JavaParser |
| `GET`  | `/repos` | Lista repos únicos indexados en Neo4j |
| `GET`  | `/repos-available` | Lista repos a los que la GitHub App tiene acceso (soporta multi-org) |
| `GET`  | `/debug/files` | Lista archivos que serían indexados (sin indexar) |
| `GET`  | `/debug/files-indexed` | Lista archivos ya indexados en Qdrant |
| `GET`  | `/debug/chunks` | Muestra chunks de un archivo específico en Qdrant |

> **Nota de seguridad:** todos los endpoints (excepto `/webhook` y `/health`) validan el header `Authorization: Bearer <INDEXER_API_KEY>` cuando la variable está configurada.

---

## 9. Estrategias de chunking

El archivo `chunker.py` implementa splitting inteligente por lenguaje:

- **Regla de oro**: si el archivo entero cabe en `CHUNK_SIZE` (8000 caracteres), va en **un solo chunk**.
- Archivos grandes se dividen por bloques lógicos:
  - **SQL**: por sentencias (`;`), respetando bloques PL/SQL y strings.
  - **Java**: por miembros (métodos, campos, clases), manteniendo anotaciones (`@...`) con el miembro siguiente.
  - **Python/JS/TS/Go/C#**: por definiciones de funciones/clases.
  - **HTML/XML/JSP**: por etiquetas de cierre de bloque.
  - **JSON**: por objetos/arrays cerrados.
- Fallback: división por tamaño con overlap (`CHUNK_OVERLAP = 400`) y corte en líneas vacías o fin de sentencia.

**Mitigación de chunks gigantes:**
- SQL inlineado en chunks Java se trunca a **6.000 caracteres** (el archivo `.sql` completo sigue indexándose por separado).
- Clases/interfaces con más de **12.000 caracteres** se truncan en el chunk visible; los métodos/fields individuales cubren el contenido restante.
- El `embed_text` de entidades AST limita el código a **6.000 caracteres** y las relaciones a **20**, evitando truncamiento por el embedder.

Cada chunk incluye `embed_text` enriquecido con metadatos del repo, rama, archivo, lenguaje y contexto semántico adicional.

---

## 10. Grafo de código (Neo4j)

### Nodos
- Etiqueta: `:CodeEntity`
- Propiedades: `id`, `name`, `type`, `language`, `repo`, `branch`, `file_path`, `start_line`, `end_line`, `signature`, `docstring`, `code`, `annotations`

### Relaciones
- `EXTENDS` — herencia de clase
- `IMPLEMENTS` — implementación de interfaz
- `HAS_METHOD` — clase contiene método
- `HAS_FIELD` — clase contiene campo
- `CALLS` — método llama a otro método
- `IMPORTS` — importación de clase
- `INJECTED` — inyección de dependencia (`@Autowired`, `@Inject`, `@Resource`)
- `ANNOTATED_WITH` — anotación sobre entidad

### IDs
Los IDs de entidad siguen el formato:
```
{repo}:{branch}:{file_path}:{type}:{name}
```

---

## 11. Testing

> **Nota importante**: el proyecto **no cuenta actualmente con tests automatizados**. No hay directorio `tests/`, ni configuración de pytest, ni unittest.

Si se agregan tests en el futuro, se recomienda:
- Usar `pytest` como runner.
- Mockear llamadas a OpenRouter, GitHub, Qdrant y Neo4j.
- Testear especialmente:
  - `chunker.py` — splitting por lenguaje y regla de oro.
  - `ast_parser.py` — extracción de entidades para cada lenguaje soportado.
  - `github_client.py` — manejo de paginación y refresh de token.
  - `rag_engine.py` — scoring y deduplicación de resultados híbridos.

---

## 12. Seguridad

- Validación HMAC-SHA256 en webhooks de GitHub (`/webhook`).
- Todas las claves y tokens se leen desde variables de entorno; nunca hardcodeados.
- La GitHub App usa autenticación con JWT firmado (RS256) + token de instalación de corta duración.
- Open WebUI requiere `LITELLM_MASTER_KEY` para hablar con LiteLLM.
- Qdrant y Neo4j usan autenticación por API key / credenciales.
- CORS habilitado para `*` en el indexador (intencional para entornos controlados; revisar si se expone públicamente).
- **API Key en endpoints:** si se define `INDEXER_API_KEY`, todos los endpoints del indexer (excepto `/webhook` y `/health`) exigen el header `Authorization: Bearer <token>`.
- El filtro `tennis_doc_rag.py` lee `INDEXER_API_KEY` desde variables de entorno y lo envía en cada request al indexer.

---

## 13. Despliegue

El despliegue es 100% Docker Compose:

- Los contenedores `indexer` y `javaparser` se **clonan a sí mismos desde GitHub** al arrancar, usando las URLs configuradas en `INDEXER_REPO_URL` y `PARSER_JAVA_REPO_URL`. Esto permite desplegar actualizaciones sin rebuild de imagen.
- Los volúmenes nombrados persisten datos de Qdrant, PostgreSQL, Neo4j, Open WebUI y cachés.
- LiteLLM se configura mediante `configs` inline en `docker-compose.yml` (no archivo externo).
- El filtro `tennis_doc_rag.py` se inyecta en Open WebUI también vía `configs` de Docker Compose.

### Escalado y limitaciones conocidas
- El embedding se hace secuencialmente por batch (máx 50 textos por request). No hay paralelismo de embeddings.
- La indexación de un repo grande puede tardar varios minutos; corre en background tasks de FastAPI.
- Neo4j Community tiene limitaciones de clustering; para alta disponibilidad requeriría Enterprise.

---

## 14. Licencia

El proyecto está licenciado bajo **GNU General Public License v3.0** (GPL-3.0). Ver `LICENSE`.

---

## 15. Contacto y mantenimiento

- Autor: **Tritech Prime**
- Web del producto: `https://tennis-doc.tritechprime.com`
- Repositorio del indexador: `https://github.com/avasquezsoft/indexer_test.git`
- Repositorio del parser Java: `https://github.com/avasquezsoft/parser_java.git`
