import os


# ─────────────────────────────────────────
# Configuración centralizada del indexador
# ─────────────────────────────────────────

INDEXER_REPO_URL = os.environ["INDEXER_REPO_URL"]
PARSER_JAVA_REPO_URL = os.environ.get("PARSER_JAVA_REPO_URL", "")

# GitHub App
GITHUB_APP_ID = os.environ["GITHUB_APP_ID"]
GITHUB_APP_CLIENT_ID = os.environ["GITHUB_APP_CLIENT_ID"]
GITHUB_APP_CLIENT_SECRET = os.environ["GITHUB_APP_CLIENT_SECRET"]
GITHUB_APP_INSTALLATION_ID = os.environ["GITHUB_APP_INSTALLATION_ID"]
GITHUB_APP_PRIVATE_KEY = os.environ["GITHUB_APP_PRIVATE_KEY"]

# OpenRouter
OPENROUTER_API_BASE = os.environ["OPENROUTER_API_BASE"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_EMBED_MODEL = os.environ["OPENROUTER_EMBED_MODEL"]

# Qdrant
QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION")

# Neo4j
NEO4J_URL = os.environ["NEO4J_URL"]
NEO4J_USER = os.environ["NEO4J_USER"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

# Vector size (dimensión del modelo de embeddings)
VECTOR_SIZE = int(os.environ.get("VECTOR_SIZE", "1024"))

# JavaParser service
JAVAPARSER_URL = os.environ.get("JAVAPARSER_URL", "http://javaparser:8080")

# Webhook
WEBHOOK_SECRET = os.environ["WEBHOOK_SECRET"]

# API Key para proteger endpoints públicos (opcional pero recomendado)
INDEXER_API_KEY = os.environ.get("INDEXER_API_KEY", "")

# Directorio base para clones locales de repos (respaldos + lectura rápida)
CLONE_BASE_DIR = os.environ.get("CLONE_BASE_DIR", "/repos")
