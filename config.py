import os


# ─────────────────────────────────────────
# Configuración centralizada del indexador
# ─────────────────────────────────────────

INDEXER_REPO_URL = os.environ["INDEXER_REPO_URL"]

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

# Webhook
WEBHOOK_SECRET = os.environ["WEBHOOK_SECRET"]
