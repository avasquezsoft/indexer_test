import logging
import time
import jwt
import httpx

from config import (
    GITHUB_APP_ID,
    GITHUB_APP_INSTALLATION_ID,
    GITHUB_APP_INSTALLATION_IDS,
    GITHUB_APP_PRIVATE_KEY,
)

log = logging.getLogger(__name__)

# Extensiones de archivo que indexamos — incluye código, queries, configs, planillas y recursos web
SUPPORTED_EXTENSIONS = {
    # Lenguajes principales
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".cs", ".go", ".rb", ".php",
    ".sql", ".graphql", ".proto",
    ".md", ".txt",
    # Web / planillas / recursos
    ".html", ".htm", ".css", ".scss", ".sass", ".less",
    ".json", ".xml", ".yaml", ".yml",
    ".properties", ".conf", ".cfg", ".ini",
    ".jsp", ".jspf", ".tag",
    ".vue", ".svelte",
    ".ftl", ".hbs", ".mustache", ".twig",
    # Shell / scripts
    ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd",
    # Otros configs / docs
    ".dockerfile", ".env", ".gitignore", ".gitattributes",
    ".sql", ".plsql", ".pks", ".pkb",  # más variantes SQL/PLSQL
}

# Archivos y carpetas que ignoramos
IGNORED_PATHS = {
    "node_modules", ".git", "dist", "build",
    "__pycache__", ".next", "coverage", "vendor",
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml"
}

# Timeout para llamadas a la API de GitHub (segundos)
_GITHUB_TIMEOUT = 30.0


def _get_jwt_token() -> str:
    """Genera un JWT firmado con la private key de la GitHub App."""
    private_key_raw = GITHUB_APP_PRIVATE_KEY
    # El .env guarda \n como texto literal — los convertimos a saltos reales
    private_key = private_key_raw.replace("\\n", "\n")

    now = int(time.time())
    payload = {
        "iat": now - 60,
        "exp": now + (10 * 60),
        "iss": GITHUB_APP_ID,
    }
    return jwt.encode(payload, private_key, algorithm="RS256")


def get_installation_token(installation_id: str | None = None) -> str:
    """Obtiene un token de instalación. Si no se pasa ID, usa el primero configurado."""
    target_id = installation_id or GITHUB_APP_INSTALLATION_ID
    jwt_token = _get_jwt_token()

    with httpx.Client(timeout=_GITHUB_TIMEOUT) as client:
        response = client.post(
            f"https://api.github.com/app/installations/{target_id}/access_tokens",
            headers={
                "Authorization": f"Bearer {jwt_token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        response.raise_for_status()
        return response.json()["token"]


def get_installation_token_for_repo(owner: str, repo: str) -> str:
    """
    Prueba todos los tokens de instalación configurados hasta encontrar uno
    que tenga acceso al repositorio solicitado.
    """
    for inst_id in GITHUB_APP_INSTALLATION_IDS:
        try:
            token = get_installation_token(inst_id)
            # Verificación ligera: preguntamos si el repo existe para este token
            resp = httpx.get(
                f"https://api.github.com/repos/{owner}/{repo}",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
                timeout=5.0,
            )
            if resp.status_code == 200:
                return token
        except Exception as exc:
            log.debug(f"Instalación {inst_id} no tiene acceso a {owner}/{repo}: {exc}")
            continue

    # Fallback al primer ID para mantener comportamiento anterior
    log.warning(f"Ninguna instalación confirmó acceso a {owner}/{repo}, usando token por defecto")
    return get_installation_token()


def list_all_repos() -> list[dict]:
    """Lista todos los repos de todas las instalaciones configuradas."""
    all_repos = []
    for inst_id in GITHUB_APP_INSTALLATION_IDS:
        try:
            token = get_installation_token(inst_id)
            repos = list_repos(token)
            all_repos.extend(repos)
        except Exception as exc:
            log.warning(f"No se pudieron listar repos de instalación {inst_id}: {exc}")
    return all_repos


def list_repos(token: str) -> list[dict]:
    """Lista todos los repos de la instalación."""
    repos = []
    url = "https://api.github.com/installation/repositories"

    with httpx.Client(timeout=_GITHUB_TIMEOUT) as client:
        while url:
            response = client.get(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
                params={"per_page": 100},
            )
            response.raise_for_status()
            data = response.json()
            repos.extend(data.get("repositories", []))
            # Paginación
            url = response.links.get("next", {}).get("url")

    return repos


def get_repo_files(token: str, owner: str, repo: str, ref: str = "HEAD") -> list[dict]:
    """Obtiene el árbol completo de archivos de un repo."""
    with httpx.Client(timeout=_GITHUB_TIMEOUT) as client:
        response = client.get(
            f"https://api.github.com/repos/{owner}/{repo}/git/trees/{ref}",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            params={"recursive": "1"},
        )
        response.raise_for_status()
        tree = response.json().get("tree", [])

    # Filtramos solo archivos con extensiones soportadas
    files = []
    for item in tree:
        if item["type"] != "blob":
            continue
        path = item["path"]
        # Ignorar carpetas bloqueadas
        parts = path.split("/")
        if any(p in IGNORED_PATHS for p in parts):
            continue
        # Solo extensiones soportadas
        ext = __import__("os").path.splitext(path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        files.append({"path": path, "sha": item["sha"], "size": item.get("size", 0)})

    return files


class GitHubTokenExpired(Exception):
    """El token de instalación de GitHub expiró (401)."""
    pass


def get_file_content(token: str, owner: str, repo: str, path: str, ref: str = "HEAD") -> str | None:
    """Descarga el contenido de un archivo."""
    with httpx.Client(timeout=_GITHUB_TIMEOUT) as client:
        response = client.get(
            f"https://api.github.com/repos/{owner}/{repo}/contents/{path}",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github.raw+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            params={"ref": ref},
        )
        if response.status_code == 200:
            return response.text
        if response.status_code == 401:
            raise GitHubTokenExpired(f"Token expirado al leer {path}")
        return None
