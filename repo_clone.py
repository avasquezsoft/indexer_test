"""
repo_clone.py — Gestión de clones locales de repositorios GitHub.

Mantiene una copia local de cada repo en disco para:
- Servir como respaldo ante fallos de la API de GitHub.
- Leer archivos completos rápidamente desde filesystem.
- Sincronizar automáticamente ante pushes (git pull).
"""

import asyncio
import logging
import os

from config import CLONE_BASE_DIR
from github_client import get_installation_token

logger = logging.getLogger(__name__)


def _get_clone_path(repo: str) -> str:
    """Devuelve la ruta local donde se clonará el repo."""
    safe_name = repo.replace("/", "_")
    return os.path.join(CLONE_BASE_DIR, safe_name)


async def clone_or_pull_repo(repo: str, branch: str = "HEAD") -> str:
    """
    Asegura que el repo esté clonado localmente en CLONE_BASE_DIR.
    Si ya existe, hace git pull. Si no, lo clona.
    Devuelve la ruta del directorio clonado.
    """
    path = _get_clone_path(repo)
    token = get_installation_token()
    owner, repo_name = repo.split("/", 1)
    clone_url = f"https://x-access-token:{token}@github.com/{owner}/{repo_name}.git"

    if os.path.exists(os.path.join(path, ".git")):
        logger.info("Repo ya clonado en %s. Haciendo fetch + reset a origin/%s", path, branch)
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "-C", path, "fetch", "origin", branch,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                logger.warning("git fetch falló para %s: %s", repo, stderr.decode().strip())

            proc = await asyncio.create_subprocess_exec(
                "git", "-C", path, "reset", "--hard", f"origin/{branch}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                logger.warning("git reset falló para %s: %s", repo, stderr.decode().strip())
        except Exception as exc:
            logger.warning("Error haciendo pull de %s: %s", repo, exc)
    else:
        logger.info("Clonando %s @ %s en %s", repo, branch, path)
        os.makedirs(path, exist_ok=True)
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "clone", "--branch", branch, "--single-branch", clone_url, path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                logger.error("git clone falló para %s: %s", repo, stderr.decode().strip())
        except Exception as exc:
            logger.error("Error clonando %s: %s", repo, exc)

    return path


def read_file_from_clone(repo: str, file_path: str) -> str | None:
    """
    Lee el contenido de un archivo desde el clon local.
    Devuelve None si el repo no está clonado o el archivo no existe.
    """
    path = _get_clone_path(repo)
    full_path = os.path.join(path, file_path)

    if not os.path.exists(full_path):
        return None

    try:
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as exc:
        logger.warning("Error leyendo %s de %s: %s", file_path, repo, exc)
        return None
