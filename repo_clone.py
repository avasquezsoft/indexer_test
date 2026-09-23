"""
repo_clone.py — Gestión de clones locales de repositorios GitHub.

Mantiene una copia local (shallow) de cada repo+rama en disco para:
- Servir como respaldo ante fallos de la API de GitHub.
- Leer archivos completos rápidamente desde filesystem.
- Buscar código "en caliente" cuando el índice no trae suficiente contexto.
"""

import asyncio
import logging
import os
import shutil

from ast_parser import _get_parser
from config import CLONE_BASE_DIR
from github_client import get_installation_token_for_repo

logger = logging.getLogger(__name__)

# Extensiones que indexamos y que son relevantes para búsqueda en clon local
_CLONE_SEARCH_EXTS = {
    ".java", ".sql", ".xml", ".properties", ".yml", ".yaml",
    ".json", ".html", ".js", ".ts", ".py", ".md", ".txt", ".jsp", ".jspf",
}

_IGNORED_DIRS = {".git", "node_modules", "__pycache__", "target", "build", "dist", ".next", "coverage", "vendor"}


def _get_clone_path(repo: str, branch: str = "HEAD") -> str:
    """Ruta local del clon. Un directorio por repo y rama ("HEAD" = rama por defecto)."""
    safe_name = f"{repo.replace('/', '_')}@{branch.replace('/', '_')}"
    return os.path.join(CLONE_BASE_DIR, safe_name)


async def _git(*args: str, secret: str) -> tuple[int, str]:
    """Ejecuta git y devuelve (código, stderr) con el token enmascarado."""
    proc = await asyncio.create_subprocess_exec(
        "git", *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    return proc.returncode, stderr.decode(errors="replace").replace(secret, "***").strip()


async def clone_or_pull_repo(repo: str, branch: str = "HEAD") -> str | None:
    """
    Asegura que repo@branch esté clonado y actualizado en CLONE_BASE_DIR.
    Devuelve la ruta del clon, o None si no se pudo clonar.

    El token de instalación dura 1 hora: se pide uno nuevo en cada operación
    y nunca se guarda en .git/config.
    """
    path = _get_clone_path(repo, branch)
    owner, repo_name = repo.split("/", 1)
    token = await asyncio.to_thread(get_installation_token_for_repo, owner, repo_name)
    auth_url = f"https://x-access-token:{token}@github.com/{owner}/{repo_name}.git"

    if os.path.isdir(os.path.join(path, ".git")):
        logger.info("Actualizando clon %s @ %s", repo, branch)
        # "git fetch <url> HEAD" trae la rama por defecto del remoto
        code, err = await _git("-C", path, "fetch", "--depth", "1", auth_url, branch, secret=token)
        if code != 0:
            logger.warning("git fetch falló para %s @ %s: %s", repo, branch, err)
            return path
        code, err = await _git("-C", path, "reset", "--hard", "FETCH_HEAD", secret=token)
        if code != 0:
            logger.warning("git reset falló para %s @ %s: %s", repo, branch, err)
        return path

    logger.info("Clonando %s @ %s en %s", repo, branch, path)
    os.makedirs(CLONE_BASE_DIR, exist_ok=True)
    # Se clona en un directorio temporal y se renombra al final: si otro worker
    # clona el mismo repo a la vez, nadie ve (ni borra) un clon a medias.
    tmp_path = f"{path}.tmp-{os.getpid()}"
    shutil.rmtree(tmp_path, ignore_errors=True)
    # git no acepta "--branch HEAD"; sin --branch clona la rama por defecto
    branch_args = [] if branch == "HEAD" else ["--branch", branch]
    code, err = await _git("clone", "--depth", "1", *branch_args, auth_url, tmp_path, secret=token)
    if code != 0:
        logger.error("git clone falló para %s @ %s: %s", repo, branch, err)
        shutil.rmtree(tmp_path, ignore_errors=True)
        return None

    await _git("-C", tmp_path, "remote", "set-url", "origin",
               f"https://github.com/{owner}/{repo_name}.git", secret=token)
    try:
        os.rename(tmp_path, path)
    except OSError:
        # Otro worker terminó primero: se usa su clon
        shutil.rmtree(tmp_path, ignore_errors=True)
    return path


def read_file_from_clone(repo: str, file_path: str, branch: str = "HEAD") -> str | None:
    """
    Lee el contenido de un archivo desde el clon local.
    Devuelve None si el repo no está clonado, el archivo no existe
    o la ruta intenta salir del clon.
    """
    base = os.path.realpath(_get_clone_path(repo, branch))
    full_path = os.path.realpath(os.path.join(base, file_path))

    if not full_path.startswith(base + os.sep) or not os.path.isfile(full_path):
        return None

    try:
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as exc:
        logger.warning("Error leyendo %s de %s: %s", file_path, repo, exc)
        return None


# ─────────────────────────────────────────
# Búsqueda en el clon
# ─────────────────────────────────────────

def search_clone(
    repo: str,
    branch: str,
    keywords: list[str],
    entities: list[str],
    methods: list[str],
    max_files: int,
    max_chars_per_file: int,
) -> list[dict]:
    """
    Busca archivos del clon relevantes para la pregunta, ordenados por:
    1. el nombre del archivo es una clase mencionada (FooService.java),
    2. cantidad de keywords distintas que contiene,
    3. alguna keyword aparece en el nombre del archivo.

    Los archivos que superan max_chars_per_file se condensan en vez de cortarse.
    """
    clone_path = _get_clone_path(repo, branch)
    entity_names = {e.lower() for e in entities}
    terms = list(dict.fromkeys(k.lower() for k in [*keywords, *entities, *methods] if k))

    ranked = []
    # ponytail: recorre el repo completo en cada búsqueda; basta para repos
    # de miles de archivos. Si se vuelve lento, usar `git grep -l`.
    for root, dirs, fnames in os.walk(clone_path):
        dirs[:] = [d for d in dirs if d not in _IGNORED_DIRS]
        for fname in fnames:
            stem, ext = os.path.splitext(fname)
            if ext.lower() not in _CLONE_SEARCH_EXTS:
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().lower()
            except OSError:
                continue
            exact = stem.lower() in entity_names
            hits = sum(1 for t in terms if t in content)
            if not exact and hits == 0:
                continue
            in_name = any(t in stem.lower() for t in terms)
            ranked.append(((exact, hits, in_name), fpath))

    ranked.sort(key=lambda item: item[0], reverse=True)

    results = []
    for (exact, hits, _), fpath in ranked[:max_files]:
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        original_chars = len(content)
        if original_chars > max_chars_per_file:
            content = _condense(content, fpath, methods, terms, max_chars_per_file)
        results.append({
            "file_path": os.path.relpath(fpath, clone_path).replace("\\", "/"),
            "content": content,
            "match": "name" if exact else "keyword",
            "keyword_hits": hits,
            "original_chars": original_chars,
            "repo": repo,
            "branch": branch,
        })
    return results


def _condense(source: str, fpath: str, methods: list[str], terms: list[str], max_chars: int) -> str:
    """
    Reduce un archivo grande sin perder la estructura:
    - Java: conserva la clase, campos y firmas de todos los métodos; incluye
      completos los cuerpos de los métodos mencionados, luego los que más
      términos de la pregunta contienen y luego el resto en orden, hasta llenar
      max_chars. Los cuerpos que no caben se reemplazan por un comentario.
    - Otros lenguajes: se corta a max_chars.
    """
    truncated = source[:max_chars] + f"\n\n// ... archivo truncado ({len(source)} chars originales) ...\n"
    parser = _get_parser("java") if fpath.lower().endswith(".java") else None
    if parser is None:
        # ponytail: solo Java se condensa por AST; agregar otros lenguajes si hace falta
        return truncated

    data = source.encode("utf-8")
    tree = parser.parse(data)

    # Cuerpos de métodos/constructores de primer nivel (sin entrar en lambdas ni clases anónimas)
    bodies = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type in ("method_declaration", "constructor_declaration"):
            body = node.child_by_field_name("body")
            name = node.child_by_field_name("name")
            if body is not None and name is not None:
                bodies.append((body, data[name.start_byte:name.end_byte].decode("utf-8", "replace")))
            continue
        stack.extend(node.children)

    if not bodies:
        return truncated

    focus = set(methods)
    scored = []
    for body, name in bodies:
        text = data[body.start_byte:body.end_byte].decode("utf-8", "replace").lower()
        relevance = 1000 if name in focus else sum(text.count(t) for t in terms)
        scored.append((relevance, body))

    def placeholder(body) -> bytes:
        lines = body.end_point[0] - body.start_point[0] + 1
        return f"{{ /* ... {lines} líneas omitidas ... */ }}".encode()

    # Tamaño con todos los cuerpos colapsados; se expanden por relevancia mientras quepan
    size = len(data) - sum((b.end_byte - b.start_byte) - len(placeholder(b)) for _, b in scored)
    keep = set()
    # Primero los relevantes; con el espacio que sobre, el resto en orden del archivo
    for relevance, body in sorted(scored, key=lambda s: (-s[0], s[1].start_byte)):
        extra = (body.end_byte - body.start_byte) - len(placeholder(body))
        if size + extra > max_chars and relevance < 1000:
            continue
        keep.add(body.start_byte)
        size += extra

    out = data
    for _, body in sorted(scored, key=lambda s: s[1].start_byte, reverse=True):
        if body.start_byte not in keep:
            out = out[:body.start_byte] + placeholder(body) + out[body.end_byte:]

    omitted = len(bodies) - len(keep)
    header = (
        f"// [Tennis Doc] Vista condensada: {source.count(chr(10)) + 1} líneas originales. "
        f"Se muestran completos {len(keep)} métodos (primero los relacionados con la "
        f"pregunta); se omitieron los cuerpos de {omitted} por espacio.\n"
    )
    condensed = header + out.decode("utf-8", "replace")
    if len(condensed) > max_chars:
        condensed = condensed[:max_chars] + "\n\n// ... vista condensada truncada ...\n"
    return condensed
