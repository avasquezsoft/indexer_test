import os
import re

# Tamaño máximo de un chunk en caracteres
CHUNK_SIZE = 1500
# Overlap entre chunks para no perder contexto
CHUNK_OVERLAP = 200


def chunk_file(content: str, file_path: str, repo: str, branch: str = "HEAD") -> list[dict]:
    """
    Divide un archivo en chunks con metadata.
    Intenta dividir por funciones/clases cuando es posible,
    si no, divide por tamaño con overlap.
    """
    ext = os.path.splitext(file_path)[1].lower()
    language = _detect_language(ext)

    # Para archivos pequeños, un solo chunk
    if len(content) <= CHUNK_SIZE:
        return [_make_chunk(content, file_path, repo, branch, language, 0)]

    # Intentar dividir por bloques lógicos según el lenguaje
    if language in ("python", "javascript", "typescript", "java", "csharp", "go"):
        chunks = _split_by_functions(content, file_path, repo, branch, language)
        if chunks:
            return chunks

    # Fallback: dividir por tamaño con overlap
    return _split_by_size(content, file_path, repo, branch, language)


def _split_by_functions(content: str, file_path: str, repo: str, branch: str, language: str) -> list[dict]:
    """Divide buscando definiciones de funciones/clases."""
    patterns = {
        "python": r"^(class |def |\s{0,4}def |\s{0,4}async def )",
        "javascript": r"^(function |const \w+ = |export (default |async )?function |class )",
        "typescript": r"^(function |const \w+ = |export (default |async )?function |class |interface |type \w+ =)",
        "java": r"^\s*(public|private|protected|static).*\{$",
        "csharp": r"^\s*(public|private|protected|static).*\{$",
        "go": r"^func ",
    }

    pattern = patterns.get(language)
    if not pattern:
        return []

    lines = content.split("\n")
    split_indices = [0]

    for i, line in enumerate(lines):
        if i > 0 and re.match(pattern, line):
            split_indices.append(i)

    if len(split_indices) <= 1:
        return []

    chunks = []
    for idx, start in enumerate(split_indices):
        end = split_indices[idx + 1] if idx + 1 < len(split_indices) else len(lines)
        block = "\n".join(lines[start:end]).strip()

        # Si el bloque es muy grande, subdividirlo
        if len(block) > CHUNK_SIZE * 2:
            sub = _split_by_size(block, file_path, repo, branch, language)
            chunks.extend(sub)
        elif block:
            chunks.append(_make_chunk(block, file_path, repo, branch, language, start))

    return chunks


def _split_by_size(content: str, file_path: str, repo: str, branch: str, language: str) -> list[dict]:
    """Divide por tamaño con overlap."""
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(content):
        end = start + CHUNK_SIZE
        chunk_text = content[start:end]

        # Intentar cortar en un salto de línea limpio
        if end < len(content):
            last_newline = chunk_text.rfind("\n")
            if last_newline > CHUNK_SIZE // 2:
                chunk_text = chunk_text[:last_newline]
                end = start + last_newline

        if chunk_text.strip():
            chunks.append(_make_chunk(chunk_text.strip(), file_path, repo, branch, language, chunk_index))

        start = end - CHUNK_OVERLAP
        chunk_index += 1

    return chunks


def _make_chunk(content: str, file_path: str, repo: str, branch: str, language: str, position: int) -> dict:
    """Crea un chunk con toda la metadata necesaria para la búsqueda."""
    return {
        "text": content,
        "metadata": {
            "repo": repo,
            "branch": branch,
            "file_path": file_path,
            "language": language,
            "position": position,
            # Texto enriquecido para el embedding: incluye repo + rama + ruta para dar contexto
            "embed_text": f"Repository: {repo}\nBranch: {branch}\nFile: {file_path}\n\n{content}",
        },
    }


def _detect_language(ext: str) -> str:
    mapping = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".java": "java",
        ".cs": "csharp",
        ".go": "go",
        ".rb": "ruby",
        ".php": "php",
        ".sql": "sql",
        ".graphql": "graphql",
        ".proto": "protobuf",
        ".md": "markdown",
        ".txt": "text",
    }
    return mapping.get(ext, "text")
