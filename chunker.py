import os
import re

# Tamaño máximo de un chunk en caracteres (aumentado para capturar queries SQL completas)
CHUNK_SIZE = 2500
# Overlap entre chunks para no perder contexto
CHUNK_OVERLAP = 300


def chunk_file(content: str, file_path: str, repo: str, branch: str = "HEAD") -> list[dict]:
    """
    Divide un archivo en chunks con metadata.
    Intenta dividir por bloques lógicos según el tipo de archivo.
    """
    ext = os.path.splitext(file_path)[1].lower()
    language = _detect_language(ext)

    # Para archivos pequeños, un solo chunk
    if len(content) <= CHUNK_SIZE:
        return [_make_chunk(content, file_path, repo, branch, language, 0)]

    # Dividir por bloques lógicos según el lenguaje
    chunks = _split_by_logical_blocks(content, file_path, repo, branch, language)
    if chunks:
        return chunks

    # Fallback: dividir por tamaño con overlap
    return _split_by_size(content, file_path, repo, branch, language)


def _split_by_logical_blocks(content: str, file_path: str, repo: str, branch: str, language: str) -> list[dict] | None:
    """Intenta dividir el archivo por bloques lógicos según su lenguaje."""

    if language == "sql":
        return _split_by_sql_statements(content, file_path, repo, branch)

    if language in ("html", "xml", "jsp"):
        return _split_by_tags(content, file_path, repo, branch, language)

    if language == "json":
        return _split_by_json_objects(content, file_path, repo, branch)

    if language in ("python", "javascript", "typescript", "java", "csharp", "go"):
        return _split_by_functions(content, file_path, repo, branch, language)

    return None


def _split_by_sql_statements(content: str, file_path: str, repo: str, branch: str) -> list[dict]:
    """Divide SQL por sentencias (terminadas en ;) respetando bloques PL/SQL."""
    # Regex que busca ; al final de línea, pero evita cortar dentro de BEGIN...END
    pattern = re.compile(r"^\s*(CREATE|ALTER|DROP|INSERT|UPDATE|DELETE|SELECT|WITH|BEGIN|DECLARE|MERGE)", re.IGNORECASE)
    lines = content.split("\n")
    split_indices = [0]

    in_plsql_block = 0
    for i, line in enumerate(lines):
        stripped = line.strip().upper()
        if stripped.startswith("BEGIN"):
            in_plsql_block += 1
        elif stripped == "END" or stripped.startswith("END;"):
            in_plsql_block = max(0, in_plsql_block - 1)

        if i > 0 and in_plsql_block == 0 and pattern.match(line) and ";" in line:
            split_indices.append(i)

    if len(split_indices) <= 1:
        return _split_by_size(content, file_path, repo, branch, "sql")

    return _build_chunks_from_indices(lines, split_indices, file_path, repo, branch, "sql")


def _split_by_tags(content: str, file_path: str, repo: str, branch: str, language: str) -> list[dict]:
    """Divide HTML/XML/JSP por etiquetas de cierre de bloque."""
    lines = content.split("\n")
    split_indices = [0]
    block_end_pattern = re.compile(r"^\s*</(div|section|article|table|tr|form|body|html|head|component|template|mapper|beans|bean|servlet|filter)\s*>", re.IGNORECASE)

    for i, line in enumerate(lines):
        if i > 0 and block_end_pattern.match(line):
            split_indices.append(i)

    if len(split_indices) <= 1:
        return _split_by_size(content, file_path, repo, branch, language)

    return _build_chunks_from_indices(lines, split_indices, file_path, repo, branch, language)


def _split_by_json_objects(content: str, file_path: str, repo: str, branch: str) -> list[dict]:
    """Divide JSON intentando cortar después de objetos/array cerrados."""
    lines = content.split("\n")
    split_indices = [0]
    brace_depth = 0
    bracket_depth = 0
    in_string = False
    escape = False

    for i, line in enumerate(lines):
        for ch in line:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"' and not in_string:
                in_string = True
            elif ch == '"' and in_string:
                in_string = False
            elif not in_string:
                if ch == "{":
                    brace_depth += 1
                elif ch == "}":
                    brace_depth -= 1
                elif ch == "[":
                    bracket_depth += 1
                elif ch == "]":
                    bracket_depth -= 1

        if i > 0 and brace_depth == 0 and bracket_depth == 0 and not in_string:
            # Intentar cortar después de objetos completos
            stripped = line.strip()
            if stripped.endswith(",") or stripped.endswith("}") or stripped.endswith("]"):
                split_indices.append(i + 1)

    if len(split_indices) <= 1:
        return _split_by_size(content, file_path, repo, branch, "json")

    return _build_chunks_from_indices(lines, split_indices, file_path, repo, branch, "json")


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

    return _build_chunks_from_indices(lines, split_indices, file_path, repo, branch, language)


def _build_chunks_from_indices(lines: list[str], split_indices: list[int], file_path: str, repo: str, branch: str, language: str) -> list[dict]:
    """Construye chunks a partir de índices de línea, subdividiendo bloques muy grandes."""
    chunks = []
    for idx, start in enumerate(split_indices):
        end = split_indices[idx + 1] if idx + 1 < len(split_indices) else len(lines)
        block = "\n".join(lines[start:end]).strip()

        if not block:
            continue

        # Si el bloque es muy grande, subdividirlo
        if len(block) > CHUNK_SIZE * 2:
            sub = _split_by_size(block, file_path, repo, branch, language)
            chunks.extend(sub)
        else:
            chunks.append(_make_chunk(block, file_path, repo, branch, language, start))

    return chunks


def _split_by_size(content: str, file_path: str, repo: str, branch: str, language: str) -> list[dict]:
    """Divide por tamaño con overlap, intentando cortar en límites lógicos."""
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(content):
        end = start + CHUNK_SIZE
        chunk_text = content[start:end]

        if end < len(content):
            # Estrategia: buscar el mejor punto de corte hacia atrás
            best_cut = _find_best_cut(chunk_text, language)
            if best_cut > CHUNK_SIZE // 3:
                chunk_text = chunk_text[:best_cut]
                end = start + best_cut

        if chunk_text.strip():
            chunks.append(_make_chunk(chunk_text.strip(), file_path, repo, branch, language, chunk_index))

        start = end - CHUNK_OVERLAP
        chunk_index += 1

    return chunks


def _find_best_cut(text: str, language: str) -> int:
    """Busca el mejor punto para cortar un chunk sin romper contenido importante."""
    candidates = []

    # 1. Línea vacía (mejor opción)
    for m in re.finditer(r"\n\s*\n", text):
        candidates.append((m.end(), 3))

    # 2. Fin de sentencia SQL
    if language == "sql":
        for m in re.finditer(r";\s*\n", text):
            candidates.append((m.end(), 2))

    # 3. Cierre de bloque / tag
    for m in re.finditer(r"[;}\]]\s*\n", text):
        candidates.append((m.end(), 1))

    # 4. Salto de línea simple
    for m in re.finditer(r"\n", text):
        candidates.append((m.end(), 0))

    if not candidates:
        return len(text)

    # Elegir el punto más lejano con mayor prioridad
    best = max(candidates, key=lambda x: (x[0], x[1]))
    return best[0]


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
            "embed_text": f"Repository: {repo}\nBranch: {branch}\nFile: {file_path}\nLanguage: {language}\n\n{content}",
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
        ".plsql": "sql",
        ".pks": "sql",
        ".pkb": "sql",
        ".graphql": "graphql",
        ".proto": "protobuf",
        ".md": "markdown",
        ".txt": "text",
        # Web / planillas
        ".html": "html",
        ".htm": "html",
        ".css": "css",
        ".scss": "scss",
        ".sass": "sass",
        ".less": "less",
        ".json": "json",
        ".xml": "xml",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".properties": "properties",
        ".conf": "conf",
        ".cfg": "cfg",
        ".ini": "ini",
        ".jsp": "jsp",
        ".jspf": "jsp",
        ".tag": "jsp",
        ".vue": "vue",
        ".svelte": "svelte",
        ".ftl": "freemarker",
        ".hbs": "handlebars",
        ".mustache": "mustache",
        ".twig": "twig",
        # Shell / scripts
        ".sh": "shell",
        ".bash": "shell",
        ".zsh": "shell",
        ".ps1": "powershell",
        ".bat": "batch",
        ".cmd": "batch",
        # Config
        ".dockerfile": "dockerfile",
        ".env": "env",
        ".gitignore": "gitignore",
        ".gitattributes": "gitattributes",
    }
    return mapping.get(ext, "text")
