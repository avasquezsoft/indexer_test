import os
import re

# Tamaño máximo de un chunk en caracteres.
# Si un archivo entero cabe aquí, NUNCA se divide. Va TODO junto en un solo chunk.
CHUNK_SIZE = 8000
# Overlap entre chunks para no perder contexto
CHUNK_OVERLAP = 400


def chunk_file(content: str, file_path: str, repo: str, branch: str = "HEAD") -> list[dict]:
    """
    Divide un archivo en chunks con metadata.
    REGLA DE ORO: si el archivo entero cabe en CHUNK_SIZE, va en UN SOLO CHUNK.
    Nunca partimos un archivo pequeño por "métodos" o "sentencias".
    """
    ext = os.path.splitext(file_path)[1].lower()
    language = _detect_language(ext)

    # REGLA DE ORO: archivo pequeño → UN SOLO CHUNK con TODO el contenido
    if len(content) <= CHUNK_SIZE:
        return [_make_chunk(content, file_path, repo, branch, language, 0)]

    # Solo archivos REALMENTE grandes se dividen por bloques lógicos
    chunks = _split_by_logical_blocks(content, file_path, repo, branch, language)
    if chunks:
        return chunks

    # Fallback: dividir por tamaño con overlap (solo para archivos enormes sin estructura)
    return _split_by_size(content, file_path, repo, branch, language)


def _split_by_logical_blocks(content: str, file_path: str, repo: str, branch: str, language: str) -> list[dict] | None:
    """Intenta dividir el archivo por bloques lógicos según su lenguaje."""

    if language == "sql":
        return _split_by_sql_statements(content, file_path, repo, branch)

    if language in ("html", "xml", "jsp"):
        return _split_by_tags(content, file_path, repo, branch, language)

    if language == "json":
        return _split_by_json_objects(content, file_path, repo, branch)

    if language == "java":
        return _split_by_java_members(content, file_path, repo, branch)

    if language in ("python", "javascript", "typescript", "csharp", "go"):
        return _split_by_functions(content, file_path, repo, branch, language)

    return None


def _split_by_sql_statements(content: str, file_path: str, repo: str, branch: str) -> list[dict]:
    """
    Divide SQL por sentencias (terminadas en ;).
    Respeta bloques PL/SQL (BEGIN...END) y no corta dentro de strings ni comentarios.
    """
    # Palabras que inician una sentencia
    stmt_start_re = re.compile(
        r"^\s*(CREATE|ALTER|DROP|INSERT|UPDATE|DELETE|SELECT|WITH|BEGIN|DECLARE|MERGE|SET|GRANT|REVOKE|EXEC|CALL|COMMENT|TRUNCATE|ANALYZE|EXPLAIN|DESCRIBE|SHOW)",
        re.IGNORECASE,
    )

    lines = content.split("\n")
    split_indices = [0]

    in_plsql_block = 0
    in_string = False
    string_char = None
    in_line_comment = False
    in_block_comment = False

    for i, line in enumerate(lines):
        stripped = line.strip().upper()

        # Contar BEGIN/END solo fuera de strings y comentarios
        if not in_string and not in_line_comment and not in_block_comment:
            if stripped.startswith("BEGIN"):
                in_plsql_block += 1
            elif stripped == "END" or stripped.startswith("END;") or stripped.startswith("END "):
                in_plsql_block = max(0, in_plsql_block - 1)

        # Detectar fin de sentencia por ; al final de línea (fuera de strings/comentarios)
        # Hacemos un scan rápido de la línea para ver si hay un ; "real"
        has_real_semicolon = False
        j = 0
        while j < len(line):
            ch = line[j]
            if in_block_comment:
                if ch == "*" and j + 1 < len(line) and line[j + 1] == "/":
                    in_block_comment = False
                    j += 1
            elif in_line_comment:
                pass  # hasta fin de línea
            elif in_string:
                if ch == "\\" and j + 1 < len(line):
                    j += 1
                elif ch == string_char:
                    in_string = False
                    string_char = None
            else:
                if ch == "'" or ch == '"':
                    in_string = True
                    string_char = ch
                elif ch == "-" and j + 1 < len(line) and line[j + 1] == "-":
                    in_line_comment = True
                elif ch == "/" and j + 1 < len(line) and line[j + 1] == "*":
                    in_block_comment = True
                    j += 1
                elif ch == ";":
                    has_real_semicolon = True
            j += 1

        in_line_comment = False  # resetea por línea

        if i > 0 and in_plsql_block == 0 and has_real_semicolon:
            split_indices.append(i + 1)  # cortar DESPUÉS de esta línea

    if len(split_indices) <= 1:
        return _split_by_size(content, file_path, repo, branch, "sql")

    return _build_chunks_from_indices(lines, split_indices, file_path, repo, branch, "sql")


def _split_by_tags(content: str, file_path: str, repo: str, branch: str, language: str) -> list[dict]:
    """Divide HTML/XML/JSP por etiquetas de cierre de bloque."""
    lines = content.split("\n")
    split_indices = [0]
    block_end_pattern = re.compile(
        r"^\s*</(div|section|article|table|tr|form|body|html|head|component|template|mapper|beans|bean|servlet|filter|script|style)\s*>",
        re.IGNORECASE,
    )

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
            stripped = line.strip()
            if stripped.endswith(",") or stripped.endswith("}") or stripped.endswith("]"):
                split_indices.append(i + 1)

    if len(split_indices) <= 1:
        return _split_by_size(content, file_path, repo, branch, "json")

    return _build_chunks_from_indices(lines, split_indices, file_path, repo, branch, "json")


def _split_by_java_members(content: str, file_path: str, repo: str, branch: str) -> list[dict]:
    """
    Divide archivos Java por miembros (métodos, campos, clases, interfaces).
    Asegura que las anotaciones (@Query, @Autowired, etc.) queden en el mismo chunk
    que el miembro que las sigue.
    """
    lines = content.split("\n")
    split_indices = [0]
    pending_annotation_idx = None

    # Palabras que indican inicio de declaración de miembro/clase
    member_start_re = re.compile(
        r"^\s*(public|private|protected|static|final|abstract|synchronized|native|strictfp|default|volatile|transient|class|interface|enum|record|import|package)\b"
    )

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Saltar líneas vacías y comentarios sueltos
        if not stripped or stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
            continue

        # Acumular anotaciones
        if stripped.startswith("@"):
            if pending_annotation_idx is None:
                pending_annotation_idx = i
            continue

        # Detectar inicio de miembro/clase
        if member_start_re.match(line):
            start_idx = pending_annotation_idx if pending_annotation_idx is not None else i
            if start_idx > 0 and start_idx not in split_indices:
                split_indices.append(start_idx)
            pending_annotation_idx = None
        else:
            # Si había anotaciones sueltas y no van seguidas de miembro, descartarlas
            pending_annotation_idx = None

    if len(split_indices) <= 1:
        return _split_by_size(content, file_path, repo, branch, "java")

    return _build_chunks_from_indices(lines, split_indices, file_path, repo, branch, "java")


def _split_by_functions(content: str, file_path: str, repo: str, branch: str, language: str) -> list[dict]:
    """Divide buscando definiciones de funciones/clases."""
    patterns = {
        "python": r"^(class |def |\s{0,4}def |\s{0,4}async def )",
        "javascript": r"^(function |const \w+ = |export (default |async )?function |class )",
        "typescript": r"^(function |const \w+ = |export (default |async )?function |class |interface |type \w+ =)",
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
    """Construye chunks a partir de índices de línea.
    Si un bloque excede CHUNK_SIZE * 2, se subdivide PERO cada sub-chunk conserva
    las primeras líneas del bloque (firma/encabezado) para no perder contexto."""
    chunks = []
    for idx, start in enumerate(split_indices):
        end = split_indices[idx + 1] if idx + 1 < len(split_indices) else len(lines)
        block = "\n".join(lines[start:end]).strip()

        if not block:
            continue

        if len(block) <= CHUNK_SIZE:
            chunks.append(_make_chunk(block, file_path, repo, branch, language, start))
        elif len(block) <= CHUNK_SIZE * 2:
            # Bloque mediano: un solo chunk aunque sea un poco grande
            chunks.append(_make_chunk(block, file_path, repo, branch, language, start))
        else:
            # Bloque muy grande: subdividir con header de contexto
            header_lines = lines[start : min(start + 5, end)]
            header = "\n".join(header_lines).strip()
            sub = _split_with_header(block, header, file_path, repo, branch, language, start)
            chunks.extend(sub)

    return chunks


def _split_with_header(content: str, header: str, file_path: str, repo: str, branch: str, language: str, base_position: int) -> list[dict]:
    """Divide un bloque grande en sub-chunks, añadiendo un header de contexto a cada uno."""
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(content):
        end = start + CHUNK_SIZE - len(header) - 50  # reservar espacio para header
        if end <= start:
            end = start + CHUNK_SIZE

        chunk_text = content[start:end]

        if end < len(content):
            best_cut = _find_best_cut(chunk_text, language)
            if best_cut > CHUNK_SIZE // 3:
                chunk_text = chunk_text[:best_cut]
                end = start + best_cut

        full_text = f"{header}\n...\n{chunk_text.strip()}"
        if full_text.strip():
            chunks.append(_make_chunk(full_text.strip(), file_path, repo, branch, language, base_position + chunk_index))

        start = end - CHUNK_OVERLAP
        chunk_index += 1

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
    # Extraer nombre de clase/archivo sin extensión para enriquecer el embedding
    basename = os.path.splitext(os.path.basename(file_path))[0]
    # Palabras clave del path (carpetas) para contexto semántico adicional
    path_keywords = " ".join(
        os.path.dirname(file_path).replace("/", " ").replace("-", " ").replace("_", " ").split()
    )

    # Construir contexto semántico adicional según el tipo de archivo
    extra_context = ""
    if language == "java":
        extra_context = f"Java class {basename}. Implementation. Business logic. "
    elif language == "sql":
        extra_context = f"SQL query {basename}. Database query. SELECT INSERT UPDATE DELETE. "
    elif language in ("html", "jsp", "vue", "xml"):
        extra_context = f"View template {basename}. Frontend UI. "

    embed_text = (
        f"Repository: {repo}\n"
        f"Branch: {branch}\n"
        f"File: {file_path}\n"
        f"Module context: {path_keywords}\n"
        f"Language: {language}\n"
        f"{extra_context}\n"
        f"{content}"
    )

    return {
        "text": content,
        "metadata": {
            "repo": repo,
            "branch": branch,
            "file_path": file_path,
            "language": language,
            "position": position,
            "embed_text": embed_text,
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
