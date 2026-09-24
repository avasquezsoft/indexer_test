"""
code_links.py — Relaciones que el parser AST no ve: SQL, tablas y endpoints HTTP.

- Qué tablas lee/escribe un archivo .sql o un SQL escrito en strings del código.
- Qué métodos usan cada archivo .sql (directo o vía una constante).
- Qué ruta HTTP expone cada método de un controller (Spring MVC y JAX-RS).

ponytail: todo con expresiones regulares, sin parser SQL ni resolución de
símbolos. Cubre el SQL y las anotaciones habituales; SQL dinámico armado en
tiempo de ejecución o rutas en constantes no se detectan.
"""

import os
import re
from collections.abc import Callable

from ast_parser import GraphEntity, Relation

# Referencia a un archivo .sql dentro de un string: "sql/SELECT_X.sql"
SQL_REF_RE = re.compile(r'["\']([^"\']*\.sql)["\']', re.IGNORECASE)

_MAX_SQL_NODE_CHARS = 20000

# ═══════════════════════════════════════════════════════════════
# Tablas en SQL
# ═══════════════════════════════════════════════════════════════

_SQL_COMMENT_RE = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)
_SQL_LITERAL_RE = re.compile(r"'(?:[^']|'')*'")
_NAME = r"([\[\]\"`\w$#.]+)"
_WRITE_RE = re.compile(
    r"\b(?:INSERT\s+INTO|UPDATE|DELETE\s+FROM|MERGE\s+INTO|TRUNCATE\s+TABLE)\s+" + _NAME,
    re.IGNORECASE,
)
_READ_RE = re.compile(r"\b(?:FROM|JOIN|USING)\s+" + _NAME, re.IGNORECASE)
_CTE_RE = re.compile(r"(?:\bWITH|,)\s*(\w+)\s+AS\s*\(", re.IGNORECASE)
_NOT_TABLES = {
    "SELECT", "DUAL", "TABLE", "LATERAL", "UNNEST", "VALUES", "SET", "WHERE",
    "ON", "OF", "NOWAIT", "SKIP", "STATISTICS",
}


def _table_name(raw: str) -> str | None:
    """'[dbo].[FACTURA]' -> 'FACTURA'. None si no parece una tabla."""
    name = raw.split(".")[-1].strip('[]"`')
    if not re.fullmatch(r"[A-Za-z_][\w$#]*", name) or name.upper() in _NOT_TABLES:
        return None
    return name.upper()


def sql_tables(sql: str) -> tuple[set[str], set[str]]:
    """Devuelve (tablas_leidas, tablas_escritas) de un texto SQL."""
    text = _SQL_LITERAL_RE.sub("''", _SQL_COMMENT_RE.sub(" ", sql))
    # "SELECT ... FOR UPDATE" y "ON DUPLICATE KEY UPDATE" no escriben en otra tabla
    text = re.sub(r"\b(?:FOR|KEY)\s+UPDATE\b", " ", text, flags=re.IGNORECASE)
    ctes = {c.upper() for c in _CTE_RE.findall(text)}

    writes = {t for raw in _WRITE_RE.findall(text) if (t := _table_name(raw)) and t not in ctes}
    # DELETE FROM x ya es escritura: que el FROM no lo cuente como lectura
    text = re.sub(r"\bDELETE\s+FROM\b", "DELETE ", text, flags=re.IGNORECASE)
    reads = {t for raw in _READ_RE.findall(text) if (t := _table_name(raw)) and t not in ctes}
    return reads, writes


_JAVA_STRING_RE = re.compile(r'"((?:[^"\\\n]|\\.)*)"')
# Un string es SQL si empieza con una palabra SQL, o trae palabras clave en mayúsculas.
# Así "Error from server" (un log) no cuenta, pero "SELECT a " + "FROM T" sí.
_SQLISH_START_RE = re.compile(
    r"^\s*(?:select|insert|update|delete|merge|with|from|where|join|inner|left|right|"
    r"and|or|order|group|set|values|into)\b",
    re.IGNORECASE,
)
_SQL_UPPER_RE = re.compile(r"\b(?:SELECT|FROM|WHERE|JOIN|INSERT|UPDATE|DELETE|VALUES)\b")
_SQL_STATEMENT_RE = re.compile(
    r"\b(?:SELECT\b.+?\bFROM|INSERT\s+INTO|UPDATE\s+\S+\s+SET|DELETE\s+FROM|MERGE\s+INTO)\b",
    re.IGNORECASE | re.DOTALL,
)


def code_sql_tables(code: str) -> tuple[set[str], set[str]]:
    """Tablas del SQL escrito en strings del código (JDBC, @Query, JPQL...)."""
    literals = [s for s in _JAVA_STRING_RE.findall(code) if _SQLISH_START_RE.match(s) or _SQL_UPPER_RE.search(s)]
    joined = " ".join(literals)
    if not _SQL_STATEMENT_RE.search(joined):
        return set(), set()
    return sql_tables(joined)


# ═══════════════════════════════════════════════════════════════
# Entidades SQL / tablas
# ═══════════════════════════════════════════════════════════════

def _table_relations(reads: set[str], writes: set[str]) -> list[Relation]:
    return [Relation(type="READS", target_name=t, target_type="Table") for t in sorted(reads)] + [
        Relation(type="WRITES", target_name=t, target_type="Table") for t in sorted(writes)
    ]


def sql_file_entity(path: str, content: str, repo: str, branch: str) -> tuple[GraphEntity, set[str]]:
    """Nodo SqlFile de un archivo .sql con sus lecturas/escrituras de tablas."""
    reads, writes = sql_tables(content)
    code = content
    if len(code) > _MAX_SQL_NODE_CHARS:
        code = code[:_MAX_SQL_NODE_CHARS] + f"\n-- ... ({len(content)} chars totales) ..."
    entity = GraphEntity(
        type="SqlFile",
        name=os.path.basename(path),
        file_path=path,
        repo=repo,
        branch=branch,
        language="sql",
        start_line=1,
        end_line=content.count("\n") + 1,
        code=code,
        relations=_table_relations(reads, writes),
    )
    return entity, reads | writes


def table_entities(names: set[str], repo: str, branch: str) -> list[GraphEntity]:
    """Un nodo Table por tabla referenciada (destino de READS/WRITES)."""
    return [
        GraphEntity(
            type="Table", name=name, file_path="", repo=repo, branch=branch,
            language="sql", start_line=0, end_line=0, code="",
        )
        for name in sorted(names)
    ]


def add_sql_links(entities: list[GraphEntity], resolve: Callable[[str, str], str | None]) -> set[str]:
    """
    Agrega a las entidades de un archivo:
    - USES_SQL: método → archivo .sql que referencia (directo o vía constante).
    - READS/WRITES: método → tablas del SQL escrito en sus strings.
    resolve(ref, file_path) devuelve la ruta del .sql en el repo o None.
    Devuelve los nombres de tablas referenciadas.
    """
    members = [e for e in entities if e.type in ("Method", "Function", "Field")]
    refs = {
        id(e): {path for ref in SQL_REF_RE.findall(e.code) if (path := resolve(ref, e.file_path))}
        for e in members
    }

    # private static final String Q = "sql/X.sql";  →  los métodos que usan Q
    for field in (e for e in members if e.type == "Field" and refs[id(e)]):
        word = re.compile(rf"\b{re.escape(field.name)}\b")
        for method in members:
            if method.type != "Field" and method.file_path == field.file_path and word.search(method.code):
                refs[id(method)] |= refs[id(field)]

    tables: set[str] = set()
    linked: set[str] = set()
    for method in (e for e in members if e.type != "Field"):
        for path in sorted(refs[id(method)]):
            method.relations.append(
                Relation(type="USES_SQL", target_name=os.path.basename(path), target_type="SqlFile", target_path=path)
            )
            linked.add(path)
        reads, writes = code_sql_tables(method.code)
        method.relations.extend(_table_relations(reads, writes))
        tables |= reads | writes

    # .sql referenciados fuera de cualquier método (bloque static, etc.): a la clase
    for owner in (e for e in entities if e.type in ("Class", "Interface", "Enum", "Record")):
        for ref in SQL_REF_RE.findall(owner.code):
            path = resolve(ref, owner.file_path)
            if path and path not in linked:
                owner.relations.append(
                    Relation(type="USES_SQL", target_name=os.path.basename(path), target_type="SqlFile", target_path=path)
                )
                linked.add(path)
    return tables


# ═══════════════════════════════════════════════════════════════
# Receptor de las llamadas (Java)
# ═══════════════════════════════════════════════════════════════

_CALL_NAME_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
_RECEIVER_RE = re.compile(r"([A-Za-z_]\w*)\s*\.\s*$")
_NOT_CALLS = {"if", "for", "while", "switch", "catch", "return", "synchronized", "new", "super", "this", "throw"}


def _declared_type(code: str, var: str) -> str | None:
    """Tipo de una variable declarada en el código: campo, parámetro, local o for-each."""
    match = re.search(
        rf"\b([A-Z][\w.]*)(?:\s*<[^;(){{}}]*?>)?(?:\[\]|\.\.\.)*\s+{re.escape(var)}\s*[=;,:)]",
        code,
    )
    return match.group(1).split(".")[-1] if match else None


def _receivers(code: str, name: str) -> set[str | None]:
    """
    Receptores con los que el código llama a `name(`:
    "" = sin receptor (misma clase), "this"/"super", una variable/clase, o None si
    no se puede saber (ej. a().name()).
    """
    found: set[str | None] = set()
    for match in _CALL_NAME_RE.finditer(code):
        if match.group(1) != name:
            continue
        before = code[: match.start()].rstrip()
        if before.endswith("."):
            receiver = _RECEIVER_RE.search(before)
            found.add(receiver.group(1) if receiver else None)
            continue
        if before.endswith("@"):
            continue  # anotación: @Nombre(...)
        previous = re.search(r"(\w+)\W*$", before)
        last_char = before[-1:] if before else ""
        if previous and (last_char.isalnum() or last_char in "_>]"):
            # "Tipo nombre(" es una declaración y "new Nombre(" un constructor;
            # "return nombre(" o "throw nombre(" sí son llamadas.
            if previous.group(1) in ("return", "throw", "else", "case", "yield"):
                found.add("")
            continue
        found.add("")
    return found


def resolve_call_owners(entities: list[GraphEntity]) -> None:
    """
    Para cada CALLS de un método Java deduce en qué clases puede estar el método
    llamado, según el tipo del receptor. Así facturacionService.consultar() enlaza
    con FacturacionService (y sus implementaciones) y no con cualquier método del
    repo que se llame consultar.

    ponytail: inferencia por texto (campos, parámetros y locales declarados con su
    tipo). Receptores sin tipo visible (lambdas, cadenas a().b(), var) quedan en
    None y se enlazan por nombre como antes.
    """
    classes = [e for e in entities if e.type in ("Class", "Interface", "Enum", "Record")]
    fields = [e for e in entities if e.type == "Field"]

    for method in (e for e in entities if e.type == "Method"):
        owners_of_file = [
            c for c in classes
            if c.file_path == method.file_path and c.start_line <= method.start_line <= c.end_line
        ]
        if not owners_of_file:
            continue
        owner = min(owners_of_file, key=lambda c: c.end_line - c.start_line)
        own_types = [owner.name] + [r.target_name for r in owner.relations if r.type == "EXTENDS"]
        own_fields = "\n".join(
            f.code for f in fields if f.file_path == owner.file_path and owner.start_line <= f.start_line <= owner.end_line
        )

        for relation in method.relations:
            if relation.type != "CALLS":
                continue
            owners: set[str] = set()
            for receiver in _receivers(method.code, relation.target_name):
                if receiver is None:
                    owners = set()
                    break
                if receiver in ("", "this"):
                    owners.update(own_types)
                elif receiver == "super":
                    owners.update(own_types[1:])
                else:
                    declared = _declared_type(method.code, receiver) or _declared_type(own_fields, receiver)
                    if declared:
                        owners.add(declared)
                    elif receiver[:1].isupper():
                        owners.add(receiver)  # llamada estática: Clase.metodo()
                    else:
                        owners = set()
                        break
            relation.target_owners = sorted(owners) or None


# ═══════════════════════════════════════════════════════════════
# Endpoints HTTP (Spring MVC y JAX-RS)
# ═══════════════════════════════════════════════════════════════

_SPRING_RE = re.compile(r"@(Get|Post|Put|Delete|Patch|Request)Mapping\b\s*(\((?:[^()]|\([^()]*\))*\))?")
_NON_PATH_ARGS_RE = re.compile(r"\b(?:produces|consumes|headers|params|name)\s*=\s*(?:\{[^}]*\}|\"[^\"]*\"|[\w.]+)")
_REQUEST_METHOD_RE = re.compile(r"RequestMethod\.(\w+)")
_QUOTED_RE = re.compile(r'"([^"]*)"')
_JAXRS_PATH_RE = re.compile(r'@Path\s*\(\s*(?:value\s*=\s*)?"([^"]*)"')
_JAXRS_VERB_RE = re.compile(r"@(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\b")


def _mappings(header: str) -> tuple[list[str], list[str]]:
    """(verbos, rutas) declarados en las anotaciones de un encabezado."""
    verbs: list[str] = []
    paths: list[str] = []
    for kind, args in _SPRING_RE.findall(header):
        args = _NON_PATH_ARGS_RE.sub("", args or "")
        paths += _QUOTED_RE.findall(args)
        if kind == "Request":
            verbs += _REQUEST_METHOD_RE.findall(args) or ["ANY"]
        else:
            verbs.append(kind.upper())
    paths += _JAXRS_PATH_RE.findall(header)
    verbs += _JAXRS_VERB_RE.findall(header)
    return verbs, paths


_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)


def _header(code: str, declaration: str) -> str:
    """Anotaciones y modificadores antes de la declaración (sin Javadoc, que puede nombrarla)."""
    code = _BLOCK_COMMENT_RE.sub("", code)
    match = re.search(declaration, code)
    return code[: match.start()] if match else ""


def _join(*parts: str) -> str:
    return "/" + "/".join(p.strip("/") for p in parts if p.strip("/"))


def add_routes(entities: list[GraphEntity]) -> None:
    """Pone en entity.route la ruta HTTP de cada método de controller."""
    classes = []
    for cls in entities:
        if cls.type not in ("Class", "Interface"):
            continue
        _, base_paths = _mappings(_header(cls.code, rf"\b(?:class|interface)\s+{re.escape(cls.name)}\b"))
        classes.append((cls, base_paths))

    for method in entities:
        if method.type != "Method":
            continue
        verbs, paths = _mappings(_header(method.code, rf"\b{re.escape(method.name)}\s*\("))
        if not verbs:
            continue
        # Clase dueña: la más interna cuyo rango de líneas contiene al método
        owners = [
            (cls, base) for cls, base in classes
            if cls.file_path == method.file_path and cls.start_line <= method.start_line <= cls.end_line
        ]
        base_paths = min(owners, key=lambda o: o[0].end_line - o[0].start_line)[1] if owners else []
        routes = [
            f"{verb} {_join(base, path)}"
            for verb in dict.fromkeys(verbs)
            for base in (base_paths or [""])
            for path in (paths or [""])
        ]
        method.route = ", ".join(dict.fromkeys(routes))
