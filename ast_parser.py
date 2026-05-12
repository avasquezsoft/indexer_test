"""
ast_parser.py — Extracción de AST y entidades de grafo desde código fuente.

Soporta Tree-sitter (multilenguaje) y JavaParser (Java/Spring Boot vía HTTP).
"""

import logging
import os
from typing import Literal

import httpx
from pydantic import BaseModel, Field

# ── Tree-sitter ──
from tree_sitter import Language, Parser
from tree_sitter_go import language as go_lang
from tree_sitter_java import language as java_lang
from tree_sitter_javascript import language as js_lang
from tree_sitter_python import language as py_lang
from tree_sitter_typescript import language_typescript

# Compatibilidad con tree-sitter 0.24+ (nueva API Language/Parser)
_ts_lang = Language(language_typescript())
_go_lang = Language(go_lang())
_java_lang = Language(java_lang())
_js_lang = Language(js_lang())
_py_lang = Language(py_lang())

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# Modelos de datos
# ═══════════════════════════════════════════════════════════════

class Relation(BaseModel):
    type: Literal[
        "EXTENDS", "IMPLEMENTS", "HAS_METHOD", "HAS_FIELD",
        "CALLS", "IMPORTS", "ANNOTATED_WITH", "INJECTED",
    ]
    target_name: str
    target_type: str = "Unknown"
    properties: dict = Field(default_factory=dict)


class GraphEntity(BaseModel):
    type: Literal["Class", "Interface", "Method", "Field", "Annotation", "Function", "Struct"]
    name: str
    file_path: str
    repo: str
    branch: str
    language: str
    start_line: int
    end_line: int
    signature: str | None = None
    docstring: str | None = None
    code: str
    annotations: list[str] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# Tree-sitter: parsers por lenguaje
# ═══════════════════════════════════════════════════════════════

_LANGUAGE_MAP = {
    "java": _java_lang,
    "python": _py_lang,
    "javascript": _js_lang,
    "typescript": _ts_lang,
    "go": _go_lang,
}

_PARSERS: dict[str, Parser] = {}


def _get_parser(language: str) -> Parser | None:
    if language not in _PARSERS:
        lang_obj = _LANGUAGE_MAP.get(language)
        if lang_obj is None:
            return None
        parser = Parser(lang_obj)
        _PARSERS[language] = parser
    return _PARSERS[language]


# ═══════════════════════════════════════════════════════════════
# JavaParser client
# ═══════════════════════════════════════════════════════════════

_JAVAPARSER_URL = os.environ.get("JAVAPARSER_URL", "http://javaparser:8080")
_JAVAPARSER_TIMEOUT = 30.0


def _parse_java_with_javaparser(source: str, file_path: str) -> list[GraphEntity] | None:
    """Delega al microservicio JavaParser. Devuelve None si falla (fallback a Tree-sitter)."""
    try:
        resp = httpx.post(
            f"{_JAVAPARSER_URL}/parse",
            json={"source": source, "file_path": file_path},
            timeout=_JAVAPARSER_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        entities = []
        for ent in data.get("entities", []):
            entities.append(GraphEntity(
                type=ent.get("type", "Class"),
                name=ent.get("name", ""),
                file_path=ent.get("file_path", file_path),
                repo="",  # se rellena después
                branch="",
                language="java",
                start_line=ent.get("start_line", 0),
                end_line=ent.get("end_line", 0),
                signature=ent.get("signature"),
                docstring=ent.get("docstring") or None,
                code=ent.get("code", ""),
                annotations=ent.get("annotations", []),
                relations=[Relation(**r) for r in ent.get("relations", [])],
            ))
        return entities
    except Exception as exc:
        logger.warning("JavaParser service no disponible, usando Tree-sitter para Java: %s", exc)
        return None


# ═══════════════════════════════════════════════════════════════
# Tree-sitter: utilidades
# ═══════════════════════════════════════════════════════════════

def _node_text(source: str, node) -> str:
    return source[node.start_byte:node.end_byte]


def _child_by_type(node, node_type: str):
    for child in node.children:
        if child.type == node_type:
            return child
    return None


def _children_by_type(node, node_type: str):
    return [c for c in node.children if c.type == node_type]


def _extract_docstring(source: str, node) -> str | None:
    """Extrae comentario que precede a un nodo (docstring/Javadoc aproximado)."""
    # Tree-sitter no asocia comentarios directamente; buscamos líneas justo antes
    start_line = node.start_point[0]
    lines = source.splitlines()
    comments = []
    for i in range(start_line - 1, max(start_line - 10, -1), -1):
        line = lines[i].strip()
        if line.startswith("//") or line.startswith("#") or line.startswith("--"):
            comments.insert(0, line.lstrip("/#-").strip())
        elif line.startswith("/*") or line.startswith("*"):
            comments.insert(0, line.lstrip("/*").strip())
        elif line == "" or line.startswith("@"):
            continue
        else:
            break
    return "\n".join(comments) if comments else None


# ═══════════════════════════════════════════════════════════════
# Tree-sitter: extractores por lenguaje
# ═══════════════════════════════════════════════════════════════

def _extract_python(source: str, tree, repo: str, branch: str, file_path: str) -> list[GraphEntity]:
    entities: list[GraphEntity] = []
    root = tree.root_node

    def walk(node, parent_name: str = ""):
        if node.type in ("class_definition",):
            name_node = _child_by_type(node, "identifier")
            name = _node_text(source, name_node) if name_node else ""
            body = _child_by_type(node, "block")
            sig = _node_text(source, node).split(":")[0].strip()
            cls_ent = GraphEntity(
                type="Class",
                name=name,
                file_path=file_path,
                repo=repo,
                branch=branch,
                language="python",
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                signature=sig,
                docstring=_extract_docstring(source, node),
                code=_node_text(source, node),
            )
            # Herencia
            for arg in _children_by_type(node, "argument_list"):
                for child in arg.children:
                    if child.type == "identifier":
                        cls_ent.relations.append(Relation(
                            type="EXTENDS", target_name=_node_text(source, child), target_type="Class"
                        ))
            # Decoradores (anotaciones)
            for dec in _children_by_type(node, "decorator"):
                cls_ent.annotations.append(_node_text(source, dec).lstrip("@").split("(")[0])

            # Métodos y campos dentro de la clase
            if body:
                for child in body.children:
                    if child.type == "function_definition":
                        mname_node = _child_by_type(child, "identifier")
                        mname = _node_text(source, mname_node) if mname_node else ""
                        msig = _node_text(source, child).split(":")[0].strip()
                        ment = GraphEntity(
                            type="Method",
                            name=mname,
                            file_path=file_path,
                            repo=repo,
                            branch=branch,
                            language="python",
                            start_line=child.start_point[0] + 1,
                            end_line=child.end_point[0] + 1,
                            signature=msig,
                            docstring=_extract_docstring(source, child),
                            code=_node_text(source, child),
                        )
                        # Llamadas dentro del método
                        for call in child.children:
                            if call.type == "call":
                                func = _child_by_type(call, "identifier")
                                if func:
                                    ment.relations.append(Relation(
                                        type="CALLS", target_name=_node_text(source, func), target_type="Method"
                                    ))
                        entities.append(ment)
                        cls_ent.relations.append(Relation(
                            type="HAS_METHOD", target_name=mname, target_type="Method"
                        ))
                    elif child.type == "expression_statement":
                        # Posible campo de clase (self.x = ...)
                        assign = _child_by_type(child, "assignment")
                        if assign:
                            left = _child_by_type(assign, "attribute")
                            if left:
                                attr = _child_by_type(left, "identifier")
                                if attr:
                                    fname = _node_text(source, attr)
                                    fent = GraphEntity(
                                        type="Field",
                                        name=fname,
                                        file_path=file_path,
                                        repo=repo,
                                        branch=branch,
                                        language="python",
                                        start_line=child.start_point[0] + 1,
                                        end_line=child.end_point[0] + 1,
                                        code=_node_text(source, child),
                                    )
                                    entities.append(fent)
                                    cls_ent.relations.append(Relation(
                                        type="HAS_FIELD", target_name=fname, target_type="Field"
                                    ))
            entities.append(cls_ent)
            if body:
                for child in body.children:
                    walk(child, name)

        elif node.type == "function_definition" and not parent_name:
            # Función a nivel de módulo
            name_node = _child_by_type(node, "identifier")
            name = _node_text(source, name_node) if name_node else ""
            sig = _node_text(source, node).split(":")[0].strip()
            fent = GraphEntity(
                type="Function",
                name=name,
                file_path=file_path,
                repo=repo,
                branch=branch,
                language="python",
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                signature=sig,
                docstring=_extract_docstring(source, node),
                code=_node_text(source, node),
            )
            for call in node.children:
                if call.type == "call":
                    func = _child_by_type(call, "identifier")
                    if func:
                        fent.relations.append(Relation(
                            type="CALLS", target_name=_node_text(source, func), target_type="Function"
                        ))
            entities.append(fent)

        else:
            for child in node.children:
                walk(child, parent_name)

    walk(root)
    return entities


def _extract_java_treesitter(source: str, tree, repo: str, branch: str, file_path: str) -> list[GraphEntity]:
    """Extractor Tree-sitter para Java (fallback cuando JavaParser no está)."""
    entities: list[GraphEntity] = []
    root = tree.root_node

    def walk(node, parent_name: str = ""):
        if node.type in ("class_declaration", "interface_declaration"):
            is_interface = node.type == "interface_declaration"
            name_node = _child_by_type(node, "identifier")
            name = _node_text(source, name_node) if name_node else ""
            body = _child_by_type(node, "class_body") or _child_by_type(node, "interface_body")
            sig = _node_text(source, node).split("{")[0].strip()
            cls_ent = GraphEntity(
                type="Interface" if is_interface else "Class",
                name=name,
                file_path=file_path,
                repo=repo,
                branch=branch,
                language="java",
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                signature=sig,
                docstring=_extract_docstring(source, node),
                code=_node_text(source, node),
            )
            # Anotaciones
            for ann in _children_by_type(node, "annotation"):
                cls_ent.annotations.append(_node_text(source, ann).lstrip("@").split("(")[0])

            # Extends / implements
            supers = _child_by_type(node, "superclass") or _child_by_type(node, "extends_interfaces")
            if supers:
                for child in supers.children:
                    if child.type == "type_identifier":
                        cls_ent.relations.append(Relation(
                            type="EXTENDS" if is_interface else "EXTENDS",
                            target_name=_node_text(source, child),
                            target_type="Class" if not is_interface else "Interface",
                        ))
            if not is_interface:
                impls = _child_by_type(node, "super_interfaces")
                if impls:
                    for child in impls.children:
                        if child.type == "type_identifier":
                            cls_ent.relations.append(Relation(
                                type="IMPLEMENTS", target_name=_node_text(source, child), target_type="Interface"
                            ))

            # Miembros
            if body:
                for child in body.children:
                    if child.type == "method_declaration":
                        mname_node = _child_by_type(child, "identifier")
                        mname = _node_text(source, mname_node) if mname_node else ""
                        msig = _node_text(source, child).split("{")[0].strip()
                        ment = GraphEntity(
                            type="Method",
                            name=mname,
                            file_path=file_path,
                            repo=repo,
                            branch=branch,
                            language="java",
                            start_line=child.start_point[0] + 1,
                            end_line=child.end_point[0] + 1,
                            signature=msig,
                            docstring=_extract_docstring(source, child),
                            code=_node_text(source, child),
                        )
                        # Llamadas
                        for call in child.children:
                            if call.type == "method_invocation":
                                func = _child_by_type(call, "identifier")
                                if func:
                                    ment.relations.append(Relation(
                                        type="CALLS", target_name=_node_text(source, func), target_type="Method"
                                    ))
                        entities.append(ment)
                        cls_ent.relations.append(Relation(
                            type="HAS_METHOD", target_name=mname, target_type="Method"
                        ))
                    elif child.type == "field_declaration":
                        for decl in _children_by_type(child, "variable_declarator"):
                            fname_node = _child_by_type(decl, "identifier")
                            fname = _node_text(source, fname_node) if fname_node else ""
                            fent = GraphEntity(
                                type="Field",
                                name=fname,
                                file_path=file_path,
                                repo=repo,
                                branch=branch,
                                language="java",
                                start_line=child.start_point[0] + 1,
                                end_line=child.end_point[0] + 1,
                                code=_node_text(source, child),
                            )
                            entities.append(fent)
                            cls_ent.relations.append(Relation(
                                type="HAS_FIELD", target_name=fname, target_type="Field"
                            ))
            entities.append(cls_ent)
            if body:
                for child in body.children:
                    walk(child, name)
        else:
            for child in node.children:
                walk(child, parent_name)

    walk(root)
    return entities


def _extract_js_ts(source: str, tree, repo: str, branch: str, file_path: str, language: str) -> list[GraphEntity]:
    entities: list[GraphEntity] = []
    root = tree.root_node

    def walk(node, parent_name: str = ""):
        if node.type in ("class_declaration", "class"):
            name_node = _child_by_type(node, "identifier") or _child_by_type(node, "type_identifier")
            name = _node_text(source, name_node) if name_node else ""
            body = _child_by_type(node, "class_body") or _child_by_type(node, "statement_block")
            sig = _node_text(source, node).split("{")[0].strip()
            cls_ent = GraphEntity(
                type="Class",
                name=name,
                file_path=file_path,
                repo=repo,
                branch=branch,
                language=language,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                signature=sig,
                docstring=_extract_docstring(source, node),
                code=_node_text(source, node),
            )
            # extends
            for ext in _children_by_type(node, "extends_clause"):
                for child in ext.children:
                    if child.type in ("identifier", "type_identifier"):
                        cls_ent.relations.append(Relation(
                            type="EXTENDS", target_name=_node_text(source, child), target_type="Class"
                        ))
            if body:
                for child in body.children:
                    if child.type in ("method_definition", "function_declaration"):
                        mname_node = _child_by_type(child, "identifier") or _child_by_type(child, "property_identifier")
                        mname = _node_text(source, mname_node) if mname_node else ""
                        msig = _node_text(source, child).split("{")[0].strip()
                        ment = GraphEntity(
                            type="Method",
                            name=mname,
                            file_path=file_path,
                            repo=repo,
                            branch=branch,
                            language=language,
                            start_line=child.start_point[0] + 1,
                            end_line=child.end_point[0] + 1,
                            signature=msig,
                            docstring=_extract_docstring(source, child),
                            code=_node_text(source, child),
                        )
                        for call in child.children:
                            if call.type == "call_expression":
                                func = _child_by_type(call, "identifier")
                                if func:
                                    ment.relations.append(Relation(
                                        type="CALLS", target_name=_node_text(source, func), target_type="Method"
                                    ))
                        entities.append(ment)
                        cls_ent.relations.append(Relation(
                            type="HAS_METHOD", target_name=mname, target_type="Method"
                        ))
            entities.append(cls_ent)
            if body:
                for child in body.children:
                    walk(child, name)
        elif node.type in ("function_declaration", "function") and not parent_name:
            name_node = _child_by_type(node, "identifier")
            name = _node_text(source, name_node) if name_node else ""
            sig = _node_text(source, node).split("{")[0].strip()
            fent = GraphEntity(
                type="Function",
                name=name,
                file_path=file_path,
                repo=repo,
                branch=branch,
                language=language,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                signature=sig,
                docstring=_extract_docstring(source, node),
                code=_node_text(source, node),
            )
            for call in node.children:
                if call.type == "call_expression":
                    func = _child_by_type(call, "identifier")
                    if func:
                        fent.relations.append(Relation(
                            type="CALLS", target_name=_node_text(source, func), target_type="Function"
                        ))
            entities.append(fent)
        else:
            for child in node.children:
                walk(child, parent_name)

    walk(root)
    return entities


def _extract_go(source: str, tree, repo: str, branch: str, file_path: str) -> list[GraphEntity]:
    entities: list[GraphEntity] = []
    root = tree.root_node

    def walk(node, parent_name: str = ""):
        if node.type == "type_declaration":
            for spec in _children_by_type(node, "type_spec"):
                name_node = _child_by_type(spec, "type_identifier")
                name = _node_text(source, name_node) if name_node else ""
                struct_type = _child_by_type(spec, "struct_type")
                interface_type = _child_by_type(spec, "interface_type")
                is_interface = interface_type is not None
                code = _node_text(source, spec)
                ent = GraphEntity(
                    type="Interface" if is_interface else "Struct",
                    name=name,
                    file_path=file_path,
                    repo=repo,
                    branch=branch,
                    language="go",
                    start_line=spec.start_point[0] + 1,
                    end_line=spec.end_point[0] + 1,
                    code=code,
                )
                if is_interface:
                    for method in _children_by_type(interface_type, "method_spec"):
                        mname_node = _child_by_type(method, "field_identifier")
                        mname = _node_text(source, mname_node) if mname_node else ""
                        ment = GraphEntity(
                            type="Method",
                            name=mname,
                            file_path=file_path,
                            repo=repo,
                            branch=branch,
                            language="go",
                            start_line=method.start_point[0] + 1,
                            end_line=method.end_point[0] + 1,
                            code=_node_text(source, method),
                        )
                        entities.append(ment)
                        ent.relations.append(Relation(
                            type="HAS_METHOD", target_name=mname, target_type="Method"
                        ))
                entities.append(ent)
        elif node.type == "function_declaration":
            name_node = _child_by_type(node, "identifier")
            name = _node_text(source, name_node) if name_node else ""
            sig = _node_text(source, node).split("{")[0].strip()
            fent = GraphEntity(
                type="Function",
                name=name,
                file_path=file_path,
                repo=repo,
                branch=branch,
                language="go",
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                signature=sig,
                code=_node_text(source, node),
            )
            for call in node.children:
                if call.type == "call_expression":
                    func = _child_by_type(call, "identifier")
                    if func:
                        fent.relations.append(Relation(
                            type="CALLS", target_name=_node_text(source, func), target_type="Function"
                        ))
            entities.append(fent)
        else:
            for child in node.children:
                walk(child, parent_name)

    walk(root)
    return entities


# ═══════════════════════════════════════════════════════════════
# Punto de entrada público
# ═══════════════════════════════════════════════════════════════

_LANGUAGE_EXTRACTOR = {
    "python": _extract_python,
    "java": _extract_java_treesitter,
    "javascript": lambda s, t, r, b, fp: _extract_js_ts(s, t, r, b, fp, "javascript"),
    "typescript": lambda s, t, r, b, fp: _extract_js_ts(s, t, r, b, fp, "typescript"),
    "go": _extract_go,
}


def parse_file(source: str, language: str, repo: str, branch: str, file_path: str) -> list[GraphEntity]:
    """
    Extrae entidades de grafo desde código fuente.

    Para Java intenta JavaParser primero; si falla, usa Tree-sitter.
    Para otros lenguajes usa Tree-sitter directamente.
    """
    if language == "java":
        javaparser_result = _parse_java_with_javaparser(source, file_path)
        if javaparser_result is not None:
            # Rellenar repo/branch que JavaParser no conoce
            for ent in javaparser_result:
                ent.repo = repo
                ent.branch = branch
            return javaparser_result
        logger.info("Usando Tree-sitter fallback para Java: %s", file_path)

    parser = _get_parser(language)
    if parser is None:
        logger.warning("No hay parser para lenguaje: %s", language)
        return []

    tree = parser.parse(bytes(source, "utf8", errors="replace"))
    extractor = _LANGUAGE_EXTRACTOR.get(language)
    if extractor is None:
        logger.warning("No hay extractor para lenguaje: %s", language)
        return []

    return extractor(source, tree, repo, branch, file_path)


def parse_file_to_chunks_and_entities(source: str, language: str, repo: str, branch: str, file_path: str) -> tuple[list[GraphEntity], list[dict]]:
    """
    Devuelve (entidades_de_grafo, chunks_para_embedding).

    Cada entidad principal (clase, método, función) genera un chunk.
    """
    entities = parse_file(source, language, repo, branch, file_path)
    chunks = []
    for ent in entities:
        chunks.append({
            "text": ent.code,
            "metadata": {
                "repo": ent.repo,
                "branch": ent.branch,
                "file_path": ent.file_path,
                "language": ent.language,
                "position": ent.start_line,
                "embed_text": _build_embed_text(ent),
                "entity_id": _make_entity_id(ent),
                "ast_type": ent.type,
                "ast_name": ent.name,
                "ast_signature": ent.signature,
            },
        })
    return entities, chunks


def _make_entity_id(ent: GraphEntity) -> str:
    return f"{ent.repo}:{ent.branch}:{ent.file_path}:{ent.type}:{ent.name}"


def _build_embed_text(ent: GraphEntity) -> str:
    """Texto enriquecido para generar embeddings de código."""
    parts = [
        f"Repository: {ent.repo} Branch: {ent.branch}",
        f"File: {ent.file_path}",
        f"Language: {ent.language}",
        f"{ent.type}: {ent.name}",
    ]
    if ent.signature:
        parts.append(f"Signature: {ent.signature}")
    if ent.docstring:
        parts.append(f"Documentation: {ent.docstring}")
    if ent.annotations:
        parts.append(f"Annotations: {', '.join(ent.annotations)}")
    for rel in ent.relations:
        parts.append(f"{rel.type} {rel.target_type} {rel.target_name}")
    parts.append("Code:")
    parts.append(ent.code)
    return "\n".join(parts)
