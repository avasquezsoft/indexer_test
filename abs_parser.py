"""
abs_parser.py — Clases del 4GL ABS (archivos .ABSClass del plugin ABS de Eclipse).

Cada .ABSClass es un XML que describe una clase: cabecera (paquete, patrón,
descripción), <Attribute> (campos) y <Method> con el código en <Code>, escrito
en el 4GL de ABS ("method x(...) returning T ... end", "invoke m on vObj",
"retrieve Entidad using ... as key") o en Java.

El XML trae cientos de banderas de UI por atributo; aquí se reduce a una vista
legible, que es lo que se indexa y lo que ve el modelo en el chat.
"""

import re

import defusedxml.ElementTree as SafeET
from defusedxml import DefusedXmlException

from ast_parser import GraphEntity, Relation

# Etiquetas de la cabecera de la clase que aportan (el resto son banderas de UI)
_CLASS_INFO = (
    ("PatternName", "Patrón"),
    ("SuperClassName", "Extiende"),
    ("ReferencedEntityName", "Entidad referenciada"),
    ("OptTableName", "Tabla"),
    ("TablesUsed", "Tablas usadas"),
    ("JoinCode", "Join"),
)

# invoke metodo on vObjeto / invoke metodo on my Atributo / invoke metodo (de la propia clase)
_INVOKE_RE = re.compile(r"\binvoke\s+(\w+)(?:\s+on\s+(my\s+\w+|that\s+\w+|\w+))?", re.IGNORECASE)
# create object|session Clase in vVariable — da el tipo del receptor de invoke
_CREATE_RE = re.compile(r"\bcreate\s+(?:object|session)\s+([\w.]+)(?:\s+using\b.*?)?\s+in\s+(\w+)", re.IGNORECASE)
# retrieve [for update] [1|all|vN] Entidad where... — "for update" bloquea para modificar: cuenta como escritura
_RETRIEVE_RE = re.compile(
    r"^\s*(?i:retrieve)\s+((?i:for\s+update)\s+)?(?:(?:\d+|(?i:all)|[av][A-Z]\w*)\s+)?([A-Z]\w*)",
    re.MULTILINE,
)
# method nombre (Tipo aParam, Tipo2 aParam2) returning T
_PARAMS_RE = re.compile(r"\(([^)]*)\)")


def _text(node, tag: str) -> str:
    return (node.findtext(tag) or "").strip()


def _one_line(text: str) -> str:
    """Descripciones multilínea en una línea, para que no rompan los comentarios //."""
    return " ".join(text.split())


def _parse_xml(source: str):
    # defusedxml con forbid_dtd: un DOCTYPE en cualquier parte (XXE, "billion laughs") se
    # rechaza en el parser; un <!DOCTYPE html> dentro de un CDATA es texto y no molesta.
    try:
        root = SafeET.fromstring(source.encode("utf-8"), forbid_dtd=True)
    except (SafeET.ParseError, DefusedXmlException):
        return None
    return root if root.tag == "Class" else None


def _signature(name: str, code: str) -> str:
    """Primera línea del código ("method x(...) returning T" o la firma Java)."""
    for line in code.splitlines():
        line = line.strip()
        if line and not line.startswith(("//", "/*", "*")):
            return line.split("{")[0].strip()
    return name


def _methods(root) -> list[tuple[str, str, str, str]]:
    """(nombre, firma, descripción, código) de cada método con código."""
    result = []
    for method in root.iter("Method"):
        name, code = _text(method, "Name"), (method.findtext("Code") or "").strip("\n")
        if name and code.strip():
            result.append((name, _signature(name, code), _text(method, "Description"), code))
    return result


def render(source: str, bodies: set[str] | None = None) -> str | None:
    """
    Vista legible de un .ABSClass: cabecera, atributos y métodos.
    bodies: métodos que se muestran con su código completo (None = todos);
    los demás solo con la firma. Devuelve None si no es un .ABSClass válido.
    """
    root = _parse_xml(source)
    if root is None:
        return None
    name, package = _text(root, "Name"), _text(root, "PackageId")
    lines = [f"// ABSClass {package}.{name}" if package else f"// ABSClass {name}"]
    description = _one_line(_text(root, "Description") or _text(root, "ShortDescription"))
    if description:
        lines.append(f"// {description}")
    lines += [f"// {label}: {value}" for tag, label in _CLASS_INFO if (value := _text(root, tag))]
    view = _text(root, "ViewSource")
    if view:
        lines.append(f"// Vista SQL:\n{view}")

    lines.append(f"class {name} {{")
    attributes = [a for a in root.findall("Attribute") if _text(a, "Name")]
    if attributes:
        lines.append("  // Atributos")
        for attr in attributes:
            attr_type = _text(attr, "RefFieldName") or _text(attr, "RelatedClassName")
            attr_desc = _one_line(_text(attr, "Description") or _text(attr, "ObjectLabel"))
            lines.append(f"  {_text(attr, 'Name')}: {attr_type}" + (f"  // {attr_desc}" if attr_desc else ""))

    omitted = 0
    for mname, signature, mdesc, code in _methods(root):
        lines.append("")
        if mdesc:
            lines.append(f"  // {_one_line(mdesc)}")
        if bodies is None or mname in bodies:
            lines.append(code)
        else:
            lines.append(f"  {signature}  // ... cuerpo omitido ...")
            omitted += 1
    lines.append("}")
    if omitted:
        lines.insert(0, f"// [Tennis Doc] Vista condensada: se omitieron los cuerpos de {omitted} métodos por espacio.")
    return "\n".join(lines)


def render_for_question(source: str, methods: list[str], terms: list[str], max_chars: int) -> str:
    """
    Vista para el chat: completa si cabe; si no, con el código completo de los
    métodos mencionados o que contienen términos de la pregunta y el resto solo con su firma.
    """
    full = render(source)
    if full is None:
        return source[:max_chars]
    if len(full) <= max_chars:
        return full
    root = _parse_xml(source)
    wanted = set(methods)
    relevant = {
        name for name, _, _, code in _methods(root)
        if name in wanted or any(t in code.lower() for t in terms)
    }
    view = render(source, bodies=relevant) or full
    if len(view) > max_chars:
        view = view[:max_chars] + "\n\n// ... vista truncada ...\n"
    return view


def _call_relations(code: str, signature: str, class_name: str) -> list[Relation]:
    """CALLS por cada invoke, con la clase del receptor cuando se puede inferir."""
    var_types = {var: cls.split(".")[-1] for cls, var in _CREATE_RE.findall(code)}
    params = _PARAMS_RE.search(signature)
    for param in (params.group(1).split(",") if params else []):
        parts = param.split()
        if len(parts) >= 2:
            var_types.setdefault(parts[-1], parts[-2].rstrip("[]"))

    relations, seen = [], set()
    for method, receiver in _INVOKE_RE.findall(code):
        if not receiver:
            owners = [class_name]
        elif receiver in var_types:
            owners = [var_types[receiver]]
        else:
            owners = None  # my X / that X / variable sin tipo conocido: se enlaza por nombre
        key = (method, tuple(owners or ()))
        if key not in seen:
            seen.add(key)
            relations.append(Relation(type="CALLS", target_name=method, target_type="Method", target_owners=owners))
    return relations


def parse(source: str, repo: str, branch: str, file_path: str) -> list[GraphEntity]:
    """Entidades de un .ABSClass: la clase (vista sin cuerpos) y un Method por método con código."""
    root = _parse_xml(source)
    if root is None:
        return []
    name = _text(root, "Name")
    end_line = source.count("\n") + 1

    def entity(ent_type: str, ent_name: str, code: str, signature: str, docstring: str) -> GraphEntity:
        # ponytail: las líneas son del archivo completo; el XML no da posiciones útiles por método
        return GraphEntity(
            type=ent_type, name=ent_name, file_path=file_path, repo=repo, branch=branch,
            language="abs", start_line=1, end_line=end_line, signature=signature,
            docstring=docstring or None, code=code,
        )

    cls = entity(
        "Class", name, render(source, bodies=set()) or "",
        f"{_text(root, 'PatternName')} {name}".strip(),
        _text(root, "Description") or _text(root, "ShortDescription"),
    )
    if super_name := _text(root, "SuperClassName"):
        cls.relations.append(Relation(type="EXTENDS", target_name=super_name.split(".")[-1], target_type="Class"))

    entities = []
    for mname, signature, mdesc, code in _methods(root):
        method = entity("Method", mname, code, signature, mdesc)
        method.relations += _call_relations(code, signature, name)
        access = {("WRITES" if for_update else "READS", t.upper()) for for_update, t in _RETRIEVE_RE.findall(code)}
        method.relations += [Relation(type=rel, target_name=t, target_type="Table") for rel, t in sorted(access)]
        entities.append(method)
        cls.relations.append(Relation(type="HAS_METHOD", target_name=mname, target_type="Method"))
    entities.append(cls)
    return entities


def tables_read(entities: list[GraphEntity]) -> set[str]:
    """Tablas (entidades) leídas o bloqueadas para escribir, para crear sus nodos Table."""
    return {r.target_name for e in entities for r in e.relations if r.type in ("READS", "WRITES")}
