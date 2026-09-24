"""
graph_store.py — Cliente Neo4j para el grafo de código.

Gestiona nodos (:CodeEntity) y relaciones entre ellos.
"""

import logging
from typing import Literal

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

from config import NEO4J_PASSWORD, NEO4J_URL, NEO4J_USER

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# Driver singleton
# ═══════════════════════════════════════════════════════════════

_driver = None


def get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return _driver


def close_driver():
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


# ═══════════════════════════════════════════════════════════════
# Inicialización de schema
# ═══════════════════════════════════════════════════════════════

_SCHEMA_QUERIES = [
    """
    CREATE CONSTRAINT code_entity_id IF NOT EXISTS
    FOR (e:CodeEntity) REQUIRE e.id IS UNIQUE
    """,
    """
    CREATE INDEX code_entity_name IF NOT EXISTS
    FOR (e:CodeEntity) ON (e.name)
    """,
    """
    CREATE INDEX code_entity_type IF NOT EXISTS
    FOR (e:CodeEntity) ON (e.type)
    """,
    """
    CREATE INDEX code_entity_repo_branch IF NOT EXISTS
    FOR (e:CodeEntity) ON (e.repo, e.branch)
    """,
]


def init_schema():
    """Crea constraints e índices si no existen."""
    driver = get_driver()
    with driver.session() as session:
        for q in _SCHEMA_QUERIES:
            try:
                session.run(q)
                logger.info("Schema OK: %s", q.strip().splitlines()[0])
            except Neo4jError as exc:
                if "already exists" in str(exc) or "EquivalentSchemaRule" in str(exc):
                    continue
                logger.warning("Error creando schema (posiblemente ya existe): %s", exc)


def ping() -> bool:
    try:
        driver = get_driver()
        with driver.session() as session:
            session.run("RETURN 1")
        return True
    except Exception as exc:
        logger.error("Neo4j ping falló: %s", exc)
        return False


# ═══════════════════════════════════════════════════════════════
# Operaciones CRUD
# ═══════════════════════════════════════════════════════════════

def clear_repo(repo: str, branch: str):
    """Elimina todos los nodos y relaciones de un repo/rama."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (e:CodeEntity {repo: $repo, branch: $branch})
            DETACH DELETE e
            RETURN count(e) AS deleted
            """,
            repo=repo, branch=branch,
        )
        record = result.single()
        deleted = record["deleted"] if record else 0
        logger.info("Neo4j: eliminados %s nodos para %s@%s", deleted, repo, branch)


def upsert_entities(entities: list) -> list[dict]:
    """
    Crea/actualiza los nodos en batch y devuelve sus relaciones SIN crearlas.

    Las relaciones se crean con upsert_relations() cuando ya existen todos los
    nodos del repo: si se crearan por lote, las que apuntan a una entidad de un
    lote posterior se perderían (el destino todavía no existe).
    """
    if not entities:
        return []

    nodes = []
    rels = []
    for ent in entities:
        ent_dict = ent if isinstance(ent, dict) else ent.model_dump()
        node_id = _make_node_id(ent_dict)
        nodes.append({
            "id": node_id,
            "name": ent_dict.get("name", ""),
            "type": ent_dict.get("type", "Class"),
            "language": ent_dict.get("language", ""),
            "repo": ent_dict.get("repo", ""),
            "branch": ent_dict.get("branch", ""),
            "file_path": ent_dict.get("file_path", ""),
            "start_line": ent_dict.get("start_line", 0),
            "end_line": ent_dict.get("end_line", 0),
            "signature": ent_dict.get("signature") or "",
            "docstring": ent_dict.get("docstring") or "",
            "code": ent_dict.get("code", ""),
            "annotations": ent_dict.get("annotations", []),
            "route": ent_dict.get("route") or "",
        })
        for r in ent_dict.get("relations", []):
            rels.append({
                "source_id": node_id,
                "rel_type": r.get("type", "CALLS"),
                "target_name": r.get("target_name", ""),
                "target_path": r.get("target_path"),
                "target_owners": r.get("target_owners"),
                "properties": r.get("properties", {}),
            })

    with get_driver().session() as session:
        session.run(
            """
            UNWIND $nodes AS node
            MERGE (e:CodeEntity {id: node.id})
            SET e.name = node.name,
                e.type = node.type,
                e.language = node.language,
                e.repo = node.repo,
                e.branch = node.branch,
                e.file_path = node.file_path,
                e.start_line = node.start_line,
                e.end_line = node.end_line,
                e.signature = node.signature,
                e.docstring = node.docstring,
                e.code = node.code,
                e.annotations = node.annotations,
                e.route = node.route
            """,
            nodes=nodes,
        )

    logger.info("Neo4j: upserted %s nodos (%s relaciones pendientes)", len(nodes), len(rels))
    return rels


_REL_BATCH = 2000


def upsert_relations(rels: list[dict]) -> int:
    """
    Crea las relaciones buscando el destino por nombre dentro del mismo repo/rama,
    validando que el tipo de destino tenga sentido (HAS_METHOD solo a métodos del
    mismo archivo, CALLS solo a métodos, USES_SQL al .sql exacto, etc.).
    Devuelve cuántas se crearon.

    CALLS usa target_owners (tipo del receptor, ver code_links.resolve_call_owners):
    solo enlaza con métodos de esas clases o de las que las implementan/extienden.
    Sin target_owners se enlaza por nombre con todos los métodos homónimos.
    """
    created = 0
    # CALLS en una segunda pasada: su validación usa HAS_METHOD/IMPLEMENTS,
    # que tienen que existir antes (dentro de una misma consulta no está garantizado).
    groups = [
        [r for r in rels if r["rel_type"] != "CALLS"],
        [r for r in rels if r["rel_type"] == "CALLS"],
    ]
    with get_driver().session() as session:
        for group, i in ((g, i) for g in groups for i in range(0, len(g), _REL_BATCH)):
            record = session.run(
                """
                UNWIND $rels AS rel
                MATCH (a:CodeEntity {id: rel.source_id})
                MATCH (b:CodeEntity)
                WHERE b.repo = a.repo AND b.branch = a.branch AND b.name = rel.target_name
                  AND b.id <> a.id
                  AND CASE rel.rel_type
                        WHEN 'HAS_METHOD' THEN b.type IN ['Method', 'Function'] AND b.file_path = a.file_path
                        WHEN 'HAS_FIELD' THEN b.type = 'Field' AND b.file_path = a.file_path
                        WHEN 'CALLS' THEN b.type IN ['Method', 'Function'] AND b.language = a.language AND (
                            rel.target_owners IS NULL
                            OR EXISTS {
                                MATCH (o:CodeEntity)-[:HAS_METHOD]->(b)
                                WHERE o.name IN rel.target_owners
                                   OR EXISTS {
                                       MATCH (o)-[:IMPLEMENTS|EXTENDS]->(t:CodeEntity)
                                       WHERE t.name IN rel.target_owners
                                   }
                            }
                        )
                        WHEN 'USES_SQL' THEN b.type = 'SqlFile' AND b.file_path = rel.target_path
                        WHEN 'READS' THEN b.type = 'Table'
                        WHEN 'WRITES' THEN b.type = 'Table'
                        WHEN 'ANNOTATED_WITH' THEN true
                        ELSE b.type IN ['Class', 'Interface', 'Enum', 'Record', 'Struct']
                      END
                CALL apoc.merge.relationship(a, rel.rel_type,
                    {target_id: b.id},
                    rel.properties,
                    b
                ) YIELD rel AS r
                RETURN count(r) AS created
                """,
                rels=group[i:i + _REL_BATCH],
            ).single()
            created += record["created"] if record else 0

    logger.info("Neo4j: %s relaciones creadas (de %s declaradas)", created, len(rels))
    return created


def _make_node_id(ent: dict) -> str:
    return f"{ent['repo']}:{ent['branch']}:{ent['file_path']}:{ent['type']}:{ent['name']}"


# ═══════════════════════════════════════════════════════════════
# Consultas de grafo
# ═══════════════════════════════════════════════════════════════

def get_related_entities(entity_id: str, depth: int = 2) -> list[dict]:
    """
    Navega el grafo desde una entidad hacia sus vecinos.
    Devuelve lista de dicts con metadata de cada entidad relacionada.
    """
    # Neo4j no acepta parámetros en rangos de patrones MATCH (*1..$depth).
    # Como depth es un entero pequeño controlado internamente, usamos f-string seguro.
    safe_depth = max(1, min(int(depth), 5))
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            f"""
            MATCH path = (start:CodeEntity {{id: $entity_id}})-[:EXTENDS|IMPLEMENTS|HAS_METHOD|HAS_FIELD|CALLS|INJECTED*1..{safe_depth}]-(related:CodeEntity)
            RETURN DISTINCT related.id AS id,
                   related.name AS name,
                   related.type AS type,
                   related.file_path AS file_path,
                   related.signature AS signature,
                   related.code AS code,
                   length(path) AS distance
            ORDER BY distance, related.name
            """,
            entity_id=entity_id,
        )
        return [dict(record) for record in result]


def search_by_name(name: str, repo: str | None = None, branch: str | None = None) -> list[dict]:
    """Búsqueda exacta por nombre de entidad."""
    driver = get_driver()
    with driver.session() as session:
        query = """
            MATCH (e:CodeEntity)
            WHERE e.name = $name
        """
        params: dict = {"name": name}
        if repo:
            query += " AND e.repo = $repo"
            params["repo"] = repo
        if branch:
            query += " AND e.branch = $branch"
            params["branch"] = branch
        query += """
            RETURN e.id AS id,
                   e.name AS name,
                   e.type AS type,
                   e.file_path AS file_path,
                   e.signature AS signature,
                   e.code AS code,
                   e.start_line AS start_line,
                   e.end_line AS end_line,
                   e.route AS route
            ORDER BY CASE WHEN e.type IN ['Class', 'Interface', 'Enum', 'Record', 'Struct'] THEN 0
                          WHEN e.type IN ['SqlFile', 'Table'] THEN 1 ELSE 2 END,
                     e.file_path
        """
        result = session.run(query, **params)
        return [dict(record) for record in result]


def find_entity_by_id(entity_id: str) -> dict | None:
    """Busca una entidad exacta por su ID completo."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (e:CodeEntity {id: $entity_id})
            RETURN e.id AS id,
                   e.name AS name,
                   e.type AS type,
                   e.file_path AS file_path,
                   e.signature AS signature,
                   e.code AS code,
                   e.start_line AS start_line,
                   e.end_line AS end_line,
                   e.annotations AS annotations
            LIMIT 1
            """,
            entity_id=entity_id,
        )
        record = result.single()
        return dict(record) if record else None


def get_entity_with_direct_relations(entity_id: str) -> dict | None:
    """Devuelve una entidad con sus relaciones directas agrupadas."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (e:CodeEntity {id: $entity_id})
            OPTIONAL MATCH (e)-[r]->(target:CodeEntity)
            WITH e, [x IN collect({rel_type: type(r), target_name: target.name, target_type: target.type})
                     WHERE x.rel_type IS NOT NULL] AS relations
            OPTIONAL MATCH (source:CodeEntity)-[r2]->(e)
            WHERE NOT type(r2) IN ['HAS_METHOD', 'HAS_FIELD', 'IMPORTS']
            WITH e, relations, [x IN collect({rel_type: type(r2), source_name: source.name, source_type: source.type})
                                WHERE x.rel_type IS NOT NULL][..30] AS used_by
            RETURN e.id AS id,
                   e.name AS name,
                   e.type AS type,
                   e.file_path AS file_path,
                   e.signature AS signature,
                   e.code AS code,
                   e.route AS route,
                   relations,
                   used_by
            """,
            entity_id=entity_id,
        )
        record = result.single()
        return dict(record) if record else None


# ═══════════════════════════════════════════════════════════════
# Consultas de uso: quién usa, flujo, SQL/tablas y endpoints
# ═══════════════════════════════════════════════════════════════

_SCOPE = """
    AND ($repo IS NULL OR t.repo = $repo)
    AND ($branch IS NULL OR t.branch = $branch)
"""


def find_usages(name: str, repo: str | None = None, branch: str | None = None, limit: int = 100) -> list[dict]:
    """
    Quién usa una entidad: llamadas, inyecciones, herencia y uso de SQL/tablas.
    Si es una clase, también incluye quién llama a sus métodos.
    """
    with get_driver().session() as session:
        result = session.run(
            """
            MATCH (t:CodeEntity {name: $name})
            WHERE true """ + _SCOPE + """
            OPTIONAL MATCH (t)-[:HAS_METHOD]->(m:CodeEntity)
            WITH t, collect(DISTINCT m) AS members
            WITH [t] + members AS targets
            UNWIND targets AS target
            MATCH (s:CodeEntity)-[r]->(target)
            WHERE NOT type(r) IN ['HAS_METHOD', 'HAS_FIELD']
            OPTIONAL MATCH (owner:CodeEntity)-[:HAS_METHOD]->(s)
            RETURN DISTINCT type(r) AS rel_type,
                   target.name AS target_name,
                   target.type AS target_type,
                   s.name AS source_name,
                   s.type AS source_type,
                   owner.name AS source_class,
                   s.file_path AS file_path,
                   s.start_line AS start_line,
                   s.route AS route,
                   s.repo AS repo
            ORDER BY file_path, start_line
            LIMIT $limit
            """,
            name=name, repo=repo, branch=branch, limit=limit,
        )
        return [dict(record) for record in result]


def get_flow(name: str, repo: str | None = None, branch: str | None = None, depth: int = 4, limit: int = 200) -> list[dict]:
    """
    Flujo hacia abajo desde una entidad: qué llama, qué inyecta, qué SQL
    ejecuta y qué tablas toca. Devuelve las aristas del recorrido.

    ponytail: recorrido de longitud variable acotado por depth y LIMIT; en repos
    enormes con CALLS muy ambiguos puede tardar, bajar depth si pasa.
    """
    # Neo4j no acepta parámetros en el rango *1..n; depth se acota aquí.
    safe_depth = max(1, min(int(depth), 6))
    with get_driver().session() as session:
        result = session.run(
            """
            MATCH (t:CodeEntity {name: $name})
            WHERE true """ + _SCOPE + """
            OPTIONAL MATCH (t)-[:HAS_METHOD]->(m:CodeEntity)
            WITH t, collect(DISTINCT m) AS members
            WITH [t] + members AS starts
            UNWIND starts AS start
            MATCH p = (start)-[:CALLS|INJECTED|USES_SQL|READS|WRITES*1..""" + str(safe_depth) + """]->(:CodeEntity)
            UNWIND relationships(p) AS r
            WITH DISTINCT r, startNode(r) AS a, endNode(r) AS b
            OPTIONAL MATCH (ac:CodeEntity)-[:HAS_METHOD]->(a)
            OPTIONAL MATCH (bc:CodeEntity)-[:HAS_METHOD]->(b)
            RETURN a.name AS source, a.type AS source_type, ac.name AS source_class,
                   type(r) AS rel_type,
                   b.name AS target, b.type AS target_type, bc.name AS target_class,
                   b.file_path AS target_file
            LIMIT $limit
            """,
            name=name, repo=repo, branch=branch, limit=limit,
        )
        return [dict(record) for record in result]


def find_table_usage(table: str, repo: str | None = None, branch: str | None = None, limit: int = 100) -> list[dict]:
    """Quién lee/escribe una tabla: el .sql (si aplica), el método y su clase."""
    with get_driver().session() as session:
        result = session.run(
            """
            MATCH (t:CodeEntity {type: 'Table', name: $table})
            WHERE true """ + _SCOPE + """
            MATCH (s:CodeEntity)-[r:READS|WRITES]->(t)
            OPTIONAL MATCH (m:CodeEntity)-[:USES_SQL]->(s)
            WITH t, r, s, CASE WHEN s.type = 'SqlFile' THEN m ELSE s END AS user
            OPTIONAL MATCH (c:CodeEntity)-[:HAS_METHOD]->(user)
            RETURN DISTINCT t.repo AS repo,
                   type(r) AS access,
                   CASE WHEN s.type = 'SqlFile' THEN s.file_path END AS sql_file,
                   user.name AS method,
                   c.name AS class,
                   user.file_path AS file_path,
                   user.route AS route
            ORDER BY repo, file_path
            LIMIT $limit
            """,
            table=table.upper(), repo=repo, branch=branch, limit=limit,
        )
        return [dict(record) for record in result]


def list_tables(repo: str, branch: str | None = None) -> list[dict]:
    """Tablas de un repo con cuántos SQL/métodos las leen y escriben."""
    with get_driver().session() as session:
        result = session.run(
            """
            MATCH (t:CodeEntity {type: 'Table', repo: $repo})
            WHERE $branch IS NULL OR t.branch = $branch
            OPTIONAL MATCH (s)-[:READS]->(t)
            WITH t, count(DISTINCT s) AS readers
            OPTIONAL MATCH (s2)-[:WRITES]->(t)
            RETURN t.name AS table, readers, count(DISTINCT s2) AS writers
            ORDER BY table
            """,
            repo=repo, branch=branch,
        )
        return [dict(record) for record in result]


def list_endpoints(repo: str, branch: str | None = None) -> list[dict]:
    """Rutas HTTP expuestas por los controllers de un repo."""
    with get_driver().session() as session:
        result = session.run(
            """
            MATCH (m:CodeEntity {repo: $repo})
            WHERE m.route <> '' AND ($branch IS NULL OR m.branch = $branch)
            OPTIONAL MATCH (c:CodeEntity)-[:HAS_METHOD]->(m)
            RETURN m.route AS route, m.name AS method, c.name AS class,
                   m.file_path AS file_path, m.start_line AS start_line
            ORDER BY route
            """,
            repo=repo, branch=branch,
        )
        return [dict(record) for record in result]
