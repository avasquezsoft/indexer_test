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


def upsert_entities(entities: list):
    """
    Crea/actualiza nodos y relaciones en batch.
    Las entidades deben ser dicts o pydantic models con los campos de GraphEntity.
    """
    if not entities:
        return

    driver = get_driver()
    # Preparar datos planos
    nodes = []
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
            "relations": [
                {
                    "rel_type": r.get("type", "CALLS"),
                    "target_name": r.get("target_name", ""),
                    "target_type": r.get("target_type", "Unknown"),
                    "properties": r.get("properties", {}),
                }
                for r in ent_dict.get("relations", [])
            ],
        })

    with driver.session() as session:
        # Upsert nodos
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
                e.annotations = node.annotations
            """,
            nodes=nodes,
        )

        # Upsert relaciones (en batch separado para evitar complicaciones con UNWIND anidado)
        rels = []
        for node in nodes:
            for r in node["relations"]:
                rels.append({
                    "source_id": node["id"],
                    "rel_type": r["rel_type"],
                    "target_name": r["target_name"],
                    "target_type": r["target_type"],
                    "properties": r["properties"],
                })

        if rels:
            session.run(
                """
                UNWIND $rels AS rel
                MATCH (a:CodeEntity {id: rel.source_id})
                MATCH (b:CodeEntity)
                WHERE b.repo = a.repo AND b.branch = a.branch AND b.name = rel.target_name
                CALL apoc.merge.relationship(a, rel.rel_type,
                    {target_id: b.id},
                    rel.properties,
                    b
                ) YIELD rel AS r
                RETURN count(r) AS created
                """,
                rels=rels,
            )

    logger.info("Neo4j: upserted %s nodos, %s relaciones", len(nodes), len(rels))


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
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH path = (start:CodeEntity {id: $entity_id})-[:EXTENDS|IMPLEMENTS|HAS_METHOD|HAS_FIELD|CALLS|INJECTED*1..$depth]-(related:CodeEntity)
            RETURN DISTINCT related.id AS id,
                   related.name AS name,
                   related.type AS type,
                   related.file_path AS file_path,
                   related.signature AS signature,
                   related.code AS code,
                   length(path) AS distance
            ORDER BY distance, related.name
            """,
            entity_id=entity_id, depth=depth,
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
                   e.end_line AS end_line
            ORDER BY e.file_path
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
            RETURN e.id AS id,
                   e.name AS name,
                   e.type AS type,
                   e.file_path AS file_path,
                   e.signature AS signature,
                   e.code AS code,
                   collect({rel_type: type(r), target_name: target.name, target_type: target.type}) AS relations
            """,
            entity_id=entity_id,
        )
        record = result.single()
        return dict(record) if record else None
