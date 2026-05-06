"""
title: Tennis Doc RAG
author: Tritech Prime
version: 1.4
description: >
  Inyecta automáticamente contexto del código fuente indexado en cada conversación.
  Detecta repositorios y ramas mencionados en la pregunta. Rama por defecto: prod.
  Permite disparar indexación escribiendo "indexa org/repo [rama]" en el chat.
  Permite generar PDFs de la última respuesta del asistente.
"""

import base64
import re
import requests

# Regex para detectar formato org/repo en cualquier parte del mensaje
_REPO_RE = re.compile(r"[\w.-]+/[\w.-]+")
# Regex para detectar rama explícita: "rama X" o "branch X"
_BRANCH_RE = re.compile(r"(?:rama|branch)\s+(\S+)", re.IGNORECASE)
# Regex para detectar solicitud de PDF
_PDF_RE = re.compile(r"(?:genera?r?|crea?r?|descarga?r?|exporta?r?)\s+(?:un\s+)?pdf", re.IGNORECASE)
# Palabras que indican que el usuario busca implementación/código
_IMPL_KEYWORDS = re.compile(
    r"\b(m[oó]dulo|module|implementaci[oó]n|implementation|expl[íi]came|explica|c[oó]mo\s+funciona|queries?|sql|dao|repositorio|service|servicio|m[eé]todos?|clase|class|business\s+logic|l[oó]gica)\b",
    re.IGNORECASE,
)
# Extrae posibles nombres de módulo de un path tipo "foo/bar/module-name"
_MODULE_PATH_RE = re.compile(r"[\w-]+/[\w-]+/([\w-]+)")


class Filter:
    def __init__(self):
        self.name = "Tennis Doc RAG"
        self.valves = self.Valves()
        print("[TennisDoc RAG] Filter cargado correctamente")

    class Valves:
        def __init__(self):
            self.indexer_url = "http://indexer:8001"
            self.limit = 12
            self.default_branch = "prod"

    def _enrich_query(self, query: str) -> str:
        """Enriquece la query del usuario con keywords técnicas para mejorar retrieval."""
        if not _IMPL_KEYWORDS.search(query):
            return query

        enrichment = []

        # Extraer posible nombre de módulo de paths mencionados
        module_match = _MODULE_PATH_RE.search(query)
        if module_match:
            enrichment.append(module_match.group(1).replace("-", " "))

        # Añadir keywords técnicas según lo que parece buscar
        lower = query.lower()
        if any(k in lower for k in ("query", "queries", "sql", "jpql", "select", "insert", "update")):
            enrichment.extend(["SQL query", "database", "DAO", "implementation"])
        if any(k in lower for k in ("módulo", "modulo", "module", "implementación", "implementation", "explícame", "explica", "cómo funciona")):
            enrichment.extend(["Java class", "implementation", "methods", "business logic", "DAO", "service"])
        if any(k in lower for k in ("dao", "repositorio", "repository")):
            enrichment.extend(["DAO", "implementation", "database queries"])

        if enrichment:
            return f"{query} {' '.join(enrichment)}"
        return query

    def inlet(self, body: dict, user: dict = None) -> dict:
        """Se ejecuta ANTES de enviar los mensajes al LLM."""
        try:
            messages = body.get("messages", [])
            if not messages:
                return body

            last_msg = messages[-1]
            if last_msg.get("role") != "user":
                return body

            # El contenido puede ser string o lista (si tiene attachments)
            raw_content = last_msg.get("content", "")
            if isinstance(raw_content, list):
                # Extraer solo el texto del primer elemento
                query = str(raw_content[0].get("text", "")).strip() if raw_content else ""
            else:
                query = str(raw_content).strip()

            if not query:
                return body

            print(f"[TennisDoc RAG] Query recibida: {query[:80]}...")

            # ── Comando especial: indexar un repo desde el chat ──
            lower_q = query.lower()
            index_commands = ("indexa ", "indexar ", "reindexa ", "reindexar ", "index ")
            if lower_q.startswith(index_commands):
                args = query.split(" ", 1)[1].strip() if " " in query else ""
                parts = args.split()
                repo = parts[0] if parts else ""
                branch = parts[1] if len(parts) > 1 else "HEAD"
                if repo:
                    print(f"[TennisDoc RAG] Comando index detectado: {repo} @ {branch}")
                    return self._trigger_index(body, repo, branch)

            # ── Generar PDF de la última respuesta del asistente ──
            if _PDF_RE.search(query):
                repo, branch = self._extract_repo_branch(query)
                return self._generate_pdf(body, repo, branch)

            # ── RAG automático: detectar repo/rama y buscar contexto ──
            repo, branch = self._extract_repo_branch(query)
            search_query = self._enrich_query(query)
            print(f"[TennisDoc RAG] Buscando contexto | repo={repo} | branch={branch} | enriched_query={search_query[:100]}...")

            payload = {"query": search_query, "limit": self.valves.limit}
            if repo:
                payload["repo"] = repo
            if branch:
                payload["branch"] = branch

            resp = requests.post(
                f"{self.valves.indexer_url}/search",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=15,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            print(f"[TennisDoc RAG] Chunks encontrados: {len(results)}")

            if results:
                context = self._build_context(results)
                scope = f"Repo: {repo} | Rama: {branch}" if repo else f"Todas las ramas (filtro: {branch})"
                system_msg = {
                    "role": "system",
                    "content": (
                        "Eres un asistente técnico especializado en el código fuente de la organización. "
                        f"Ámbito de búsqueda: {scope}. "
                        "Responde ÚNICAMENTE basándote en el siguiente contexto del código. "
                        "IMPORTANTE: los fragmentos pueden contener el código Java junto con las queries SQL referenciadas (marcadas como '-- Referenced SQL:'). "
                        "Busca en TODOS los fragmentos: firmas de métodos, implementaciones, queries SQL/JPQL, lógica de negocio y mapeo de entidades. "
                        "Si la respuesta no está en el contexto, indica que no tienes información suficiente.\n\n"
                        f"{context}"
                    ),
                }
                messages.insert(-1, system_msg)
                body["messages"] = messages

        except Exception as e:
            print(f"[TennisDoc RAG] ERROR en inlet: {e}")
            # En caso de error, devolvemos el body sin modificar para no romper el chat

        return body

    def _extract_repo_branch(self, query: str):
        """Extrae repo (org/repo) y rama del query."""
        repos = _REPO_RE.findall(query)
        repo = repos[0] if repos else None
        branch_match = _BRANCH_RE.search(query)
        branch = branch_match.group(1) if branch_match else self.valves.default_branch
        return repo, branch

    def _trigger_index(self, body: dict, repo: str, branch: str = "HEAD") -> dict:
        """Dispara la indexación y reemplaza el mensaje del usuario con una confirmación."""
        try:
            payload = {"repo": repo, "branch": branch}
            resp = requests.post(
                f"{self.valves.indexer_url}/index",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
            data = resp.json()
            body["messages"][-1]["content"] = (
                f"[Sistema interno] He iniciado la indexación de `{repo}` @ `{branch}`. "
                f"Estado: {data.get('status', 'ok')}. "
                f"Por favor espera unos minutos y luego haz tu pregunta sobre ese repositorio."
            )
            print(f"[TennisDoc RAG] Indexación iniciada: {repo} @ {branch}")
        except Exception as e:
            body["messages"][-1]["content"] = (
                f"[Sistema interno] Error al iniciar indexación de `{repo}` @ `{branch}`: {e}"
            )
            print(f"[TennisDoc RAG] ERROR indexando: {e}")
        return body

    def _generate_pdf(self, body: dict, repo: str, branch: str) -> dict:
        """Genera un PDF de la última respuesta del asistente y la ofrece como descarga."""
        try:
            messages = body.get("messages", [])
            # Buscar la última respuesta del asistente
            assistant_msg = None
            for msg in reversed(messages[:-1]):
                if msg.get("role") == "assistant":
                    assistant_msg = msg
                    break

            if not assistant_msg:
                body["messages"][-1]["content"] = "No hay una respuesta previa del asistente para convertir a PDF."
                return body

            content = assistant_msg.get("content", "")
            title = f"Respuesta_{repo.replace('/', '_')}" if repo else "Respuesta"

            resp = requests.post(
                f"{self.valves.indexer_url}/pdf",
                json={"title": title, "content": content, "repo": repo, "branch": branch},
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            resp.raise_for_status()
            pdf_bytes = resp.content
            pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")

            download_link = f'<a href="data:application/pdf;base64,{pdf_b64}" download="{title}.pdf">📄 Descargar PDF</a>'

            body["messages"][-1]["content"] = (
                f"He generado el PDF con la respuesta anterior. "
                f"Haz clic para descargarlo:\n\n{download_link}"
            )
            print(f"[TennisDoc RAG] PDF generado: {title}.pdf ({len(pdf_bytes)} bytes)")
        except Exception as e:
            body["messages"][-1]["content"] = f"[Sistema] Error generando PDF: {e}"
            print(f"[TennisDoc RAG] ERROR generando PDF: {e}")
        return body

    def _build_context(self, results: list) -> str:
        """Formatea los chunks recuperados para el prompt."""
        parts = []
        for i, r in enumerate(results, 1):
            score = r.get("score", 0)
            file_path = r.get("file_path", "unknown")
            repo = r.get("repo", "unknown")
            branch = r.get("branch", "")
            text = r.get("text", "")[:12000]
            branch_info = f" | Rama: {branch}" if branch else ""
            parts.append(
                f"[Fragmento {i}] Score: {score:.3f} | Repo: {repo}{branch_info} | Archivo: {file_path}\n```\n{text}\n```"
            )
        return "\n\n---\n\n".join(parts)
