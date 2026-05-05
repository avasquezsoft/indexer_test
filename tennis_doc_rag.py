"""
title: Tennis Doc RAG
author: Tritech Prime
version: 1.2
description: >
  Inyecta automáticamente contexto del código fuente indexado en cada conversación.
  Detecta repositorios y ramas mencionados en la pregunta. Rama por defecto: prod.
  Permite también disparar indexación escribiendo "indexa org/repo [rama]" en el chat.
"""

import re
import requests

# Regex para detectar formato org/repo en cualquier parte del mensaje
_REPO_RE = re.compile(r"[\w.-]+/[\w.-]+")
# Regex para detectar rama explícita: "rama X" o "branch X"
_BRANCH_RE = re.compile(r"(?:rama|branch)\s+(\S+)", re.IGNORECASE)


class Filter:
    def __init__(self):
        self.name = "Tennis Doc RAG"
        self.valves = self.Valves()
        print("[TennisDoc RAG] Filter cargado correctamente")

    class Valves:
        def __init__(self):
            self.indexer_url = "http://indexer:8001"
            self.limit = 6
            self.default_branch = "prod"

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

            # ── RAG automático: detectar repo/rama y buscar contexto ──
            repo, branch = self._extract_repo_branch(query)
            print(f"[TennisDoc RAG] Buscando contexto | repo={repo} | branch={branch}")

            payload = {"query": query, "limit": self.valves.limit}
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

    def _build_context(self, results: list) -> str:
        """Formatea los chunks recuperados para el prompt."""
        parts = []
        for i, r in enumerate(results, 1):
            score = r.get("score", 0)
            file_path = r.get("file_path", "unknown")
            repo = r.get("repo", "unknown")
            branch = r.get("branch", "")
            text = r.get("text", "")[:1200]
            branch_info = f" | Rama: {branch}" if branch else ""
            parts.append(
                f"[Fragmento {i}] Score: {score:.3f} | Repo: {repo}{branch_info} | Archivo: {file_path}\n```\n{text}\n```"
            )
        return "\n\n---\n\n".join(parts)
