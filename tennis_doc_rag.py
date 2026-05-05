"""
title: Tennis Doc RAG
author: Tritech Prime
version: 1.1
description: >
  Inyecta automáticamente contexto del código fuente indexado en cada conversación.
  Detecta repositorios y ramas mencionados en la pregunta. Rama por defecto: prod.
  Permite también disparar indexación escribiendo "indexa org/repo [rama]" en el chat.
"""

import re
import requests
from typing import Optional

# Regex para detectar formato org/repo en cualquier parte del mensaje
_REPO_RE = re.compile(r"[\w.-]+/[\w.-]+")
# Regex para detectar rama explícita: "rama X" o "branch X"
_BRANCH_RE = re.compile(r"(?:rama|branch)\s+(\S+)", re.IGNORECASE)


class Filter:
    def __init__(self):
        self.name = "Tennis Doc RAG"
        self.valves = self.Valves()

    class Valves:
        def __init__(self):
            # URL del servicio indexer dentro de la red de Docker Compose
            self.indexer_url = "http://indexer:8001"
            # Cantidad de chunks a recuperar por cada pregunta
            self.limit = 6
            # Rama por defecto cuando el usuario no la especifica
            self.default_branch = "prod"

    def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        """
        Se ejecuta ANTES de enviar los mensajes al LLM.
        Busca contexto relevante en el indexer y lo inyecta como mensaje de sistema.
        """
        messages = body.get("messages", [])
        if not messages:
            return body

        last_msg = messages[-1]
        if last_msg.get("role") != "user":
            return body

        query = last_msg.get("content", "").strip()
        if not query:
            return body

        # ── Comando especial: indexar un repo desde el chat ──
        lower_q = query.lower()
        if lower_q.startswith(("indexa ", "indexar ", "reindexa ", "reindexar ", "index ")):
            args = query.split(" ", 1)[1].strip() if " " in query else ""
            parts = args.split()
            repo = parts[0] if parts else ""
            branch = parts[1] if len(parts) > 1 else "HEAD"
            if repo:
                return self._trigger_index(body, repo, branch)

        # ── RAG automático: detectar repo/rama y buscar contexto ──
        repo, branch = self._extract_repo_branch(query)
        try:
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
                # Insertar justo antes del último mensaje del usuario
                messages.insert(-1, system_msg)
                body["messages"] = messages
        except Exception as e:
            print(f"[TennisDoc RAG] Error buscando contexto: {e}")

        return body

    def _extract_repo_branch(self, query: str) -> tuple[Optional[str], str]:
        """Extrae repo (org/repo) y rama del query. Si no hay rama explícita, usa default_branch."""
        # Buscar repos mencionados
        repos = _REPO_RE.findall(query)
        repo = repos[0] if repos else None

        # Buscar rama explícita
        branch_match = _BRANCH_RE.search(query)
        branch = branch_match.group(1) if branch_match else self.valves.default_branch

        return repo, branch

    def _trigger_index(self, body: dict, repo: str, branch: str = "HEAD") -> dict:
        """Dispara la indexación de un repo (rama opcional) y reemplaza el mensaje del usuario con una confirmación."""
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
        except Exception as e:
            body["messages"][-1]["content"] = (
                f"[Sistema interno] Error al iniciar indexación de `{repo}` @ `{branch}`: {e}"
            )
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
