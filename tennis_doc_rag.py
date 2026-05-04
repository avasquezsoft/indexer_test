"""
title: Tennis Doc RAG
author: Tritech Prime
version: 1.0
description: >
  Inyecta automáticamente contexto del código fuente indexado en cada conversación.
  Permite también disparar indexación escribiendo "indexa org/repo" en el chat.
"""

import requests
from typing import Optional


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
            repo = query.split(" ", 1)[1].strip() if " " in query else ""
            if repo:
                return self._trigger_index(body, repo)

        # ── RAG automático: buscar contexto relevante ──
        try:
            resp = requests.post(
                f"{self.valves.indexer_url}/search",
                json={"query": query, "limit": self.valves.limit},
                headers={"Content-Type": "application/json"},
                timeout=15,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])

            if results:
                context = self._build_context(results)
                system_msg = {
                    "role": "system",
                    "content": (
                        "Eres un asistente técnico especializado en el código fuente de la organización. "
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

    def _trigger_index(self, body: dict, repo: str) -> dict:
        """Dispara la indexación de un repo y reemplaza el mensaje del usuario con una confirmación."""
        try:
            resp = requests.post(
                f"{self.valves.indexer_url}/index",
                json={"repo": repo},
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
            data = resp.json()
            body["messages"][-1]["content"] = (
                f"[Sistema interno] He iniciado la indexación de `{repo}`. "
                f"Estado: {data.get('status', 'ok')}. "
                f"Por favor espera unos minutos y luego haz tu pregunta sobre ese repositorio."
            )
        except Exception as e:
            body["messages"][-1]["content"] = (
                f"[Sistema interno] Error al iniciar indexación de `{repo}`: {e}"
            )
        return body

    def _build_context(self, results: list) -> str:
        """Formatea los chunks recuperados para el prompt."""
        parts = []
        for i, r in enumerate(results, 1):
            score = r.get("score", 0)
            file_path = r.get("file_path", "unknown")
            repo = r.get("repo", "unknown")
            text = r.get("text", "")[:1200]
            parts.append(
                f"[Fragmento {i}] Score: {score:.3f} | Repo: {repo} | Archivo: {file_path}\n```\n{text}\n```"
            )
        return "\n\n---\n\n".join(parts)
