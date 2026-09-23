"""
title: Tennis Doc Tools
version: 1.0
requirements: requests
description: Acciones de Tennis Doc IA que el modelo puede llamar solo (function calling):
             listar/verificar repos indexados, indexar, ver el grafo de una clase y exportar a PDF.

Uso
===
1. Workspace → Tools → "+" → pega este archivo. Guarda con el ID `tennis_doc_tools`
   (o cambia la valve `tools_id` del filtro Tennis Doc RAG).
2. En el modelo: activa esta Tool y, en Parámetros avanzados, Function Calling = "Native"
   si el modelo lo soporta (Qwen 2.5+, Llama 3.1+, GPT-4o, etc.).
3. Cuando la Tool está activa en un chat, el filtro desactiva sus comandos por texto
   automáticamente, así no se ejecuta nada dos veces.
"""

import base64
import io
import os
import uuid
from typing import Any, Callable, Optional

import requests
from pydantic import BaseModel, Field


class Tools:

    class Valves(BaseModel):

        indexer_url: str = Field(
            default_factory=lambda: os.environ.get(
                "TENNIS_DOC_INDEXER_URL", "http://indexer:8001"
            ),
            description="URL base del indexer.",
        )

        api_key: str = Field(
            default_factory=lambda: os.environ.get("INDEXER_API_KEY", ""),
            description="API key del indexer (Bearer).",
        )

        default_branch: str = Field(
            default_factory=lambda: os.environ.get("TENNIS_DOC_DEFAULT_BRANCH", "prod"),
            description="Rama por defecto.",
        )

    def __init__(self):

        self.valves = self.Valves()

    # ========================================================
    # HTTP
    # ========================================================

    def _headers(self):

        headers = {"Content-Type": "application/json", "Accept": "application/json"}

        if self.valves.api_key:

            headers["Authorization"] = f"Bearer {self.valves.api_key}"

        return headers

    def _url(self, endpoint):

        return f"{self.valves.indexer_url.rstrip('/')}{endpoint}"

    def _repos(self):

        response = requests.get(self._url("/repos"), headers=self._headers(), timeout=10)

        response.raise_for_status()

        data = response.json()

        raw = data.get("repos", []) if isinstance(data, dict) else data

        repos = []

        for item in raw or []:

            if isinstance(item, dict):

                name = item.get("repo") or item.get("name") or item.get("full_name")

                branch = item.get("branch") or item.get("branches")

                if isinstance(branch, list):

                    branch = ", ".join(str(b) for b in branch)

            else:

                name, branch = item, None

            if name:

                repos.append({"name": str(name), "branch": branch})

        return sorted(repos, key=lambda r: r["name"].lower())

    @staticmethod
    async def _status(emitter, description, done=False):

        if emitter:

            await emitter(
                {"type": "status", "data": {"description": description, "done": done}}
            )

    # ========================================================
    # TOOLS
    # ========================================================

    async def list_indexed_repositories(
        self,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Lista los repositorios de código que están indexados en Tennis Doc y cuántos hay.
        Úsala cuando el usuario pregunte qué repositorios o proyectos hay indexados,
        cuántos hay, o cuáles puede consultar.
        """

        await self._status(__event_emitter__, "📚 Consultando repositorios indexados…")

        try:

            repos = self._repos()

        except Exception as exc:

            await self._status(__event_emitter__, "Error consultando /repos", True)

            return f"ERROR: el indexer no respondió ({exc}). No inventes repositorios."

        await self._status(__event_emitter__, f"{len(repos)} repositorios", True)

        if not repos:

            return "TOTAL: 0. No hay repositorios indexados. Se indexa con: indexa org/repositorio"

        lines = [f"TOTAL: {len({r['name'] for r in repos})}"]

        for repo in repos:

            lines.append(
                f"- {repo['name']}" + (f" (rama: {repo['branch']})" if repo.get("branch") else "")
            )

        return "\n".join(lines)

    async def check_repository_indexed(
        self,
        repo: str,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Verifica si un repositorio concreto está indexado en Tennis Doc.

        :param repo: Repositorio en formato "organizacion/repositorio", o solo el nombre del repositorio.
        """

        try:

            repos = self._repos()

        except Exception as exc:

            return f"ERROR: el indexer no respondió ({exc})."

        target = repo.strip().strip("`").lower()

        for item in repos:

            name = item["name"].lower()

            if name == target or name.split("/")[-1] == target:

                branch = f" (rama: {item['branch']})" if item.get("branch") else ""

                return f"SÍ: {item['name']}{branch} está indexado."

        return f"NO: {repo} no está indexado. Se puede indexar con la tool index_repository."

    async def index_repository(
        self,
        repo: str,
        branch: str = "",
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Inicia la indexación (o reindexación) de un repositorio en Tennis Doc.
        Úsala SOLO cuando el usuario pida explícitamente indexar o reindexar un repositorio.

        :param repo: Repositorio en formato "organizacion/repositorio".
        :param branch: Rama a indexar. Vacío para usar la rama por defecto.
        """

        repo = repo.strip().strip("`")

        branch = (branch or self.valves.default_branch).strip()

        if "/" not in repo:

            return "ERROR: el repositorio debe tener el formato organizacion/repositorio."

        await self._status(__event_emitter__, f"🗂️ Indexando {repo} @ {branch}…")

        try:

            response = requests.post(
                self._url("/index"),
                json={"repo": repo, "branch": branch},
                headers=self._headers(),
                timeout=10,
            )

            response.raise_for_status()

            status = response.json().get("status", "ok")

        except Exception as exc:

            await self._status(__event_emitter__, "Error al indexar", True)

            return f"ERROR iniciando la indexación de {repo} @ {branch}: {exc}"

        await self._status(__event_emitter__, "Indexación iniciada", True)

        return (
            f"Indexación iniciada. Repositorio: {repo}. Rama: {branch}. Estado: {status}. "
            "Cuando termine se podrán hacer preguntas sobre el repositorio."
        )

    async def show_class_graph(
        self,
        class_name: str,
        repo: str = "",
        branch: str = "",
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Muestra el grafo de una clase o entidad de código: tipo, archivo, firma y sus
        relaciones directas (llama a, extiende, implementa, usa...). Úsala cuando el
        usuario pida el grafo, las relaciones o las dependencias directas de una clase.

        :param class_name: Nombre exacto de la clase, por ejemplo PaymentService.
        :param repo: Repositorio "organizacion/repositorio" (opcional).
        :param branch: Rama (opcional).
        """

        await self._status(__event_emitter__, f"🕸️ Consultando el grafo de {class_name}…")

        params = {k: v for k, v in {"repo": repo, "branch": branch}.items() if v}

        try:

            response = requests.get(
                self._url(f"/graph/entity/{class_name.strip().strip('`')}"),
                params=params,
                headers=self._headers(),
                timeout=15,
            )

            if response.status_code == 404:

                await self._status(__event_emitter__, "No encontrada", True)

                return f"No se encontró {class_name} en el grafo."

            response.raise_for_status()

            data = response.json()

        except Exception as exc:

            await self._status(__event_emitter__, "Error consultando el grafo", True)

            return f"ERROR consultando el grafo de {class_name}: {exc}"

        await self._status(__event_emitter__, "Grafo obtenido", True)

        lines = [
            f"Clase: {class_name}",
            f"Tipo: {data.get('type', 'Unknown')}",
            f"Archivo: {data.get('file_path', 'N/A')}",
        ]

        if data.get("signature"):

            lines.append(f"Firma: {data['signature']}")

        relations = data.get("relations") or []

        if relations:

            lines.append("Relaciones:")

            for relation in relations:

                lines.append(
                    f"- {relation.get('rel_type', 'REL')} -> "
                    f"{relation.get('target_name', '?')} ({relation.get('target_type', '?')})"
                )

        else:

            lines.append("Sin relaciones directas.")

        return "\n".join(lines)

    async def export_last_answer_pdf(
        self,
        title: str = "",
        __messages__: Optional[list] = None,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Convierte la última respuesta del asistente en un PDF descargable.
        Úsala cuando el usuario pida exportar, descargar o guardar la respuesta en PDF.

        :param title: Título del documento (opcional).
        """

        previous = None

        for message in reversed(__messages__ or []):

            if message.get("role") == "assistant" and message.get("content"):

                previous = message

                break

        if not previous:

            return "No hay una respuesta anterior para convertir a PDF."

        content = previous["content"]

        if isinstance(content, list):

            content = " ".join(str(i.get("text", "")) for i in content if isinstance(i, dict))

        title = (title or "Respuesta").strip().replace("/", "_").replace(" ", "_")

        await self._status(__event_emitter__, "📄 Generando PDF…")

        try:

            response = requests.post(
                self._url("/pdf"),
                json={"title": title, "content": content},
                headers=self._headers(),
                timeout=60,
            )

            response.raise_for_status()

            pdf = response.content

        except Exception as exc:

            await self._status(__event_emitter__, "Error generando PDF", True)

            return f"ERROR generando el PDF: {exc}"

        url = _store_file(pdf, f"{title}.pdf", "application/pdf", (__user__ or {}).get("id"))

        if url:

            link = f"📄 [Descargar {title}.pdf]({url})"

        else:

            b64 = base64.b64encode(pdf).decode("utf-8")

            link = (
                f'<a href="data:application/pdf;base64,{b64}" download="{title}.pdf">'
                "📄 Descargar PDF</a>"
            )

        # El enlace se agrega directo al mensaje; el modelo no tiene que copiarlo.
        if __event_emitter__:

            await __event_emitter__({"type": "message", "data": {"content": f"\n\n{link}\n\n"}})

        await self._status(__event_emitter__, "PDF listo", True)

        return "PDF generado y el enlace de descarga ya se mostró al usuario. Confírmalo en una frase."


def _store_file(data, filename, content_type, user_id):
    """Guarda el archivo en Open WebUI y devuelve su URL (None si falla)."""

    try:

        from open_webui.models.files import FileForm, Files
        from open_webui.storage.provider import Storage

        file_id = str(uuid.uuid4())

        stored_name = f"{file_id}_{filename}"

        try:

            _, path = Storage.upload_file(io.BytesIO(data), stored_name, {})

        except TypeError:

            _, path = Storage.upload_file(io.BytesIO(data), stored_name)

        Files.insert_new_file(
            user_id,
            FileForm(
                id=file_id,
                filename=filename,
                path=path,
                meta={"name": filename, "content_type": content_type, "size": len(data)},
            ),
        )

        return f"/api/v1/files/{file_id}/content"

    except Exception as exc:

        print(f"[TennisDoc Tools] No pude guardar el archivo en Open WebUI: {exc}")

        return None
