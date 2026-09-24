"""
title: Tennis Doc Tools
version: 1.3
requirements: requests
description: Acciones de Tennis Doc IA que el modelo puede llamar solo (function calling):
             explorar el código (buscar, leer archivos, quién usa qué, flujo hasta SQL y
             tablas, endpoints), listar/verificar repos, indexar, ver el grafo y exportar a PDF o .md.

Uso
===
1. Workspace → Tools → "+" → pega este archivo. Guarda con el ID `tennis_doc_tools`
   (o cambia la valve `tools_id` del filtro Tennis Doc RAG).
2. En el modelo: activa esta Tool y, en Parámetros avanzados, Function Calling = "Native"
   si el modelo lo soporta (Qwen 2.5+, Llama 3.1+, GPT-4o, etc.).
3. Cuando la Tool está activa en un chat, el filtro desactiva sus comandos por texto
   automáticamente, así no se ejecuta nada dos veces.

Exploración
===========
Con estas funciones el modelo puede investigar en varios pasos: buscar la clase,
leer el archivo, ver quién la llama y seguir el flujo hasta el SQL y las tablas.
"""

import base64
import inspect
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

        max_output_chars: int = Field(
            default=30000, description="Máximo de caracteres que devuelve cada función."
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

    def _get_json(self, endpoint, params=None, timeout=30):

        response = requests.get(
            self._url(endpoint),
            params={k: v for k, v in (params or {}).items() if v not in (None, "")},
            headers=self._headers(),
            timeout=timeout,
        )

        if response.status_code == 404:

            return None

        response.raise_for_status()

        return response.json()

    def _cap(self, text):

        limit = self.valves.max_output_chars

        if len(text) <= limit:

            return text

        return text[:limit] + f"\n… (recortado: {len(text)} caracteres en total)"

    @staticmethod
    def _qualified(owner, name):

        return f"{owner}.{name}" if owner else str(name)

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

        if data.get("route"):

            lines.append(f"Endpoint: {data['route']}")

        relations = [r for r in data.get("relations") or [] if r.get("rel_type")]

        if relations:

            lines.append("Relaciones:")

            for relation in relations:

                lines.append(
                    f"- {relation.get('rel_type', 'REL')} -> "
                    f"{relation.get('target_name', '?')} ({relation.get('target_type', '?')})"
                )

        else:

            lines.append("Sin relaciones directas.")

        used_by = [r for r in data.get("used_by") or [] if r.get("rel_type")]

        if used_by:

            lines.append("Usado por:")

            for relation in used_by:

                lines.append(
                    f"- {relation.get('source_name', '?')} ({relation.get('source_type', '?')}) "
                    f"{relation.get('rel_type')}"
                )

        return "\n".join(lines)

    async def search_code(
        self,
        question: str,
        repo: str = "",
        branch: str = "",
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Busca en el código indexado (búsqueda semántica + grafo) y devuelve los
        fragmentos más relevantes con su archivo. Úsala para encontrar dónde está
        implementado algo cuando no sabes el nombre exacto de la clase o método.

        :param question: Qué buscar, en lenguaje natural o con nombres de clases/métodos.
        :param repo: Repositorio "organizacion/repositorio" (recomendado).
        :param branch: Rama (opcional).
        """

        await self._status(__event_emitter__, f"🔎 Buscando: {question[:60]}…")

        try:

            response = requests.post(
                self._url("/search-graph"),
                json={"query": question, "repo": repo or None, "branch": branch or None, "limit": 8},
                headers=self._headers(),
                timeout=60,
            )

            response.raise_for_status()

            results = response.json().get("results", [])

        except Exception as exc:

            await self._status(__event_emitter__, "Error en la búsqueda", True)

            return f"ERROR buscando en el código: {exc}"

        await self._status(__event_emitter__, f"{len(results)} resultados", True)

        if not results:

            return "Sin resultados. Prueba con otros términos o con el nombre de la clase."

        blocks = []

        for r in results:

            title = r.get("file_path", "?")

            if r.get("ast_name"):

                title += f" · {r.get('ast_type', '')} {r['ast_name']}"

            text = (r.get("text") or "")[:2500]

            blocks.append(f"### {title}\n{text}")

        return self._cap("\n\n".join(blocks))

    async def read_file(
        self,
        repo: str,
        file_path: str,
        branch: str = "",
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Lee el contenido completo de un archivo del repositorio (código, SQL, XML,
        properties...). Úsala cuando necesites ver un archivo entero, por ejemplo
        el .sql que ejecuta un DAO o la clase completa de un servicio.

        :param repo: Repositorio "organizacion/repositorio".
        :param file_path: Ruta del archivo dentro del repo, ej. src/main/java/.../FacturaDao.java
        :param branch: Rama (opcional; por defecto la configurada).
        """

        await self._status(__event_emitter__, f"📄 Leyendo {file_path}…")

        try:

            response = requests.post(
                self._url("/fetch-file"),
                json={
                    "repo": repo,
                    "file_path": file_path.strip().strip("`"),
                    "branch": branch or self.valves.default_branch,
                },
                headers=self._headers(),
                timeout=60,
            )

            response.raise_for_status()

            content = response.json().get("content") or ""

        except Exception as exc:

            await self._status(__event_emitter__, "No se pudo leer el archivo", True)

            return f"ERROR leyendo {file_path}: {exc}"

        await self._status(__event_emitter__, "Archivo leído", True)

        return self._cap(f"Archivo: {file_path}\n\n{content}")

    async def find_usages(
        self,
        name: str,
        repo: str = "",
        branch: str = "",
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Dice quién usa una clase, método o tabla: qué métodos la llaman, qué clases
        la inyectan o heredan de ella, y qué endpoint expone a quien la llama.
        Úsala para "¿dónde se usa X?", "¿quién llama a X?" o "¿qué impacto tiene cambiar X?".

        :param name: Nombre exacto de la clase, método o tabla.
        :param repo: Repositorio "organizacion/repositorio" (opcional).
        :param branch: Rama (opcional).
        """

        await self._status(__event_emitter__, f"🔗 Buscando usos de {name}…")

        try:

            data = self._get_json(
                f"/graph/usages/{name.strip().strip('`')}",
                {"repo": repo, "branch": branch, "limit": 150},
            )

        except Exception as exc:

            await self._status(__event_emitter__, "Error buscando usos", True)

            return f"ERROR buscando usos de {name}: {exc}"

        usages = (data or {}).get("usages", [])

        await self._status(__event_emitter__, f"{len(usages)} usos", True)

        if not usages:

            return f"No se encontraron usos de {name} en el grafo."

        lines = [f"Usos de {name} ({len(usages)}):"]

        for r in usages:

            line = (
                f"- {self._qualified(r.get('source_class'), r.get('source_name'))} "
                f"{r.get('rel_type')} {r.get('target_name')} "
                f"({r.get('file_path')}:{r.get('start_line')})"
            )

            if r.get("route"):

                line += f" [endpoint {r['route']}]"

            lines.append(line)

        lines.append(
            "Nota: las llamadas se resuelven por el tipo del receptor; las que no se pudieron "
            "inferir se enlazan por nombre y pueden incluir métodos homónimos de otras clases."
        )

        return self._cap("\n".join(lines))

    async def get_call_flow(
        self,
        name: str,
        repo: str = "",
        branch: str = "",
        depth: int = 4,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Muestra el flujo hacia abajo desde una clase o método: qué métodos llama,
        qué dependencias inyecta, qué archivos SQL ejecuta y qué tablas lee o escribe.
        Úsala para explicar "cómo funciona" un proceso de punta a punta.

        :param name: Nombre exacto de la clase o método de inicio (ej. un Controller).
        :param repo: Repositorio "organizacion/repositorio" (opcional).
        :param branch: Rama (opcional).
        :param depth: Cuántos saltos seguir (1 a 6).
        """

        await self._status(__event_emitter__, f"🧭 Siguiendo el flujo de {name}…")

        try:

            data = self._get_json(
                f"/graph/flow/{name.strip().strip('`')}",
                {"repo": repo, "branch": branch, "depth": depth},
                timeout=60,
            )

        except Exception as exc:

            await self._status(__event_emitter__, "Error obteniendo el flujo", True)

            return f"ERROR obteniendo el flujo de {name}: {exc}"

        edges = (data or {}).get("edges", [])

        await self._status(__event_emitter__, f"{len(edges)} pasos", True)

        if not edges:

            return f"No se encontró flujo desde {name} en el grafo."

        lines = [f"Flujo desde {name}:"]

        for r in edges:

            line = (
                f"- {self._qualified(r.get('source_class'), r.get('source'))} "
                f"{r.get('rel_type')} {self._qualified(r.get('target_class'), r.get('target'))}"
            )

            if r.get("rel_type") == "USES_SQL" and r.get("target_file"):

                line += f" ({r['target_file']})"

            lines.append(line)

        return self._cap("\n".join(lines))

    async def find_table_usage(
        self,
        table: str,
        repo: str = "",
        branch: str = "",
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Dice qué archivos SQL y qué métodos (con su clase) leen o escriben una tabla
        de base de datos. Úsala para "¿quién usa la tabla X?" o "¿dónde se inserta en X?".

        :param table: Nombre de la tabla, ej. FACTURA o dbo.FACTURA.
        :param repo: Repositorio "organizacion/repositorio" (opcional: sin él busca en todos).
        :param branch: Rama (opcional).
        """

        table = table.strip().split(".")[-1].strip("`[]\" ")

        await self._status(__event_emitter__, f"🗃️ Buscando uso de la tabla {table}…")

        try:

            data = self._get_json(f"/sql/table/{table}", {"repo": repo, "branch": branch})

        except Exception as exc:

            await self._status(__event_emitter__, "Error consultando la tabla", True)

            return f"ERROR consultando la tabla {table}: {exc}"

        usage = (data or {}).get("usage", [])

        await self._status(__event_emitter__, f"{len(usage)} usos", True)

        if not usage:

            return f"No se encontró uso de la tabla {table.upper()} en el grafo."

        lines = [f"Uso de la tabla {table.upper()}:"]

        for r in usage:

            if not r.get("method"):

                lines.append(f"- {r.get('access')} [{r.get('repo')}] en {r.get('sql_file')} (ningún método lo usa)")

                continue

            line = f"- {r.get('access')} [{r.get('repo')}] {self._qualified(r.get('class'), r.get('method'))}"

            if r.get("sql_file"):

                line += f" vía {r['sql_file']}"

            if r.get("file_path"):

                line += f" ({r['file_path']})"

            if r.get("route"):

                line += f" [endpoint {r['route']}]"

            lines.append(line)

        return self._cap("\n".join(lines))

    async def list_tables(
        self,
        repo: str,
        branch: str = "",
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Lista las tablas de base de datos que usa un repositorio, con cuántos
        SQL/métodos las leen y cuántos las escriben.

        :param repo: Repositorio "organizacion/repositorio".
        :param branch: Rama (opcional).
        """

        try:

            data = self._get_json("/sql/tables", {"repo": repo, "branch": branch})

        except Exception as exc:

            return f"ERROR listando tablas de {repo}: {exc}"

        tables = (data or {}).get("tables", [])

        if not tables:

            return f"No se encontraron tablas en {repo}."

        lines = [f"Tablas de {repo} ({len(tables)}):"]

        lines += [
            f"- {r.get('table')}: {r.get('readers', 0)} lecturas, {r.get('writers', 0)} escrituras"
            for r in tables
        ]

        return self._cap("\n".join(lines))

    async def list_endpoints(
        self,
        repo: str,
        branch: str = "",
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Lista los endpoints HTTP (Spring MVC / JAX-RS) que expone un repositorio,
        con el controller y el método que atiende cada ruta.

        :param repo: Repositorio "organizacion/repositorio".
        :param branch: Rama (opcional).
        """

        try:

            data = self._get_json("/endpoints", {"repo": repo, "branch": branch})

        except Exception as exc:

            return f"ERROR listando endpoints de {repo}: {exc}"

        endpoints = (data or {}).get("endpoints", [])

        if not endpoints:

            return f"No se encontraron endpoints HTTP en {repo}."

        lines = [f"Endpoints de {repo} ({len(endpoints)}):"]

        lines += [
            f"- {r.get('route')} → {self._qualified(r.get('class'), r.get('method'))} "
            f"({r.get('file_path')}:{r.get('start_line')})"
            for r in endpoints
        ]

        return self._cap("\n".join(lines))

    @staticmethod
    def _content_to_export(content, messages):
        """El texto que pasó el modelo o, si viene vacío, la última respuesta del asistente."""

        if content and content.strip():

            return content

        for message in reversed(messages or []):

            if message.get("role") == "assistant" and message.get("content"):

                value = message["content"]

                if isinstance(value, list):

                    value = " ".join(str(i.get("text", "")) for i in value if isinstance(i, dict))

                return value

        return ""

    async def _deliver(self, data, filename, content_type, user, emitter):
        """Guarda el archivo y muestra el enlace de descarga directo en el mensaje."""

        url = await _store_file(data, filename, content_type, (user or {}).get("id"))

        if url:

            link = f"📥 [Descargar {filename}]({url})"

        else:

            b64 = base64.b64encode(data).decode("utf-8")

            link = f'<a href="data:{content_type};base64,{b64}" download="{filename}">📥 Descargar {filename}</a>'

        # El enlace se agrega directo al mensaje; el modelo no tiene que copiarlo.
        if emitter:

            await emitter({"type": "message", "data": {"content": f"\n\n{link}\n\n"}})

    async def export_markdown(
        self,
        content: str = "",
        title: str = "",
        __messages__: Optional[list] = None,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Guarda un documento Markdown descargable (.md). Úsala cuando el usuario pida
        la información en un archivo .md o "markdown para descargar". Si lo pide en el
        mismo mensaje que la pregunta, primero redacta el documento completo en Markdown
        y pásalo en content; si content va vacío se exporta tu respuesta anterior.

        :param content: Documento Markdown completo a guardar (opcional).
        :param title: Nombre del archivo sin extensión (opcional).
        """

        text = self._content_to_export(content, __messages__)

        if not text:

            return "No hay contenido para exportar a Markdown."

        title = (title or "Tennis_Doc").strip().replace("/", "_").replace(" ", "_")

        await self._status(__event_emitter__, "📝 Generando Markdown…")

        await self._deliver(
            text.encode("utf-8"), f"{title}.md", "text/markdown", __user__, __event_emitter__
        )

        await self._status(__event_emitter__, "Markdown listo", True)

        return "Archivo .md generado y el enlace de descarga ya se mostró al usuario. Confírmalo en una frase."

    async def export_last_answer_pdf(
        self,
        title: str = "",
        content: str = "",
        __messages__: Optional[list] = None,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Genera un PDF descargable. Úsala cuando el usuario pida exportar, descargar o
        guardar en PDF. Si lo pide en el mismo mensaje que la pregunta, redacta primero
        el contenido completo y pásalo en content; si content va vacío se exporta tu
        respuesta anterior.

        :param title: Título del documento (opcional).
        :param content: Texto/Markdown a convertir en PDF (opcional).
        """

        text = self._content_to_export(content, __messages__)

        if not text:

            return "No hay una respuesta anterior para convertir a PDF."

        title = (title or "Respuesta").strip().replace("/", "_").replace(" ", "_")

        await self._status(__event_emitter__, "📄 Generando PDF…")

        try:

            response = requests.post(
                self._url("/pdf"),
                json={"title": title, "content": text},
                headers=self._headers(),
                timeout=60,
            )

            response.raise_for_status()

            pdf = response.content

        except Exception as exc:

            await self._status(__event_emitter__, "Error generando PDF", True)

            return f"ERROR generando el PDF: {exc}"

        await self._deliver(pdf, f"{title}.pdf", "application/pdf", __user__, __event_emitter__)

        await self._status(__event_emitter__, "PDF listo", True)

        return "PDF generado y el enlace de descarga ya se mostró al usuario. Confírmalo en una frase."


async def _maybe_await(value):
    """Open WebUI cambió estas funciones a async en versiones recientes: soporta ambas."""

    return await value if inspect.isawaitable(value) else value


async def _store_file(data, filename, content_type, user_id):
    """Guarda el archivo en Open WebUI y devuelve su URL (None si falla)."""

    try:

        from open_webui.models.files import FileForm, Files
        from open_webui.storage.provider import Storage

        file_id = str(uuid.uuid4())

        stored_name = f"{file_id}_{filename}"

        try:

            uploaded = Storage.upload_file(io.BytesIO(data), stored_name, {})

        except TypeError:

            uploaded = Storage.upload_file(io.BytesIO(data), stored_name)

        _, path = await _maybe_await(uploaded)

        saved = await _maybe_await(
            Files.insert_new_file(
                user_id,
                FileForm(
                    id=file_id,
                    filename=filename,
                    path=path,
                    meta={"name": filename, "content_type": content_type, "size": len(data)},
                ),
            )
        )

        if not saved:

            print(f"[TennisDoc Tools] Open WebUI no registró el archivo {filename}")

            return None

        return f"/api/v1/files/{file_id}/content"

    except Exception as exc:

        print(f"[TennisDoc Tools] No pude guardar el archivo en Open WebUI: {exc}")

        return None
