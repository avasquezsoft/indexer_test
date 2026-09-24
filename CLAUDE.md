# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`AGENTS.md` is the detailed reference (endpoints, env vars, chunking rules, Neo4j schema). Read it first; this file only adds what it misses or gets wrong.

## Commands

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8001 --reload   # needs env vars from .env.example
docker compose up -d                                   # full stack

# Java parser (parser_java/ is a separate git repo, nested and untracked here)
cd parser_java && mvn package -DskipTests
java -jar target/javaparser-service-1.0-SNAPSHOT-jar-with-dependencies.jar   # POST /parse, GET /health on :8080
```

No tests, linter, or CI exist. Verify changes by running the service and hitting `/health`, `/debug/files`, `/debug/chunks`.

## Deployment gotchas

- The `indexer` and `javaparser` containers `git clone` their code from `INDEXER_REPO_URL` / `PARSER_JAVA_REPO_URL` on every start. Local edits do nothing in Docker until pushed; restart the container to pick them up.
- The indexer runs with `--workers 3`, so in-process state (caches, globals) is not shared between requests.
- `docker-compose.yml` embeds an old **inline copy** of the Open WebUI filter under `configs.tennis_doc_rag` (v1.6). It is not what runs: `tennis_doc_rag.py` is pasted into Open WebUI by hand.
- The indexer image has no `git` binary and `apt-get` fails on the host's network; use `dulwich` (already in `requirements.txt`) for any git work.
- Changes to entity IDs, node/relation types or chunk payloads require reindexing the repos.

## Architecture

Three layers:

1. **Indexing** (`main.py:index_repo`): shallow-clone the repo with dulwich (`repo_clone.py`, one clone per repo+branch in `CLONE_BASE_DIR`) → list files via GitHub API (minified `*.min.js/css` skipped) → read each file from the clone (API as fallback) → `ast_parser.py` (JavaParser service for `.java`, Tree-sitter otherwise) → `code_links.py` adds SQL/table/HTTP-route links → nodes flushed to Neo4j and chunks to Qdrant every 500 entities → **all relations created at the end** (`graph_store.upsert_relations`) so none are lost across batches. `email_notifier.py` mails the result if `EMAIL_*` vars are set.
2. **Retrieval** (FastAPI endpoints): `/search` (vector), `/search-augmented` (vector → whole files), `/search-graph` (`rag_engine.py`: vector + graph expansion + exact-name keyword), `/search-clone` (ranked search over the local clone; large Java files are condensed to signatures + relevant method bodies). Graph navigation: `/graph/usages`, `/graph/flow`, `/sql/table`, `/sql/tables`, `/endpoints`.
3. **Open WebUI plugins**. These run inside Open WebUI, not the indexer, and talk to it over HTTP (`TENNIS_DOC_INDEXER_URL`, default `http://indexer:8001`):
   - `tennis_doc_rag.py`: a filter. `inlet` runs the retrieval chain in parallel (vector, graph, clone, plus a "graph map" of usages/flow/tables/endpoints chosen by the question's intent) and injects it as a system message; it also handles text commands (`indexa`, `grafo`, `repos`, PDF/markdown export). It must never raise: on any error it returns `body` unchanged.
   - `tennis_doc_tools.py`: function-calling Tools for the model to explore on its own (`search_code`, `read_file`, `find_usages`, `get_call_flow`, `find_table_usage`, `list_tables`, `list_endpoints`) plus the admin actions. When this Tool is active in a chat (matched via the filter's `tools_id` valve), the filter turns off its text commands and skips RAG for command messages.
   - Both files are pasted by hand into Open WebUI, so each must stay self-contained (only `requests` + `pydantic`); duplicated helpers between them are intentional.

Entity IDs are `{repo}:{branch}:{file_path}:{type}:{name}`. Qdrant and Neo4j data are keyed by repo+branch, so a change to ID format or payload fields needs a reindex.

## Conventions

- All code, comments, logs, and user-facing messages are in Spanish.
- Errors from external services (Qdrant, Neo4j, OpenRouter, GitHub) are logged and swallowed. They must never break the chat flow.
- All config comes from env vars in `config.py`. When you add a var, also add it to `.env.example` and to the `docker-compose.yml` service env.
