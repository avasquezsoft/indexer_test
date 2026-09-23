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
- `docker-compose.yml` embeds an **inline copy** of the Open WebUI filter under `configs.tennis_doc_rag` (v1.6), while `tennis_doc_rag.py` is v3.0. They have drifted. Editing the `.py` file does not change what compose mounts; the filter is usually pasted into Open WebUI by hand.

## Architecture

Three layers:

1. **Indexing** (`main.py:index_repo`): GitHub App token → list files → `ast_parser.py` (Tree-sitter, or HTTP to the Java service for `.java`) → entities + `chunker.py` chunks → Java files get referenced `.sql` files inlined → batched flush to Neo4j (`graph_store.py`) and Qdrant (`embedder.py` + `qdrant_store.py`). `repo_clone.py` also keeps a local clone per repo in `CLONE_BASE_DIR` (default `/repos`). `email_notifier.py` sends an SMTP mail when indexing ends if `EMAIL_*` vars are set.
2. **Retrieval** (FastAPI endpoints): `/search` (vector), `/search-augmented` (vector → whole files), `/search-graph` (`rag_engine.py`: vector + graph expansion + exact-name keyword), `/search-clone` (keyword grep over the local clone, used as a fallback).
3. **Open WebUI plugins**. These run inside Open WebUI, not the indexer, and talk to it over HTTP (`TENNIS_DOC_INDEXER_URL`, default `http://indexer:8001`):
   - `tennis_doc_rag.py`: a filter. `inlet` runs the retrieval chain and injects context as a system message; it also handles text commands (`indexa`, `grafo`, `repos`, PDF/markdown export). It must never raise: on any error it returns `body` unchanged.
   - `tennis_doc_tools.py`: the same actions as function-calling Tools. When this Tool is active in a chat (matched via the filter's `tools_id` valve), the filter turns off its text commands so nothing runs twice.

Entity IDs are `{repo}:{branch}:{file_path}:{type}:{name}`. Qdrant and Neo4j data are keyed by repo+branch, so a change to ID format or payload fields needs a reindex.

## Conventions

- All code, comments, logs, and user-facing messages are in Spanish.
- Errors from external services (Qdrant, Neo4j, OpenRouter, GitHub) are logged and swallowed. They must never break the chat flow.
- All config comes from env vars in `config.py`. When you add a var, also add it to `.env.example` and to the `docker-compose.yml` service env.
