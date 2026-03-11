# OpenMemory — Aify Hybrid Memory System

Orchestrator for the hybrid memory stack. Manages 4 sub-containers via Docker SDK.
Based on the aify-container boilerplate.

## Architecture

```
openmemory-service (FastAPI orchestrator, port 8800)
  ├── qdrant      (vector store, port 6333)
  ├── neo4j       (graph store, port 7474/7687)
  ├── api         (mem0 FastAPI backend, port 8765)
  └── ui          (Next.js dashboard, port 3000→3100)
```

LLM and embedding are provided by **llamacpp-router** on the shared `aify-network`.

## Sub-Container Images

- `qdrant`: `qdrant/qdrant:v1.17.0` (pulled from Docker Hub)
- `neo4j`: `neo4j:5.26.4` (pulled from Docker Hub)
- `api`: `openmemory-api:latest` — build from `./mem0-fork/openmemory/api/Dockerfile`
- `ui`: `openmemory-ui:latest` — build from `./mem0-fork/openmemory/ui/`

## Build API/UI Images

```bash
docker build -t openmemory-api:latest -f ./mem0-fork/openmemory/api/Dockerfile ./mem0-fork
docker build -t openmemory-ui:latest --build-arg NEXT_PUBLIC_API_URL=http://localhost:8765 --build-arg NEXT_PUBLIC_USER_ID=steven ./mem0-fork/openmemory/ui
```

## Quick Start

```bash
docker network create aify-network 2>/dev/null || true
docker compose up -d --build
```

## Key Config

- `config/service.json` — sub-container definitions (qdrant, neo4j, api, ui)
- `.env` — service identity, ports, credentials
- All data persisted in named volumes (openmemory-qdrant-data, openmemory-neo4j-data, openmemory-db)

## External Dependencies

- `llamacpp-router-service:11434` — LLM inference + embeddings (must be running on aify-network)

## Endpoints

| Path | Purpose |
|------|---------|
| `:8800/health` | Orchestrator health |
| `:8800/info` | Service discovery (all containers, URLs) |
| `:8800/route/api/*` | Proxy to OpenMemory API |
| `:8800/route/ui/*` | Proxy to OpenMemory UI |
| `:8800/mcp/sse` | MCP SSE endpoint |
| `:8765` (direct) | OpenMemory API (via port mapping) |
| `:3100` (direct) | OpenMemory UI (via port mapping) |

## MCP Tools

Container management tools are built-in (list, start, stop, logs, GPU status).
Memory-specific MCP tools (search, add, delete, brain agent) are on the API sub-container at port 8765.

## Conventions

Same as aify-container boilerplate — see parent CLAUDE.md for full details.
- Named volumes for persistence
- Docker socket for container management
- Config precedence: env > service.json > defaults
