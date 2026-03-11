# Aify OpenMemory

Hybrid memory system (vector + graph + temporal) for AI agents. Built on a [mem0](https://github.com/mem0ai/mem0) fork with significant enhancements for local LLM compatibility.

Orchestrates 4 sub-containers via Docker SDK, based on the [aify-container](https://github.com/zimdin12/aify-container) boilerplate.

## Architecture

| Container | Purpose | Port |
|-----------|---------|------|
| `openmemory-api` | FastAPI memory API + MCP server | 8765 |
| `openmemory-ui` | Next.js dashboard | 3100 |
| `openmemory-qdrant` | Vector store (1024d cosine) | 6333 |
| `openmemory-neo4j` | Graph store (APOC enabled) | 8474/8687 |

LLM inference and embeddings are provided externally via [aify-llamacpp-router](https://github.com/zimdin12/aify-llamacpp-router).

## Quick Start

```bash
git clone --recurse-submodules https://github.com/zimdin12/aify-openmemory.git
cd aify-openmemory
cp .env.example .env
# Edit .env: set LLM_URL, EMBEDDING_URL, ports

docker compose up -d --build
```

## Memory Sources

- **Vector** (Qdrant): Semantic similarity search over extracted facts
- **Graph** (Neo4j): Entity-relationship knowledge graph with typed nodes
- **Temporal** (SQLite): Time-ordered memory with metadata

## API

### v1 (7 MCP tools)
Direct tools: `search_memory`, `add_memories`, `delete_memories`, `get_related_memories`, `conversation_memory`, `list_memories`, `delete_all_memories`

### v2 Brain Agent (1 tool)
Single natural language endpoint: `POST /api/v1/brain` — agent autonomously searches, stores, deletes, and updates across all 3 databases.

## Key Fork Changes

See [mem0-fork/FORK_CHANGES.md](mem0-fork/FORK_CHANGES.md) for full technical changelog. Highlights:

- JSON-based graph extraction (replaces unreliable tool calling)
- Per-model sampling configs (qwen3, qwen3.5, ministral, GLM families)
- Hub-preferring relationship flipping
- Self-referential edge guards (fuzzy + element-ID)
- User reference normalization (prompt + regex + core mem0)
- Dedup thresholds (0.85 semantic / 0.95 exact)
- Graph cleanup on memory delete

## Configuration

```
.env                    -> Ports, LLM URLs, credentials
config/service.json     -> Container definitions, custom settings
```

Environment variables override service.json. Key env vars:
- `LLM_URL` / `EMBEDDING_URL` — llama.cpp endpoints
- `LLM_MODEL` / `EMBEDDING_MODEL` — model names
- `NEO4J_URI` / `NEO4J_AUTH` — graph database
- `QDRANT_HOST` / `QDRANT_PORT` — vector database

## License

MIT
