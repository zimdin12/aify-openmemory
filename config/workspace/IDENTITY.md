# Service Identity

## Name
aify-openmemory - Hybrid memory system orchestrator

## What This Service Does
Orchestrates the OpenMemory hybrid memory stack: Qdrant (vector), Neo4j (graph), SQLite (temporal/metadata), and the mem0-based FastAPI API. Manages 4 sub-containers via Docker SDK.

## Core Capabilities
- Hybrid search across vector, graph, and temporal stores
- Brain agent (v2) — natural language memory operations
- 7 v1 MCP tools for direct memory access
- MCP SSE server for AI agent integration
- REST API with full CRUD + graph endpoints
- On-demand sub-container management with health monitoring

## How AI Agents Use This
1. Connect via MCP SSE at port 8765
2. Use `memory_agent` (v2) for natural language memory ops
3. Or use specific v1 tools (search_memory, add_memories, etc.)
4. Orchestrator at port 8800 manages container lifecycle
