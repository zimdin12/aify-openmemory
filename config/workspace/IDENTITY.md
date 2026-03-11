# Service Identity

## Template
Agentify Container - AI agent-friendly Docker service orchestrator

## What This Service Does
Runs as a FastAPI application inside Docker. Manages sub-containers (LLMs, databases, other services) on demand via the Docker SDK. Provides a unified API and MCP interface for AI agents to discover, start, stop, and use sub-containers.

## Core Capabilities
- REST API with OpenAPI documentation at /docs
- MCP server (SSE + stdio) for AI agent tool access
- On-demand Docker sub-container orchestration
- GPU-aware scheduling with memory fraction tracking
- Streaming reverse proxy for LLM inference (SSE/chunked)
- Automatic idle shutdown and health monitoring
- Container sharing to avoid duplicate resource usage

## How AI Agents Use This
1. Call /info or list_containers to discover available containers
2. Start containers via API or MCP tools (or just call /route/{name}/... which auto-starts)
3. Access container APIs through the proxy at /route/{name}/{path}
4. Containers auto-stop after idle, freeing GPU memory for others

## Customization
Built as a template. AI agents building on this should:
- Define containers in config/service.json
- Add domain endpoints in service/routers/api.py
- Register MCP tools in mcp/sse_server.py
- Update platform integrations in integrations/
