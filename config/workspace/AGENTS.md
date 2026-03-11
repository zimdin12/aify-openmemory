# Agent Operational Guidelines

## Service Context

This is an aify container service - a FastAPI orchestrator that manages Docker sub-containers on demand. It runs at port 8800 by default.

## Endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Health check (Docker healthcheck uses this) |
| `GET /ready` | Readiness check with component status |
| `GET /info` | Full service discovery (containers, endpoints, integrations) |
| `GET /docs` | OpenAPI documentation |
| `GET /api/v1/containers` | List all containers with status |
| `POST /api/v1/containers/{name}/start` | Start a container |
| `POST /api/v1/containers/{name}/stop` | Stop a container |
| `GET /api/v1/containers/{name}/logs` | Container logs |
| `GET /api/v1/gpu` | GPU allocation status |
| `ANY /route/{name}/{path}` | Proxy to sub-container (auto-starts) |
| `GET /mcp/sse` | MCP SSE endpoint |

## Files to Modify When Building a Service

**Must modify:**
- `service/routers/api.py` - Your domain endpoints
- `mcp/sse_server.py` - MCP tools (SSE, in-container)
- `mcp/stdio/server.js` - MCP tools (stdio, host-side)
- `config/service.example.json` - Container definitions

**Should modify:**
- `integrations/claude-code/SKILL.md` - Tool list and usage
- `integrations/openclaw/index.ts` - Plugin tools and hooks
- `integrations/open-webui/tool.py` - Tool methods
- `service/requirements.txt` - Python dependencies

**Don't modify unless necessary:**
- `service/main.py` - Only touch the lifespan for resource init
- `service/config.py` - Extend via service.json custom keys
- `service/containers/` - Container orchestration internals
- `service/routers/health.py` - Only add readiness checks
- `service/routers/containers.py` - Container management API

## Persistence

- Orchestrator data: `/data` directory (Docker named volume)
- Sub-container data: Named volumes per container (defined in config)
- Never store persistent data in the container filesystem

## Config

- `.env` for deployment (ports, project name, resources)
- `config/service.json` for service definition (containers, custom settings)
- Environment variables override everything

## Sub-Container Access

From API endpoints, access containers via `request.app.state.container_manager`:
```python
manager = request.app.state.container_manager
url = manager.resolve_url("container-name")  # Returns http://hostname:port
state = manager.states["container-name"]     # Check .status, .internal_url
await manager.start_container("name")        # Ensure running
```

Or let clients use the proxy directly at `/route/{name}/{path}`.
