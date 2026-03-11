# Agentify Container

A boilerplate for building AI agent-friendly Docker services with on-demand sub-container orchestration.

Services built on this template are automatically accessible to AI agents via REST API and MCP protocol, with integrations for Claude Code, OpenClaw, and Open WebUI. The built-in container manager can spin up and route requests to sub-containers (like llama.cpp instances) on demand, with GPU-aware scheduling and automatic idle shutdown.

## Quick Start

```bash
git clone https://github.com/zimdin12/agentify-container.git
cd agentify-container
bash setup.sh          # Copies config templates

# Edit .env (ports, project name, resources)
# Edit config/service.json (container definitions) - see config/examples/

docker compose up -d --build
bash scripts/test-endpoints.sh
```

## What You Get

| Component | Location | Purpose |
|-----------|----------|---------|
| FastAPI orchestrator | `service/` | REST API + container management + streaming proxy |
| Container manager | `service/containers/` | Docker SDK-based lifecycle management with GPU tracking |
| MCP SSE server | `mcp/sse_server.py` | In-container MCP with container mgmt tools |
| MCP stdio server | `mcp/stdio/` | Host-side MCP for Claude Code CLI |
| Claude Code skill | `integrations/claude-code/` | Skill with all tools documented |
| OpenClaw plugin | `integrations/openclaw/` | Plugin with tools and hooks |
| Open WebUI tool | `integrations/open-webui/` | Tool for chat interfaces |
| Config examples | `config/examples/` | Ready-to-use configs for common setups |

## Container Management

Define sub-containers in `config/service.json`. They start on demand and auto-stop after idle.

### Simple example (one LLM):

```json
{
  "containers": {
    "defaults": {
      "image": "ghcr.io/ggerganov/llama.cpp:server-cuda",
      "internal_port": 8080,
      "volumes": { "llm-models": "/models" },
      "gpu": { "device_ids": ["0"] },
      "idle_timeout_seconds": 300
    },
    "definitions": {
      "llm": {
        "command": ["--model", "/models/my-model.gguf", "--port", "8080", "--gpu-layers", "99"]
      }
    }
  }
}
```

### Multi-LLM router (3 models, GPU scheduling):

See `config/examples/llama-cpp-router.json`

### Full stack with shared containers:

See `config/examples/openclaw-full-stack.json` - demonstrates how openmemory reuses openclaw's LLM containers instead of running its own.

### Key features:

- **On-demand**: First request to `/route/{name}/...` starts the container
- **Auto-stop**: Containers shut down after configurable idle timeout
- **GPU scheduling**: Memory fraction tracking prevents over-subscription
- **Shared containers**: `"shared_with": "other-name"` reuses another container
- **Streaming proxy**: Full SSE/chunked support for LLM inference
- **Health monitoring**: Auto-restart on health check failure
- **Groups**: Logical grouping for organizational clarity

### Accessing sub-containers:

```bash
# Via proxy (auto-starts if needed):
curl http://localhost:8800/route/qwen/v1/chat/completions -d '{"messages":[...]}'

# Management API:
curl http://localhost:8800/api/v1/containers           # List all
curl -X POST http://localhost:8800/api/v1/containers/qwen/start
curl -X POST http://localhost:8800/api/v1/containers/qwen/stop
curl http://localhost:8800/api/v1/gpu                   # GPU status
```

## For AI Agents

See [CLAUDE.md](CLAUDE.md) for build instructions. See [docs/AGENT_GUIDE.md](docs/AGENT_GUIDE.md) for step-by-step implementation guide.

## Configuration

```
.env                    -> Deployment: ports, project name, resources, credentials
config/service.json     -> Service: container definitions, custom settings
```

Environment variables always override service.json. All ports and resource limits are overrideable.

## MCP Integration

```bash
# SSE (recommended, no host install):
claude mcp add my-service --transport sse http://localhost:8800/mcp/sse

# stdio (alternative):
cd mcp/stdio && npm install
claude mcp add my-service -- node /path/to/mcp/stdio/server.js
```

## Security Note

The orchestrator container mounts `/var/run/docker.sock` to manage sub-containers. This grants Docker API access. For production, consider using a [Docker socket proxy](https://github.com/Tecnativa/docker-socket-proxy) to restrict API calls.

## Roadmap

Planned improvements and features for future development. Contributions welcome.

### Security

- **API key middleware** - The `API_KEY` config field exists but is not enforced yet. Add a FastAPI dependency that checks `Authorization: Bearer <key>` header when `config.api_key` is non-empty. This is the minimum viable auth for production use.
- **Rate limiting** - Add `slowapi` or similar to protect against runaway agent loops. AI agents can make many rapid requests; without rate limiting a misconfigured agent could overwhelm the service or burn through GPU time.
- **Docker socket proxy** - Instead of mounting the raw Docker socket, integrate [Tecnativa/docker-socket-proxy](https://github.com/Tecnativa/docker-socket-proxy) as a default sub-service. Restrict API access to only containers, networks, and volumes endpoints (block exec, images push, system).
- **Secret management** - Container definitions can include environment variables with credentials (e.g., `NEO4J_AUTH`). Add a pattern for marking sensitive env vars so they are redacted from `/info` and `list_containers` responses. Consider Docker secrets or a `.secrets.env` file that is never exposed via API.

### Observability

- **Container event stream** - Add an SSE endpoint (e.g., `/api/v1/events`) that streams container lifecycle events (started, stopped, failed, health_changed) in real-time. Useful for dashboards and for AI agents that need to react to state changes without polling.
- **Metrics endpoint** - Add a `/metrics` endpoint (Prometheus format) with counters for: requests proxied per container, container start/stop counts, GPU utilization fractions, proxy latency histograms, health check results.
- **Structured logging improvements** - Current JSON logging is basic. Consider adding request ID tracing, container name context, and log correlation across proxy requests to sub-containers.

### Container Management

- **Container profiles/presets** - Define named profiles (e.g., "low-memory", "high-throughput") that bundle resource limits, GPU settings, and parallelism configs. An agent could say `start_container("qwen", profile="low-memory")` to override defaults.
- **Warm standby mode** - Instead of fully stopping idle containers, pause them (Docker pause) to preserve loaded models in memory. Resume is near-instant vs. cold start. Useful for models that take 30+ seconds to load.
- **Container dependencies** - Allow containers to declare dependencies (e.g., openmemory depends on qdrant and embed-llm). The manager would start dependencies first and stop dependents before stopping a dependency.
- **Image pre-pull on setup** - Add a `make pull` target and `/api/v1/containers/pull-all` endpoint that pre-pulls all configured images. First-start latency is currently dominated by image pull time.
- **Container resource monitoring** - Query Docker stats API to report actual CPU/memory/GPU usage per container, not just configured limits. Show in `list_containers` response.

### Developer Experience

- **GitHub Actions CI** - Add `.github/workflows/test.yml` that builds the image, starts the service, and runs `test-endpoints.sh`. Include a matrix for Python 3.12/3.13. This validates the template works on every push.
- **Response body validation in tests** - `test-endpoints.sh` currently only checks HTTP status codes. Add `jq` assertions to verify `/health` returns `{"status":"healthy"}`, `/info` has the expected structure, and container management endpoints return valid schemas.
- **Template instantiation script** - Add a `scripts/init-service.sh <name>` that renames the project: updates `.env.example`, `service.example.json`, `package.json`, `SKILL.md`, and `openclaw.plugin.json` with the given service name. Saves the first 5 minutes of manual find-and-replace.
- **VS Code devcontainer** - Add `.devcontainer/devcontainer.json` for one-click dev environment setup with Python, Node, Docker-in-Docker, and the MCP stdio server pre-configured.

### Integration

- **OpenClaw plugin API verification** - The TypeScript plugin (`integrations/openclaw/index.ts`) uses assumed interfaces (`PluginContext`, etc.). These should be verified against the actual OpenClaw plugin API as it stabilizes. The plugin may need adaptation.
- **MCP Streamable HTTP transport** - The MCP ecosystem is moving toward Streamable HTTP as the recommended transport (replacing SSE). Add support alongside SSE for forward compatibility.
- **A2A (Agent-to-Agent) protocol** - Add an A2A endpoint so this service can be discovered and used by agents following the Agent-to-Agent protocol specification.
- **OpenAI-compatible API layer** - For services that wrap LLMs (like llama.cpp), add an optional OpenAI-compatible endpoint layer (`/v1/chat/completions`, `/v1/embeddings`) that routes to the appropriate container. This makes the service a drop-in replacement for OpenAI API in any tool.

### Architecture

- **Multi-node support** - Currently all containers run on the same Docker host. For scaling, add support for Docker Swarm or remote Docker hosts so containers can be distributed across machines with different GPU configurations.
- **Config hot-reload** - Watch `config/service.json` for changes and apply them without restart. New containers would be added to the manager, removed ones would be stopped, and changed ones would be flagged for restart.
- **Backup/restore** - Add endpoints to export and import the full service state: container definitions, named volume data, and configuration. Useful for migrating between hosts or disaster recovery.

## License

MIT
