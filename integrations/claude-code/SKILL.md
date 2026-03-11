---
name: agentify-service
description: Manage containerized AI services with on-demand sub-containers
triggers:
  - when the user asks about the service or its containers
  - when container management is needed (start, stop, status)
  - when GPU allocation info is needed
  - when service data or operations are needed
tools:
  - service_info
  - service_health
  - list_containers
  - start_container
  - stop_container
  - gpu_status
  - container_logs
---

# Agentify Service Skill

You have access to a containerized service that orchestrates on-demand sub-containers (e.g., llama.cpp instances for different models).

## Available Tools

### Service
- `service_info` - Get service metadata, available containers, endpoints, and capabilities
- `service_health` - Check service health and dependency status

### Container Management
- `list_containers` - List all sub-containers with status, GPU, groups, URLs
- `start_container(name)` - Start a container (auto-resolves shared containers)
- `stop_container(name)` - Stop a running container
- `gpu_status` - GPU device allocation (which containers on which GPUs, memory fractions)
- `container_logs(name, tail)` - Get recent container logs

## Usage Flow

1. Call `service_info` or `list_containers` to see what's available
2. Call `start_container("name")` to start a specific container
3. Use the proxy URL `http://localhost:8800/route/{name}/{path}` to access the container's API
4. Containers auto-stop after idle timeout (configurable per container)

## Key Concepts

- **Groups**: Containers are logically grouped (e.g., "llm", "openmemory")
- **Shared containers**: Some containers share another's URL instead of running their own (saves GPU)
- **On-demand**: Containers start when first accessed via `/route/{name}/...`
- **GPU fractions**: Each container declares how much GPU memory it needs; the scheduler prevents over-subscription

## Proxy Pattern

Access sub-container APIs through the router:
```
POST http://localhost:8800/route/qwen/v1/chat/completions
POST http://localhost:8800/route/embed/embedding
GET  http://localhost:8800/route/qdrant/collections
```

## Installation

### MCP SSE (recommended):
```
claude mcp add agentify-service --transport sse http://localhost:8800/mcp/sse
```

### MCP stdio:
```
cd mcp/stdio && npm install
claude mcp add agentify-service -- node /absolute/path/to/mcp/stdio/server.js
```
