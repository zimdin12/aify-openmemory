"""
MCP Server - SSE Transport (runs inside the Docker container)

This MCP server is mounted into the FastAPI app and accessible via SSE at:
  http://<host>:<port>/mcp/sse

AI agents building on this template should:
1. Register tools that expose the service's core functionality
2. Each tool should be self-documenting with clear descriptions
3. Tools should handle errors gracefully and return helpful messages

The tools registered here become available to any MCP-compatible client
(Claude Code, OpenClaw, Cursor, etc.)
"""

import logging
from contextvars import ContextVar

import httpx
from mcp.server.fastmcp import FastMCP

from service.config import get_config

logger = logging.getLogger(__name__)

# Context variables for per-request user/client tracking
user_id_var: ContextVar[str] = ContextVar("user_id", default="default")
client_name_var: ContextVar[str] = ContextVar("client_name", default="unknown")

# Create MCP server instance
config = get_config()
mcp_server = FastMCP(
    config.name,
    description=config.description,
)

# Reference to the FastAPI app (set during setup)
_app = None


def _get_manager():
    """Get container manager from app state."""
    if _app is None:
        return None
    return getattr(_app.state, "container_manager", None)


# ---------------------------------------------------------------------------
# Service Tools
# ---------------------------------------------------------------------------

@mcp_server.tool()
async def service_info() -> dict:
    """Get information about this service, its capabilities, managed containers, and available tools."""
    cfg = get_config()
    result = {
        "name": cfg.name,
        "version": cfg.version,
        "description": cfg.description,
        "status": "running",
    }
    manager = _get_manager()
    if manager:
        result["containers"] = manager.list_containers()
        result["groups"] = manager.get_groups()
    return result


@mcp_server.tool()
async def service_health() -> dict:
    """Check if the service and its dependencies are healthy."""
    checks = {}
    manager = _get_manager()
    if manager:
        checks["docker"] = "connected" if manager.docker else "unavailable"
        checks["containers"] = {
            name: state.status.value
            for name, state in manager.states.items()
        }
    return {"status": "healthy", "checks": checks}


# ---------------------------------------------------------------------------
# Container Management Tools
# ---------------------------------------------------------------------------

@mcp_server.tool()
async def list_containers() -> dict:
    """List all managed sub-containers, their status, GPU allocation, and URLs."""
    manager = _get_manager()
    if not manager:
        return {"error": "No container manager configured"}
    return {
        "containers": manager.list_containers(),
        "groups": manager.get_groups(),
    }


@mcp_server.tool()
async def start_container(name: str) -> dict:
    """
    Start a managed sub-container by name. If already running, returns current state.
    If the container is shared with another, starts the target container instead.
    """
    manager = _get_manager()
    if not manager:
        return {"error": "No container manager configured"}
    if name not in manager.definitions:
        return {"error": f"Unknown container: {name}", "available": list(manager.definitions.keys())}
    try:
        state = await manager.start_container(name)
        return {"status": state.status.value, "url": state.internal_url}
    except Exception as e:
        return {"error": str(e)}


@mcp_server.tool()
async def stop_container(name: str) -> dict:
    """Stop a running sub-container by name."""
    manager = _get_manager()
    if not manager:
        return {"error": "No container manager configured"}
    if name not in manager.definitions:
        return {"error": f"Unknown container: {name}"}
    try:
        await manager.stop_container(name)
        return {"status": "stopped", "name": name}
    except Exception as e:
        return {"error": str(e)}


@mcp_server.tool()
async def gpu_status() -> dict:
    """Get GPU device allocation status showing which containers are using which GPUs."""
    manager = _get_manager()
    if not manager:
        return {"error": "No container manager configured"}
    return manager.gpu.get_status()


@mcp_server.tool()
async def container_logs(name: str, tail: int = 50) -> str:
    """Get recent logs from a managed sub-container."""
    manager = _get_manager()
    if not manager:
        return "No container manager configured"
    if name not in manager.definitions:
        return f"Unknown container: {name}"
    return manager.get_container_logs(name, tail=tail)


# ---------------------------------------------------------------------------
# TODO: Add your service-specific tools below
# ---------------------------------------------------------------------------
#
# @mcp_server.tool()
# async def generate_text(prompt: str, max_tokens: int = 512, container: str = "qwen") -> str:
#     """Generate text using a specific LLM container."""
#     manager = _get_manager()
#     url = manager.resolve_url(container)
#     async with httpx.AsyncClient(timeout=120.0) as client:
#         resp = await client.post(f"{url}/completion", json={
#             "prompt": prompt, "n_predict": max_tokens
#         })
#         return resp.json()["content"]


# ---------------------------------------------------------------------------
# MCP Resources (optional) - Expose data as browsable resources
# ---------------------------------------------------------------------------
# @mcp_server.resource("config://service")
# async def get_service_config() -> str:
#     """Expose service configuration as a browsable MCP resource."""
#     ...


# ---------------------------------------------------------------------------
# MCP Prompts (optional) - Predefined prompt templates
# ---------------------------------------------------------------------------
# @mcp_server.prompt()
# async def analyze_prompt(topic: str) -> str:
#     """A prompt template for analyzing topics using this service."""
#     ...


def setup_mcp_server(app):
    """Mount the MCP server onto the FastAPI app."""
    global _app
    _app = app
    cfg = get_config()

    # Get the SSE app from FastMCP
    sse_app = mcp_server.sse_app()

    # Mount under the configured prefix
    app.mount(cfg.mcp_path_prefix, sse_app)

    logger.info(
        f"MCP SSE server mounted at {cfg.mcp_path_prefix}/ "
        f"- Connect at {cfg.mcp_path_prefix}/sse"
    )
