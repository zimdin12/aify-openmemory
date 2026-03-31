"""
Health and service info endpoints.
Used by Docker healthchecks and by AI agents to discover the service.
"""

from fastapi import APIRouter, Request
from service.config import get_config

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    """Health check endpoint. Returns 200 if service is running."""
    return {"status": "healthy"}


@router.get("/ready")
async def ready(request: Request):
    """Readiness check. Verifies all components are initialized."""
    checks = {}
    all_ok = True

    # Container manager
    manager = getattr(request.app.state, "container_manager", None)
    if manager is not None:
        checks["container_manager"] = "initialized"
        checks["docker"] = "connected" if manager.docker else "unavailable"
        if not manager.docker:
            all_ok = False

    # Database
    try:
        from service.database import SessionLocal
        db = SessionLocal()
        db.execute("SELECT 1" if hasattr(db, 'execute') else None)
        db.close()
        checks["database"] = "connected"
    except Exception:
        checks["database"] = "unavailable"
        all_ok = False

    # Memory client (Qdrant + Neo4j)
    try:
        from service.memory.client import get_memory_client
        client = get_memory_client()
        checks["memory_client"] = "connected" if client else "unavailable"
        if not client:
            all_ok = False
    except Exception:
        checks["memory_client"] = "unavailable"
        all_ok = False

    # LLM backend
    try:
        from service.memory.llm import _backend
        if _backend:
            checks["llm_backend"] = f"{_backend.mode}:{_backend.llm_model}"
        else:
            checks["llm_backend"] = "not_initialized"
    except Exception:
        checks["llm_backend"] = "unavailable"

    return {"status": "ready" if all_ok else "degraded", "checks": checks}


@router.get("/info")
async def info(request: Request):
    """
    Service discovery endpoint for AI agents.
    Returns everything an agent needs to use this service.
    """
    config = get_config()

    # Use request host for URLs so they work from other containers/machines
    host = request.headers.get("host", f"localhost:{config.port}")
    base = f"http://{host}"

    response = {
        "name": config.name,
        "version": config.version,
        "description": config.description,
        "endpoints": {
            "api": f"{base}/api/v1",
            "docs": f"{base}/docs",
            "openapi": f"{base}/openapi.json",
            "health": f"{base}/health",
            "ready": f"{base}/ready",
        },
        "integrations": {
            "mcp_sse": f"{base}{config.mcp_path_prefix}/sse" if config.mcp_enabled else None,
            "mcp_stdio": "See mcp/stdio/ directory for host-side MCP server",
            "claude_code_skill": "See integrations/claude-code/SKILL.md",
            "openclaw_plugin": "See integrations/openclaw/",
            "open_webui_tool": "See integrations/open-webui/",
        },
    }

    manager = getattr(request.app.state, "container_manager", None)
    if manager is not None:
        response["endpoints"]["containers"] = f"{base}/api/v1/containers"
        response["endpoints"]["gpu"] = f"{base}/api/v1/gpu"
        response["endpoints"]["route"] = f"{base}/route/{{container_name}}/{{path}}"
        response["containers"] = manager.list_containers()
        response["groups"] = manager.get_groups()

    return response
