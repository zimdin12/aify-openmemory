"""
OpenMemory MCP Server — SSE Transport

Unified MCP server exposing both memory tools and container management tools.
Memory tools are ported from the original openmemory MCP server.
Container tools are from the agentify-container base.

Connect at: http://<host>:<port>/mcp/{client_name}/sse/{user_id}
"""

import datetime
import json
import logging
import uuid
from contextvars import ContextVar

import httpx
from fastapi import Request
from fastapi.routing import APIRouter
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport

from service.config import get_config

logger = logging.getLogger(__name__)

# Context variables for per-request user/client tracking
user_id_var: ContextVar[str] = ContextVar("user_id", default="default")
client_name_var: ContextVar[str] = ContextVar("client_name", default="unknown")

# Create MCP server instance
config = get_config()
mcp_server = FastMCP("openmemory-mcp-server")

# Reference to the FastAPI app (set during setup)
_app = None

# MCP router for SSE endpoints
mcp_router = APIRouter(prefix="/mcp")

# SSE transport
sse = SseServerTransport("/mcp/messages/")


def _get_manager():
    """Get container manager from app state."""
    if _app is None:
        return None
    return getattr(_app.state, "container_manager", None)


# ===========================================================================
# Memory Tools
# ===========================================================================

@mcp_server.tool(description="Store new memories. Send facts as one fact per line — each line should be a complete, self-contained statement that includes the subject (person, project, or entity name). The system deduplicates against existing memories and only stores truly new information.")
async def add_memories(text: str) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"

    try:
        from service.database import SessionLocal
        from service.database.models import Memory, MemoryState, MemoryStatusHistory
        from service.database.utils import get_user_and_app
        from service.memory.enhanced import enhanced_memory_manager

        db = SessionLocal()
        try:
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            if not app.is_active:
                return f"Error: App {app.name} is currently paused."

            addition_result = enhanced_memory_manager.smart_add_memory(
                text, uid,
                metadata={"source_app": "openmemory", "mcp_client": client_name},
                client_name=client_name,
            )

            response = {
                "status": addition_result.status,
                "summary": addition_result.summary,
                "added_memories": addition_result.added_memories,
                "related_memories": addition_result.related_memories[:3],
                "insights": f"Processed {len(addition_result.added_memories)} new memories, "
                           f"found {len(addition_result.related_memories)} related memories, "
                           f"skipped {len(addition_result.skipped_facts)} duplicate facts."
            }

            for memory_data in addition_result.added_memories:
                if 'id' in memory_data:
                    memory_id = uuid.UUID(memory_data['id'])
                    memory = db.query(Memory).filter(Memory.id == memory_id).first()
                    if not memory:
                        memory = Memory(
                            id=memory_id, user_id=user.id, app_id=app.id,
                            content=memory_data.get('memory', text),
                            state=MemoryState.active,
                        )
                        db.add(memory)
                        db.add(MemoryStatusHistory(
                            memory_id=memory_id, changed_by=user.id,
                            old_state=MemoryState.active, new_state=MemoryState.active,
                        ))

            db.commit()
            return json.dumps(response, indent=2)
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error adding to memory: {e}")
        return f"Error adding to memory: {e}"


@mcp_server.tool(description="Search memories using hybrid vector + graph + temporal search. Returns up to 10 results per call. Use offset to paginate.")
async def search_memory(query: str, offset: int = 0) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)
    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"

    try:
        from service.database import SessionLocal
        from service.database.models import Memory, MemoryAccessLog
        from service.database.utils import get_user_and_app
        from service.database.permissions import check_memory_access_permissions
        from service.memory.enhanced import enhanced_memory_manager

        db = SessionLocal()
        try:
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            fetch_limit = 10 + offset
            search_results = enhanced_memory_manager.hybrid_search(query, uid, limit=fetch_limit)

            user_memories = db.query(Memory).filter(Memory.user_id == user.id).all()
            accessible_memory_ids = {str(memory.id) for memory in user_memories if check_memory_access_permissions(db, memory, app.id)}

            filtered_results = []
            for result in search_results:
                if result.source in ('graph', 'temporal') or result.id in accessible_memory_ids:
                    filtered_results.append({
                        "id": result.id,
                        "memory": result.content,
                        "score": result.score,
                        "source": result.source,
                        "metadata": result.metadata,
                        "relationships": result.relationships,
                        "created_at": result.created_at.isoformat() if result.created_at and hasattr(result.created_at, 'isoformat') else str(result.created_at) if result.created_at else None,
                        "updated_at": result.updated_at.isoformat() if result.updated_at and hasattr(result.updated_at, 'isoformat') else str(result.updated_at) if result.updated_at else None,
                    })

            total_before_offset = len(filtered_results)
            filtered_results = filtered_results[offset:offset + 10]

            for result in filtered_results:
                if result.get("id") and result.get("source") == "vector":
                    try:
                        db.add(MemoryAccessLog(
                            memory_id=uuid.UUID(result["id"]), app_id=app.id,
                            access_type="hybrid_search",
                            metadata_={"query": query, "score": result.get("score"), "source": result.get("source")},
                        ))
                    except ValueError:
                        pass

            db.commit()

            response = {
                "query": query,
                "total_results": len(filtered_results),
                "total_available": total_before_offset,
                "offset": offset,
                "has_more": total_before_offset > offset + 10,
                "results": filtered_results,
                "search_strategy": "hybrid (vector + graph + temporal)",
                "sources_found": list(set(r["source"] for r in filtered_results)),
            }

            return json.dumps(response, indent=2)
        finally:
            db.close()
    except Exception as e:
        logging.exception(e)
        return f"Error searching memory: {e}"


@mcp_server.tool(description="List all memories in the user's memory")
async def list_memories() -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)
    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"

    try:
        from service.database import SessionLocal
        from service.database.models import Memory, MemoryAccessLog
        from service.database.utils import get_user_and_app
        from service.database.permissions import check_memory_access_permissions
        from service.memory.client import get_memory_client

        memory_client = get_memory_client()
        if not memory_client:
            return "Error: Memory system is currently unavailable."

        db = SessionLocal()
        try:
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)
            memories = memory_client.get_all(user_id=uid)
            filtered_memories = []

            user_memories = db.query(Memory).filter(Memory.user_id == user.id).all()
            accessible_memory_ids = [memory.id for memory in user_memories if check_memory_access_permissions(db, memory, app.id)]
            if isinstance(memories, dict) and 'results' in memories:
                for memory_data in memories['results']:
                    if 'id' in memory_data:
                        memory_id = uuid.UUID(memory_data['id'])
                        if memory_id in accessible_memory_ids:
                            db.add(MemoryAccessLog(
                                memory_id=memory_id, app_id=app.id,
                                access_type="list", metadata_={"hash": memory_data.get('hash')},
                            ))
                            filtered_memories.append(memory_data)
                db.commit()
            return json.dumps(filtered_memories, indent=2)
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error getting memories: {e}")
        return f"Error getting memories: {e}"


@mcp_server.tool(description="Delete specific memories by their IDs")
async def delete_memories(memory_ids: list[str]) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)
    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"

    try:
        from service.database import SessionLocal
        from service.database.models import Memory, MemoryAccessLog, MemoryState, MemoryStatusHistory
        from service.database.utils import get_user_and_app
        from service.database.permissions import check_memory_access_permissions
        from service.memory.client import get_memory_client
        from service.memory.enhanced import enhanced_memory_manager

        memory_client = get_memory_client()
        if not memory_client:
            return "Error: Memory system is currently unavailable."

        db = SessionLocal()
        try:
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            requested_ids = [uuid.UUID(mid) for mid in memory_ids]
            user_memories = db.query(Memory).filter(Memory.user_id == user.id).all()
            accessible_memory_ids = [memory.id for memory in user_memories if check_memory_access_permissions(db, memory, app.id)]
            ids_to_delete = [mid for mid in requested_ids if mid in accessible_memory_ids]

            if not ids_to_delete:
                return "Error: No accessible memories found with provided IDs"

            deleted_items = []
            related_memories = []
            for memory_id in ids_to_delete:
                memory = db.query(Memory).filter(Memory.id == memory_id).first()
                content = memory.content if memory else ""
                deleted_items.append({"id": str(memory_id), "memory": content[:200]})

                if content:
                    try:
                        related = enhanced_memory_manager.hybrid_search(content, uid, limit=5)
                        for r in related:
                            if r.id != str(memory_id):
                                related_memories.append({
                                    "id": r.id, "memory": r.content[:200],
                                    "score": round(r.score, 3), "source": r.source,
                                })
                    except Exception:
                        pass

            for memory_id in ids_to_delete:
                try:
                    memory_client.delete(str(memory_id))
                except Exception as e:
                    logging.warning(f"Failed to delete {memory_id} from vector store: {e}")

            now = datetime.datetime.now(datetime.UTC)
            for memory_id in ids_to_delete:
                memory = db.query(Memory).filter(Memory.id == memory_id).first()
                if memory:
                    memory.state = MemoryState.deleted
                    memory.deleted_at = now
                    db.add(MemoryStatusHistory(
                        memory_id=memory_id, changed_by=user.id,
                        old_state=MemoryState.active, new_state=MemoryState.deleted,
                    ))
                    db.add(MemoryAccessLog(
                        memory_id=memory_id, app_id=app.id,
                        access_type="delete", metadata_={"operation": "delete_by_id"},
                    ))

            db.commit()

            seen_ids = {d["id"] for d in deleted_items}
            unique_related = [r for r in related_memories if r["id"] not in seen_ids]

            return json.dumps({
                "status": "success",
                "message": f"Deleted {len(ids_to_delete)} memories.",
                "deleted": deleted_items,
                "related_memories": unique_related[:10],
            })
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error deleting memories: {e}")
        return f"Error deleting memories: {e}"


@mcp_server.tool(description="Delete all memories in the user's memory")
async def delete_all_memories() -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)
    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"

    try:
        from service.database import SessionLocal
        from service.database.models import Memory, MemoryAccessLog, MemoryState, MemoryStatusHistory
        from service.database.utils import get_user_and_app
        from service.database.permissions import check_memory_access_permissions
        from service.memory.client import get_memory_client

        memory_client = get_memory_client()
        if not memory_client:
            return "Error: Memory system is currently unavailable."

        db = SessionLocal()
        try:
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)
            user_memories = db.query(Memory).filter(Memory.user_id == user.id).all()
            accessible_memory_ids = [memory.id for memory in user_memories if check_memory_access_permissions(db, memory, app.id)]

            for memory_id in accessible_memory_ids:
                try:
                    memory_client.delete(str(memory_id))
                except Exception as e:
                    logging.warning(f"Failed to delete {memory_id}: {e}")

            now = datetime.datetime.now(datetime.UTC)
            for memory_id in accessible_memory_ids:
                memory = db.query(Memory).filter(Memory.id == memory_id).first()
                if memory:
                    memory.state = MemoryState.deleted
                    memory.deleted_at = now
                    db.add(MemoryStatusHistory(
                        memory_id=memory_id, changed_by=user.id,
                        old_state=MemoryState.active, new_state=MemoryState.deleted,
                    ))
                    db.add(MemoryAccessLog(
                        memory_id=memory_id, app_id=app.id,
                        access_type="delete_all", metadata_={"operation": "bulk_delete"},
                    ))

            db.commit()
            return "Successfully deleted all memories"
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error deleting memories: {e}")
        return f"Error deleting memories: {e}"


@mcp_server.tool(description="Conversation memory — extracts and stores memorable facts from a conversation turn. Pass the user's message and your response.")
async def conversation_memory(user_message: str, llm_response: str) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)
    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"

    try:
        from service.database import SessionLocal
        from service.database.models import Memory, MemoryState, MemoryStatusHistory
        from service.database.utils import get_user_and_app
        from service.memory.enhanced import enhanced_memory_manager

        db = SessionLocal()
        try:
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)
            if not app.is_active:
                return f"Error: App {app.name} is currently paused."

            result = enhanced_memory_manager.comprehensive_memory_handle(
                user_message, llm_response, uid, client_name=client_name,
            )

            for memory_update in result.get("memory_updates", []):
                addition_result = memory_update.get("result", {})
                for memory_data in addition_result.get("added_memories", []):
                    if 'id' in memory_data:
                        memory_id = uuid.UUID(memory_data['id'])
                        memory = db.query(Memory).filter(Memory.id == memory_id).first()
                        if not memory:
                            memory = Memory(
                                id=memory_id, user_id=user.id, app_id=app.id,
                                content=memory_data.get('memory', memory_update.get('content', '')),
                                state=MemoryState.active,
                            )
                            db.add(memory)
                            db.add(MemoryStatusHistory(
                                memory_id=memory_id, changed_by=user.id,
                                old_state=MemoryState.active, new_state=MemoryState.active,
                            ))

            db.commit()

            updates = result.get("memory_updates", [])
            added = sum(1 for u in updates if u.get("result", {}).get("added_memories"))
            skipped = sum(len(u.get("result", {}).get("skipped_facts", [])) for u in updates)
            extracted = result.get("extracted_memories", [])

            response = {
                "status": result.get("status", "processed"),
                "facts_extracted": len(extracted),
                "facts_stored": added,
                "duplicates_skipped": skipped,
                "extracted_facts": extracted[:20],
                "related_memories": len(result.get("related_context", [])),
                "summary": f"Extracted {len(extracted)} facts, stored {added} new, skipped {skipped} duplicates."
            }

            return json.dumps(response, indent=2)
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error in conversation memory: {e}")
        return f"Error processing conversation: {e}"


@mcp_server.tool(description="Talk to the Memory Agent in natural language. It autonomously searches, stores, deletes, or updates memories across all 3 databases (vector, graph, metadata). Returns a synthesized answer.")
async def memory_agent(request: str) -> str:
    uid = user_id_var.get(None)
    if not uid:
        return "Error: user_id not provided"

    try:
        from service.brain.agent import brain_agent
        result = brain_agent.run(request=request, user_id=uid)
        response = {
            "answer": result.answer,
            "steps": result.steps,
            "tools_used": result.tools_called,
            "success": result.success,
            "elapsed_seconds": round(result.elapsed_seconds, 2),
        }
        if result.error:
            response["error"] = result.error
        return json.dumps(response, indent=2)
    except Exception as e:
        logging.exception(f"memory_agent error: {e}")
        return f"Error: {e}"


@mcp_server.tool(description="Get memories related to specific entities or topics, with relationship traversal.")
async def get_related_memories(topic: str, max_depth: int = 2) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)
    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"

    try:
        from service.memory.enhanced import enhanced_memory_manager

        search_results = enhanced_memory_manager.hybrid_search(topic, uid, limit=20)

        related_memories = {
            "direct_matches": [],
            "relationship_connections": [],
            "temporal_context": [],
            "topic_summary": topic,
        }

        for result in search_results:
            memory_data = {
                "id": result.id, "content": result.content,
                "score": result.score, "metadata": result.metadata,
                "relationships": result.relationships,
            }
            if result.source == "vector":
                related_memories["direct_matches"].append(memory_data)
            elif result.source == "graph":
                related_memories["relationship_connections"].append(memory_data)
            elif result.source == "temporal":
                related_memories["temporal_context"].append(memory_data)

        related_memories["insights"] = [
            f"Found {len(search_results)} memories related to '{topic}'",
            f"Direct semantic matches: {len(related_memories['direct_matches'])}",
            f"Relationship connections: {len(related_memories['relationship_connections'])}",
            f"Temporal context: {len(related_memories['temporal_context'])}",
        ]

        return json.dumps(related_memories, indent=2)
    except Exception as e:
        logging.exception(f"Error getting related memories: {e}")
        return f"Error getting related memories: {e}"


# ===========================================================================
# Container Management Tools
# ===========================================================================

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


@mcp_server.tool()
async def list_containers() -> dict:
    """List all managed sub-containers, their status, GPU allocation, and URLs."""
    manager = _get_manager()
    if not manager:
        return {"error": "No container manager configured"}
    return {"containers": manager.list_containers(), "groups": manager.get_groups()}


@mcp_server.tool()
async def start_container(name: str) -> dict:
    """Start a managed sub-container by name."""
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
    """Get GPU device allocation status."""
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


# ===========================================================================
# SSE Connection Handlers
# ===========================================================================

@mcp_router.get("/{client_name}/sse/{user_id}")
async def handle_sse(request: Request):
    """Handle SSE connections for a specific user and client."""
    uid = request.path_params.get("user_id")
    user_token = user_id_var.set(uid or "")
    client_name = request.path_params.get("client_name")
    client_token = client_name_var.set(client_name or "")

    try:
        async with sse.connect_sse(
            request.scope, request.receive, request._send,
        ) as (read_stream, write_stream):
            await mcp_server._mcp_server.run(
                read_stream, write_stream,
                mcp_server._mcp_server.create_initialization_options(),
            )
    finally:
        user_id_var.reset(user_token)
        client_name_var.reset(client_token)


@mcp_router.post("/messages/")
async def handle_messages_root(request: Request):
    return await _handle_post(request)


@mcp_router.post("/{client_name}/sse/{user_id}/messages/")
async def handle_messages_user(request: Request):
    return await _handle_post(request)


async def _handle_post(request: Request):
    """Handle POST messages for SSE."""
    try:
        body = await request.body()
        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}
        async def send(message):
            return {}
        await sse.handle_post_message(request.scope, receive, send)
        return {"status": "ok"}
    except Exception:
        pass


def setup_mcp_server(app):
    """Mount the MCP server onto the FastAPI app."""
    global _app
    _app = app

    mcp_server._mcp_server.name = "openmemory-mcp-server"

    # Include the MCP router with SSE endpoints
    app.include_router(mcp_router)

    logger.info("MCP SSE server mounted at /mcp/{client_name}/sse/{user_id}")
