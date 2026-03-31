"""
Unified Memory API Router.

Consolidates memory CRUD, app management, and stats endpoints
from the original openmemory routers into a single file with sub-routers.

All endpoints are prefixed with /api/v1/ via sub-router prefixes.
"""

import logging
import os
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Set
from uuid import UUID

from service.database import get_db
from service.database.models import (
    AccessControl,
    App,
    Category,
    Memory,
    MemoryAccessLog,
    MemoryState,
    MemoryStatusHistory,
    User,
)
from service.database.permissions import check_memory_access_permissions
from service.memory.client import get_memory_client

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi_pagination import Page, Params
from fastapi_pagination.ext.sqlalchemy import paginate as sqlalchemy_paginate
from pydantic import BaseModel, validator
from sqlalchemy import desc, func
from sqlalchemy.orm import Session, joinedload

# Qdrant imports for sync operations
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class MemoryResponse(BaseModel):
    id: UUID
    content: str
    created_at: int
    state: str
    app_id: UUID
    app_name: str
    categories: List[str]
    metadata_: Optional[dict] = None

    @validator('created_at', pre=True)
    def convert_to_epoch(cls, v):
        if isinstance(v, datetime):
            return int(v.timestamp())
        return v


# ---------------------------------------------------------------------------
# Top-level router
# ---------------------------------------------------------------------------

router = APIRouter(tags=["api"])

# Sub-routers
memories_router = APIRouter(prefix="/memories", tags=["memories"])
apps_router = APIRouter(prefix="/apps", tags=["apps"])
stats_router = APIRouter(prefix="/stats", tags=["stats"])


# ===========================================================================
# Helper functions
# ===========================================================================

def get_memory_or_404(db: Session, memory_id: UUID) -> Memory:
    memory = db.query(Memory).filter(Memory.id == memory_id).first()
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    return memory


def update_memory_state(db: Session, memory_id: UUID, new_state: MemoryState, user_id: UUID):
    memory = get_memory_or_404(db, memory_id)
    old_state = memory.state

    memory.state = new_state
    if new_state == MemoryState.archived:
        memory.archived_at = datetime.now(UTC)
    elif new_state == MemoryState.deleted:
        memory.deleted_at = datetime.now(UTC)

    history = MemoryStatusHistory(
        memory_id=memory_id,
        changed_by=user_id,
        old_state=old_state,
        new_state=new_state
    )
    db.add(history)
    db.commit()
    return memory


def get_accessible_memory_ids(db: Session, app_id: UUID) -> Set[UUID]:
    """
    Get the set of memory IDs that the app has access to based on app-level ACL rules.
    Returns None if no specific restrictions are found (all memories accessible).
    """
    app_access = db.query(AccessControl).filter(
        AccessControl.subject_type == "app",
        AccessControl.subject_id == app_id,
        AccessControl.object_type == "memory"
    ).all()

    if not app_access:
        return None

    allowed_memory_ids = set()
    denied_memory_ids = set()

    for rule in app_access:
        if rule.effect == "allow":
            if rule.object_id:
                allowed_memory_ids.add(rule.object_id)
            else:
                return None
        elif rule.effect == "deny":
            if rule.object_id:
                denied_memory_ids.add(rule.object_id)
            else:
                return set()

    if allowed_memory_ids:
        allowed_memory_ids -= denied_memory_ids

    return allowed_memory_ids


def query_graph(user_id: str, query_str: str, params: dict = None) -> List[Dict]:
    """Execute a Cypher query on the graph database"""
    try:
        memory_client = get_memory_client()
        if not memory_client or not hasattr(memory_client, 'graph'):
            return []

        graph = memory_client.graph
        params = params or {}
        params['user_id'] = user_id

        if hasattr(graph, 'graph') and hasattr(graph.graph, 'query'):
            result = graph.graph.query(query_str, params)
            if isinstance(result, list):
                return result
            return [result] if result else []
        return []
    except Exception as e:
        logging.error(f"Graph query error: {e}")
        return []


def get_memory_entities_from_graph(memory_id: UUID, user_id: str) -> List[Dict]:
    """Get entities extracted from a specific memory"""
    query = """
    MATCH (n)-[:MENTIONED_IN]->(m:Memory {id: $memory_id})
    WHERE m.user_id = $user_id
    RETURN n.name as name, labels(n) as types, properties(n) as properties
    """
    return query_graph(user_id, query, {"memory_id": str(memory_id)})


def get_related_memories_from_graph(memory_id: UUID, user_id: str, limit: int = 10) -> List[str]:
    """Find related memories through shared entities"""
    query = """
    MATCH (entity)-[:MENTIONED_IN]->(m1:Memory {id: $memory_id})
    MATCH (entity)-[:MENTIONED_IN]->(m2:Memory)
    WHERE m1.user_id = $user_id AND m2.user_id = $user_id AND m1.id <> m2.id
    RETURN DISTINCT m2.id as related_memory_id, count(entity) as shared_entities
    ORDER BY shared_entities DESC
    LIMIT $limit
    """
    results = query_graph(user_id, query, {"memory_id": str(memory_id), "limit": limit})
    return [r["related_memory_id"] for r in results]


def _is_advanced_mode() -> bool:
    return os.environ.get("MEMORY_MODE", "advanced").lower() == "advanced"


# ===========================================================================
# MEMORIES ENDPOINTS
# ===========================================================================

@memories_router.get("/", response_model=Page[MemoryResponse])
async def list_memories(
    user_id: str,
    app_id: Optional[UUID] = None,
    from_date: Optional[int] = Query(
        None,
        description="Filter memories created after this date (timestamp)",
        examples=[1718505600]
    ),
    to_date: Optional[int] = Query(
        None,
        description="Filter memories created before this date (timestamp)",
        examples=[1718505600]
    ),
    categories: Optional[str] = None,
    params: Params = Depends(),
    search_query: Optional[str] = None,
    sort_column: Optional[str] = Query(None, description="Column to sort by (memory, categories, app_name, created_at)"),
    sort_direction: Optional[str] = Query(None, description="Sort direction (asc or desc)"),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    query = db.query(Memory).filter(
        Memory.user_id == user.id,
        Memory.state != MemoryState.deleted,
        Memory.state != MemoryState.archived,
        Memory.content.ilike(f"%{search_query}%") if search_query else True
    )

    if app_id:
        query = query.filter(Memory.app_id == app_id)

    if from_date:
        from_datetime = datetime.fromtimestamp(from_date, tz=UTC)
        query = query.filter(Memory.created_at >= from_datetime)

    if to_date:
        to_datetime = datetime.fromtimestamp(to_date, tz=UTC)
        query = query.filter(Memory.created_at <= to_datetime)

    query = query.outerjoin(App, Memory.app_id == App.id)
    query = query.outerjoin(Memory.categories)

    if categories:
        category_list = [c.strip() for c in categories.split(",")]
        query = query.filter(Category.name.in_(category_list))

    if sort_column:
        sort_field = getattr(Memory, sort_column, None)
        if sort_field:
            query = query.order_by(sort_field.desc()) if sort_direction == "desc" else query.order_by(sort_field.asc())

    query = query.options(
        joinedload(Memory.app),
        joinedload(Memory.categories)
    ).distinct(Memory.id)

    return sqlalchemy_paginate(
        query,
        params,
        transformer=lambda items: [
            MemoryResponse(
                id=memory.id,
                content=memory.content,
                created_at=memory.created_at,
                state=memory.state.value,
                app_id=memory.app_id,
                app_name=memory.app.name if memory.app else None,
                categories=[category.name for category in memory.categories],
                metadata_=memory.metadata_
            )
            for memory in items
            if check_memory_access_permissions(db, memory, app_id)
        ]
    )


@memories_router.get("/categories")
async def get_categories(
    user_id: str,
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    memories = db.query(Memory).filter(
        Memory.user_id == user.id,
        Memory.state != MemoryState.deleted,
        Memory.state != MemoryState.archived
    ).all()
    categories = [category for memory in memories for category in memory.categories]
    unique_categories = list(set(categories))

    return {
        "categories": unique_categories,
        "total": len(unique_categories)
    }


class SearchMemoriesRequest(BaseModel):
    query: str
    user_id: str
    limit: int = 10
    offset: int = 0


@memories_router.post("/search")
async def search_memories_hybrid(
    request: SearchMemoriesRequest,
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.user_id == request.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if _is_advanced_mode():
        from service.memory.enhanced import enhanced_memory_manager
        fetch_limit = request.limit + request.offset
        all_results = enhanced_memory_manager.hybrid_search(
            request.query, request.user_id, limit=fetch_limit
        )
        results = all_results[request.offset:request.offset + request.limit]
        sources_used = list(set(r.source for r in results))
        return {
            "query": request.query,
            "results": [
                {
                    "id": r.id,
                    "memory": r.content,
                    "score": round(r.score, 3),
                    "source": r.source,
                    "created_at": r.created_at.isoformat() if hasattr(r.created_at, 'isoformat') else r.created_at,
                }
                for r in results
            ],
            "total": len(results),
            "total_available": len(all_results),
            "offset": request.offset,
            "has_more": len(all_results) > request.offset + request.limit,
            "sources_used": sources_used,
        }
    else:
        query = db.query(Memory).filter(
            Memory.user_id == user.id,
            Memory.state != MemoryState.deleted,
            Memory.state != MemoryState.archived,
            Memory.content.ilike(f"%{request.query}%"),
        ).limit(request.limit).all()
        return {
            "query": request.query,
            "results": [
                {
                    "id": str(m.id),
                    "memory": m.content,
                    "score": 1.0,
                    "source": "sqlite",
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                }
                for m in query
            ],
            "total": len(query),
            "sources_used": ["sqlite"],
        }


class CreateMemoryRequest(BaseModel):
    user_id: str
    text: Optional[str] = None
    messages: Optional[List[dict]] = None
    metadata: dict = {}
    infer: bool = True
    app: str = "openmemory"


@memories_router.post("/")
async def create_memory(
    request: CreateMemoryRequest,
    db: Session = Depends(get_db)
):
    from uuid import uuid4

    user = db.query(User).filter(User.user_id == request.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not request.text and request.messages:
        request.text = " ".join([
            msg.get("content", "")
            for msg in request.messages
            if msg.get("content")
        ])

    if not request.text:
        raise HTTPException(
            status_code=400,
            detail="Either 'text' or 'messages' must be provided"
        )

    app_obj = db.query(App).filter(App.name == request.app,
                                   App.owner_id == user.id).first()
    if not app_obj:
        app_obj = App(name=request.app, owner_id=user.id)
        db.add(app_obj)
        db.commit()
        db.refresh(app_obj)

    if not app_obj.is_active:
        raise HTTPException(status_code=403, detail=f"App {request.app} is currently paused on OpenMemory. Cannot create new memories.")

    logging.info(f"Creating memory for user_id: {request.user_id} with app: {request.app}")

    if _is_advanced_mode():
        from service.memory.enhanced import enhanced_memory_manager
        try:
            add_result = enhanced_memory_manager.smart_add_memory(
                request.text,
                request.user_id,
                metadata={"source_app": "openmemory", "mcp_client": request.app, **(request.metadata or {})},
            )
        except Exception as add_error:
            logging.error(f"smart_add_memory failed: {add_error}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Memory extraction failed: {str(add_error)}")

        created_memories = []
        results_output = []
        processed_ids: Set[str] = set()
        for result in add_result.added_memories:
            result_id = result.get("id")
            if not result_id or str(result_id) in processed_ids:
                continue
            processed_ids.add(str(result_id))
            memory_id = UUID(result_id) if isinstance(result_id, str) else result_id
            memory_text = result.get("memory", "")
            event = result.get("event", "ADD")

            existing_memory = db.query(Memory).filter(Memory.id == memory_id).first()
            if existing_memory:
                existing_memory.state = MemoryState.active
                existing_memory.content = memory_text
                memory_obj = existing_memory
            else:
                memory_obj = Memory(
                    id=memory_id, user_id=user.id, app_id=app_obj.id,
                    content=memory_text, metadata_=request.metadata, state=MemoryState.active,
                )
                db.add(memory_obj)

            db.add(MemoryStatusHistory(
                memory_id=memory_id, changed_by=user.id,
                old_state=MemoryState.active, new_state=MemoryState.active,
            ))
            created_memories.append(memory_obj)
            results_output.append({"id": str(memory_id), "memory": memory_text, "event": event})

        db.commit()
        for m in created_memories:
            db.refresh(m)

        response = {
            "results": results_output,
            "count": len(results_output),
            "status": add_result.status,
            "skipped_duplicates": len(add_result.skipped_facts),
            "related_memories": [
                {"id": r.get("id", ""), "memory": r.get("content", "")[:200], "score": round(r.get("score", 0), 3)}
                for r in add_result.related_memories[:5]
            ],
            "summary": add_result.summary,
        }
        return response

    # Simple mode: raw mem0 add
    try:
        memory_client = get_memory_client()
        if not memory_client:
            raise Exception("Memory client is not available")
    except Exception as client_error:
        logging.warning(f"Memory client unavailable: {client_error}.")
        raise HTTPException(status_code=503, detail=f"Memory service unavailable: {str(client_error)}")

    qdrant_response = None
    try:
        qdrant_response = memory_client.add(
            request.text,
            user_id=request.user_id,
            metadata={"source_app": "openmemory", "mcp_client": request.app},
            infer=request.infer,
        )
        logging.info(f"Memory add response: {qdrant_response}")
    except Exception as add_error:
        logging.error(f"memory_client.add() failed: {add_error}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Memory extraction failed: {str(add_error)}")

    created_memories = []
    results_output = []
    processed_ids: Set[str] = set()

    if isinstance(qdrant_response, dict) and 'results' in qdrant_response:
        for result in qdrant_response['results']:
            event = result.get('event', 'NONE')

            if event == 'DELETE':
                result_id = result.get('id')
                if result_id:
                    memory_id = UUID(result_id) if isinstance(result_id, str) else result_id
                    try:
                        existing = db.query(Memory).filter(Memory.id == memory_id).first()
                        if existing:
                            update_memory_state(db, memory_id, MemoryState.deleted, user.id)
                            results_output.append({"id": str(memory_id), "memory": result.get('memory', ''), "event": "DELETE"})
                    except Exception:
                        logging.warning(f"Failed to sync DELETE for {result_id}")
                continue

            if event not in ('ADD', 'UPDATE'):
                continue

            result_id = result.get('id')
            if not result_id:
                continue

            memory_id = UUID(result_id) if isinstance(result_id, str) else result_id
            id_str = str(memory_id)
            if id_str in processed_ids:
                continue
            processed_ids.add(id_str)

            memory_text = result.get('memory', request.text)
            existing_memory = db.query(Memory).filter(Memory.id == memory_id).first()
            if existing_memory:
                existing_memory.state = MemoryState.active
                existing_memory.content = memory_text
                memory = existing_memory
            else:
                memory = Memory(
                    id=memory_id, user_id=user.id, app_id=app_obj.id,
                    content=memory_text, metadata_=request.metadata, state=MemoryState.active,
                )
                db.add(memory)

            db.add(MemoryStatusHistory(
                memory_id=memory_id, changed_by=user.id,
                old_state=MemoryState.active, new_state=MemoryState.active,
            ))
            created_memories.append(memory)
            results_output.append({"id": id_str, "memory": memory_text, "event": event})

    db.commit()
    for memory in created_memories:
        db.refresh(memory)

    return {"results": results_output, "count": len(results_output)}


class ConversationMemoryRequest(BaseModel):
    user_id: str
    user_message: str
    llm_response: str
    metadata: dict = {}
    app: str = "openmemory"


@memories_router.post("/conversation")
async def conversation_memory(
    request: ConversationMemoryRequest,
    db: Session = Depends(get_db)
):
    """
    REST endpoint for conversation_memory.
    Extracts memorable facts from a conversation turn, reviews with LLM, deduplicates, stores.
    """
    from uuid import uuid4
    from service.memory.enhanced import enhanced_memory_manager

    user = db.query(User).filter(User.user_id == request.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    app_obj = db.query(App).filter(App.name == request.app,
                                   App.owner_id == user.id).first()
    if not app_obj:
        app_obj = App(name=request.app, owner_id=user.id)
        db.add(app_obj)
        db.commit()
        db.refresh(app_obj)

    if not app_obj.is_active:
        raise HTTPException(status_code=403, detail=f"App {request.app} is currently paused.")

    result = enhanced_memory_manager.comprehensive_memory_handle(
        request.user_message,
        request.llm_response,
        request.user_id,
        client_name=request.app,
    )

    # Sync added memories to SQLite
    for memory_update in result.get("memory_updates", []):
        addition_result = memory_update.get("result", {})
        for memory_data in addition_result.get("added_memories", []):
            if 'id' in memory_data:
                from uuid import UUID as UUIDType
                memory_id = UUIDType(memory_data['id'])
                memory = db.query(Memory).filter(Memory.id == memory_id).first()
                if not memory:
                    memory = Memory(
                        id=memory_id,
                        user_id=user.id,
                        app_id=app_obj.id,
                        content=memory_data.get('memory', memory_update.get('content', '')),
                        state=MemoryState.active
                    )
                    db.add(memory)
                    db.add(MemoryStatusHistory(
                        memory_id=memory_id,
                        changed_by=user.id,
                        old_state=MemoryState.active,
                        new_state=MemoryState.active
                    ))

    db.commit()

    updates = result.get("memory_updates", [])
    added = sum(1 for u in updates if u.get("result", {}).get("added_memories"))
    skipped = sum(len(u.get("result", {}).get("skipped_facts", [])) for u in updates)
    extracted = result.get("extracted_memories", [])

    return {
        "status": result.get("status", "processed"),
        "facts_extracted": len(extracted),
        "facts_stored": added,
        "duplicates_skipped": skipped,
        "extracted_facts": extracted[:20],
        "related_memories": len(result.get("related_context", [])),
        "summary": f"Extracted {len(extracted)} facts, stored {added} new, skipped {skipped} duplicates."
    }


@memories_router.get("/{memory_id}")
async def get_memory(
    memory_id: UUID,
    include_entities: bool = Query(False, description="Include extracted entities from graph"),
    db: Session = Depends(get_db)
):
    memory = get_memory_or_404(db, memory_id)

    response = {
        "id": memory.id,
        "text": memory.content,
        "created_at": int(memory.created_at.timestamp()),
        "state": memory.state.value,
        "app_id": memory.app_id,
        "app_name": memory.app.name if memory.app else None,
        "categories": [category.name for category in memory.categories],
        "metadata_": memory.metadata_
    }

    if include_entities:
        user = db.query(User).filter(User.id == memory.user_id).first()
        if user:
            entities = get_memory_entities_from_graph(memory_id, user.user_id)
            response["entities"] = [
                {
                    "name": e["name"],
                    "type": e["types"][0] if e["types"] else "Unknown"
                }
                for e in entities
            ]

    return response


class DeleteAllRequest(BaseModel):
    user_id: str


@memories_router.post("/delete_all")
async def delete_all_memories(
    request: DeleteAllRequest,
    db: Session = Depends(get_db)
):
    """Delete ALL memories for a user (vector store + SQLite + graph)."""
    user = db.query(User).filter(User.user_id == request.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    try:
        memory_client = get_memory_client()
        if not memory_client:
            raise HTTPException(status_code=503, detail="Memory client is not available")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Memory service unavailable: {str(e)}")

    user_memories = db.query(Memory).filter(
        Memory.user_id == user.id,
        Memory.state != MemoryState.deleted,
    ).all()

    deleted_count = 0
    for memory in user_memories:
        try:
            memory_client.delete(memory_id=str(memory.id))
        except Exception as e:
            logging.warning(f"Failed to delete {memory.id} from vector store: {e}")
        memory.state = MemoryState.deleted
        memory.deleted_at = datetime.now(UTC)
        deleted_count += 1

    # Clean orphaned Qdrant points
    try:
        qdrant = QdrantClient(
            host=os.environ.get("QDRANT_HOST", "mem0_store"),
            port=int(os.environ.get("QDRANT_PORT", 6333)),
        )
        collection = os.environ.get("QDRANT_COLLECTION", "openmemory")
        qdrant.delete(
            collection_name=collection,
            points_selector=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=request.user_id))]
            ),
        )
    except Exception as e:
        logging.warning(f"Failed to clean orphaned Qdrant points: {e}")

    # Clear Neo4j graph for this user
    try:
        if memory_client and hasattr(memory_client, 'graph') and memory_client.graph:
            graph = memory_client.graph
            if hasattr(graph, 'graph') and hasattr(graph.graph, 'query'):
                graph.graph.query(
                    "MATCH (n) WHERE n.user_id = $user_id DETACH DELETE n",
                    params={"user_id": request.user_id}
                )
    except Exception as e:
        logging.warning(f"Failed to clear graph: {e}")

    db.commit()
    return {"message": f"Deleted {deleted_count} memories", "count": deleted_count}


class DeleteMemoriesRequest(BaseModel):
    memory_ids: List[UUID]
    user_id: str


@memories_router.delete("/")
async def delete_memories(
    request: DeleteMemoriesRequest,
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.user_id == request.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    try:
        memory_client = get_memory_client()
        if not memory_client:
            raise HTTPException(status_code=503, detail="Memory client is not available")
    except HTTPException:
        raise
    except Exception as client_error:
        raise HTTPException(status_code=503, detail=f"Memory service unavailable: {str(client_error)}")

    deleted_items = []
    related_memories = []

    for memory_id in request.memory_ids:
        memory_obj = db.query(Memory).filter(Memory.id == memory_id).first()
        deleted_content = memory_obj.content if memory_obj else ""

        if _is_advanced_mode() and deleted_content:
            try:
                from service.memory.enhanced import enhanced_memory_manager
                related = enhanced_memory_manager.hybrid_search(
                    deleted_content, request.user_id, limit=5
                )
                for r in related:
                    if r.id != str(memory_id):
                        related_memories.append({
                            "id": r.id,
                            "memory": r.content[:200],
                            "score": round(r.score, 3),
                            "source": r.source,
                            "relationship": "similar content",
                        })
            except Exception as e:
                logging.warning(f"Failed to find related memories for {memory_id}: {e}")

        try:
            memory_client.delete(memory_id=str(memory_id))
        except Exception as delete_error:
            logging.warning(f"Failed to delete memory {memory_id} from vector store: {delete_error}")

        update_memory_state(db, memory_id, MemoryState.deleted, user.id)
        deleted_items.append({"id": str(memory_id), "memory": deleted_content[:200]})

    seen_ids = {d["id"] for d in deleted_items}
    unique_related = []
    for r in related_memories:
        if r["id"] not in seen_ids:
            seen_ids.add(r["id"])
            unique_related.append(r)

    response = {
        "message": f"Deleted {len(request.memory_ids)} memories.",
        "deleted": deleted_items,
    }
    if unique_related:
        response["related_memories"] = unique_related[:10]
        response["message"] += f" Found {len(unique_related)} related memories that may also need attention."

    return response


@memories_router.post("/actions/archive")
async def archive_memories(
    memory_ids: List[UUID],
    user_id: UUID,
    db: Session = Depends(get_db)
):
    for memory_id in memory_ids:
        update_memory_state(db, memory_id, MemoryState.archived, user_id)
    return {"message": f"Successfully archived {len(memory_ids)} memories"}


class PauseMemoriesRequest(BaseModel):
    memory_ids: Optional[List[UUID]] = None
    category_ids: Optional[List[UUID]] = None
    app_id: Optional[UUID] = None
    all_for_app: bool = False
    global_pause: bool = False
    state: Optional[MemoryState] = None
    user_id: str


@memories_router.post("/actions/pause")
async def pause_memories(
    request: PauseMemoriesRequest,
    db: Session = Depends(get_db)
):
    global_pause = request.global_pause
    all_for_app = request.all_for_app
    app_id = request.app_id
    memory_ids = request.memory_ids
    category_ids = request.category_ids
    state = request.state or MemoryState.paused

    user = db.query(User).filter(User.user_id == request.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user_id = user.id

    if global_pause:
        memories = db.query(Memory).filter(
            Memory.state != MemoryState.deleted,
            Memory.state != MemoryState.archived
        ).all()
        for memory in memories:
            update_memory_state(db, memory.id, state, user_id)
        return {"message": "Successfully paused all memories"}

    if app_id:
        memories = db.query(Memory).filter(
            Memory.app_id == app_id,
            Memory.user_id == user.id,
            Memory.state != MemoryState.deleted,
            Memory.state != MemoryState.archived
        ).all()
        for memory in memories:
            update_memory_state(db, memory.id, state, user_id)
        return {"message": f"Successfully paused all memories for app {app_id}"}

    if all_for_app and memory_ids:
        memories = db.query(Memory).filter(
            Memory.user_id == user.id,
            Memory.state != MemoryState.deleted,
            Memory.id.in_(memory_ids)
        ).all()
        for memory in memories:
            update_memory_state(db, memory.id, state, user_id)
        return {"message": "Successfully paused all memories"}

    if memory_ids:
        for memory_id in memory_ids:
            update_memory_state(db, memory_id, state, user_id)
        return {"message": f"Successfully paused {len(memory_ids)} memories"}

    if category_ids:
        memories = db.query(Memory).join(Memory.categories).filter(
            Category.id.in_(category_ids),
            Memory.state != MemoryState.deleted,
            Memory.state != MemoryState.archived
        ).all()
        for memory in memories:
            update_memory_state(db, memory.id, state, user_id)
        return {"message": f"Successfully paused memories in {len(category_ids)} categories"}

    raise HTTPException(status_code=400, detail="Invalid pause request parameters")


@memories_router.post("/actions/sync-storage")
async def sync_storage(
    user_id: str,
    direction: str = Query(..., description="'db-to-vector' or 'vector-to-db'"),
    db: Session = Depends(get_db)
):
    """Sync memories between SQLite database and Qdrant vector store."""
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    try:
        memory_client = get_memory_client()
        if not memory_client:
            raise HTTPException(status_code=503, detail="Memory client not available")

        qdrant_host = os.getenv('QDRANT_HOST', 'mem0_store')
        qdrant_port = int(os.getenv('QDRANT_PORT', 6333))
        collection_name = os.getenv('QDRANT_COLLECTION', 'openmemory')

        qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)

        db_memories = db.query(Memory).filter(
            Memory.user_id == user.id,
            Memory.state != MemoryState.deleted
        ).all()
        db_memory_ids = {str(mem.id) for mem in db_memories}

        try:
            scroll_result = qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="user_id",
                            match=MatchValue(value=user_id)
                        )
                    ]
                ),
                limit=1000,
                with_payload=True
            )
            vector_memory_ids = {str(point.id) for point, _ in scroll_result}
        except Exception as e:
            logging.error(f"Error reading from Qdrant: {e}")
            vector_memory_ids = set()

        if direction == "db-to-vector":
            orphaned_vectors = vector_memory_ids - db_memory_ids
            if orphaned_vectors:
                qdrant_client.delete(
                    collection_name=collection_name,
                    points_selector=list(orphaned_vectors)
                )
                return {
                    "message": f"Removed {len(orphaned_vectors)} orphaned vectors from Qdrant",
                    "removed_ids": list(orphaned_vectors)
                }
            return {"message": "No orphaned vectors found in Qdrant"}

        elif direction == "vector-to-db":
            orphaned_records = db_memory_ids - vector_memory_ids
            if orphaned_records:
                for memory_id in orphaned_records:
                    update_memory_state(db, UUID(memory_id), MemoryState.deleted, user.id)
                return {
                    "message": f"Removed {len(orphaned_records)} orphaned records from database",
                    "removed_ids": list(orphaned_records)
                }
            return {"message": "No orphaned records found in database"}

        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid direction. Must be 'db-to-vector' or 'vector-to-db'"
            )

    except Exception as e:
        logging.error(f"Error syncing storage: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to sync storage: {str(e)}")


@memories_router.get("/{memory_id}/access-log")
async def get_memory_access_log(
    memory_id: UUID,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    query = db.query(MemoryAccessLog).filter(MemoryAccessLog.memory_id == memory_id)
    total = query.count()
    logs = query.order_by(MemoryAccessLog.accessed_at.desc()).offset((page - 1) * page_size).limit(page_size).all()

    for log in logs:
        app = db.query(App).filter(App.id == log.app_id).first()
        log.app_name = app.name if app else None

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "logs": logs
    }


class UpdateMemoryRequest(BaseModel):
    memory_content: str
    user_id: str


@memories_router.put("/{memory_id}")
async def update_memory(
    memory_id: UUID,
    request: UpdateMemoryRequest,
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.user_id == request.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    memory = get_memory_or_404(db, memory_id)
    memory.content = request.memory_content
    db.commit()
    db.refresh(memory)
    return memory


class FilterMemoriesRequest(BaseModel):
    user_id: str
    page: int = 1
    size: int = 10
    search_query: Optional[str] = None
    app_ids: Optional[List[UUID]] = None
    category_ids: Optional[List[UUID]] = None
    sort_column: Optional[str] = None
    sort_direction: Optional[str] = None
    from_date: Optional[int] = None
    to_date: Optional[int] = None
    show_archived: Optional[bool] = False


@memories_router.post("/filter", response_model=Page[MemoryResponse])
async def filter_memories(
    request: FilterMemoriesRequest,
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.user_id == request.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    query = db.query(Memory).filter(
        Memory.user_id == user.id,
        Memory.state != MemoryState.deleted,
    )

    if not request.show_archived:
        query = query.filter(Memory.state != MemoryState.archived)

    if request.search_query:
        query = query.filter(Memory.content.ilike(f"%{request.search_query}%"))

    if request.app_ids:
        query = query.filter(Memory.app_id.in_(request.app_ids))

    query = query.outerjoin(App, Memory.app_id == App.id)

    if request.category_ids:
        query = query.join(Memory.categories).filter(Category.id.in_(request.category_ids))
    else:
        query = query.outerjoin(Memory.categories)

    if request.from_date:
        from_datetime = datetime.fromtimestamp(request.from_date, tz=UTC)
        query = query.filter(Memory.created_at >= from_datetime)

    if request.to_date:
        to_datetime = datetime.fromtimestamp(request.to_date, tz=UTC)
        query = query.filter(Memory.created_at <= to_datetime)

    if request.sort_column and request.sort_direction:
        sort_direction = request.sort_direction.lower()
        if sort_direction not in ['asc', 'desc']:
            raise HTTPException(status_code=400, detail="Invalid sort direction")

        sort_mapping = {
            'memory': Memory.content,
            'app_name': App.name,
            'created_at': Memory.created_at
        }

        if request.sort_column not in sort_mapping:
            raise HTTPException(status_code=400, detail="Invalid sort column")

        sort_field = sort_mapping[request.sort_column]
        if sort_direction == 'desc':
            query = query.order_by(sort_field.desc())
        else:
            query = query.order_by(sort_field.asc())
    else:
        query = query.order_by(Memory.created_at.desc())

    query = query.options(
        joinedload(Memory.categories)
    ).distinct(Memory.id)

    return sqlalchemy_paginate(
        query,
        Params(page=request.page, size=request.size),
        transformer=lambda items: [
            MemoryResponse(
                id=memory.id,
                content=memory.content,
                created_at=memory.created_at,
                state=memory.state.value,
                app_id=memory.app_id,
                app_name=memory.app.name if memory.app else None,
                categories=[category.name for category in memory.categories],
                metadata_=memory.metadata_
            )
            for memory in items
        ]
    )


@memories_router.get("/{memory_id}/related", response_model=Page[MemoryResponse])
async def get_related_memories(
    memory_id: UUID,
    user_id: str,
    use_graph: bool = Query(True, description="Use graph relationships for finding related memories"),
    params: Params = Depends(),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    memory = get_memory_or_404(db, memory_id)

    category_ids = [category.id for category in memory.categories]

    graph_memory_ids = []
    if use_graph:
        graph_memory_ids = get_related_memories_from_graph(memory_id, user_id, limit=20)

    if category_ids or graph_memory_ids:
        query = db.query(Memory).filter(
            Memory.user_id == user.id,
            Memory.id != memory_id,
            Memory.state != MemoryState.deleted
        )

        if graph_memory_ids:
            graph_uuids = [UUID(mid) for mid in graph_memory_ids]
            query = query.filter(Memory.id.in_(graph_uuids))
        elif category_ids:
            query = query.join(Memory.categories).filter(
                Category.id.in_(category_ids)
            )

        query = query.options(
            joinedload(Memory.categories),
            joinedload(Memory.app)
        ).order_by(Memory.created_at.desc())
    else:
        return Page.create([], total=0, params=params)

    params = Params(page=params.page, size=5)

    return sqlalchemy_paginate(
        query,
        params,
        transformer=lambda items: [
            MemoryResponse(
                id=memory.id,
                content=memory.content,
                created_at=memory.created_at,
                state=memory.state.value,
                app_id=memory.app_id,
                app_name=memory.app.name if memory.app else None,
                categories=[category.name for category in memory.categories],
                metadata_=memory.metadata_
            )
            for memory in items
        ]
    )


# ========== GRAPH ENDPOINTS ==========

@memories_router.get("/graph/stats")
async def get_graph_stats(
    user_id: str,
    db: Session = Depends(get_db)
):
    """Get statistics about the knowledge graph"""
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    try:
        memory_client = get_memory_client()
        if not memory_client or not hasattr(memory_client, 'graph'):
            return {
                "error": "Graph store not available",
                "graph_enabled": False
            }

        node_result = query_graph(
            user_id,
            "MATCH (n) WHERE n.user_id = $user_id RETURN count(n) as count"
        )
        node_count = node_result[0]['count'] if node_result else 0

        rel_result = query_graph(
            user_id,
            "MATCH (n)-[r]-(m) WHERE n.user_id = $user_id RETURN count(r) as count"
        )
        rel_count = rel_result[0]['count'] if rel_result else 0

        entity_result = query_graph(
            user_id,
            """
            MATCH (n)
            WHERE n.user_id = $user_id
            RETURN labels(n) as labels, count(*) as count
            ORDER BY count DESC
            LIMIT 10
            """
        )
        entity_types = [
            {"type": record["labels"][0] if record.get("labels") else "Unknown",
             "count": record["count"]}
            for record in entity_result
        ]

        return {
            "graph_enabled": True,
            "total_entities": node_count,
            "total_relationships": rel_count,
            "entity_types": entity_types,
            "user_id": user_id
        }

    except Exception as e:
        logging.error(f"Error getting graph stats: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "graph_enabled": False
        }


@memories_router.get("/{memory_id}/entities")
async def get_memory_graph_entities(
    memory_id: UUID,
    user_id: str,
    db: Session = Depends(get_db)
):
    """Get entities extracted from a specific memory"""
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    memory = get_memory_or_404(db, memory_id)

    entities = get_memory_entities_from_graph(memory_id, user_id)

    return {
        "memory_id": str(memory_id),
        "entities": [
            {
                "name": e["name"],
                "type": e["types"][0] if e["types"] else "Unknown",
                "properties": e.get("properties", {})
            }
            for e in entities
        ],
        "count": len(entities)
    }


@memories_router.get("/graph/search")
async def graph_search(
    query: str,
    user_id: str,
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """Search for entities in the knowledge graph"""
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    cypher_query = """
    MATCH (n)
    WHERE n.user_id = $user_id
    AND toLower(n.name) CONTAINS toLower($query)
    RETURN n.name as name,
           labels(n) as types,
           properties(n) as properties
    LIMIT $limit
    """

    results = query_graph(user_id, cypher_query, {"query": query, "limit": limit})

    return {
        "query": query,
        "results": [
            {
                "name": r["name"],
                "type": r["types"][0] if r["types"] else "Unknown",
                "properties": r.get("properties", {})
            }
            for r in results
        ],
        "count": len(results)
    }


@memories_router.get("/graph/entity/{entity_name}/related")
async def get_related_entities(
    entity_name: str,
    user_id: str,
    depth: int = Query(1, ge=1, le=3),
    db: Session = Depends(get_db)
):
    """Get entities related to a specific entity"""
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    cypher_query = f"""
    MATCH path = (start)-[*1..{depth}]-(related)
    WHERE start.name = $entity_name
    AND start.user_id = $user_id
    AND related.user_id = $user_id
    RETURN
        start.name as source,
        related.name as target,
        type(relationships(path)[0]) as relationship,
        length(path) as distance
    LIMIT 50
    """

    results = query_graph(user_id, cypher_query, {"entity_name": entity_name})

    return {
        "entity": entity_name,
        "relationships": [
            {
                "source": r["source"],
                "target": r["target"],
                "relationship": r.get("relationship"),
                "distance": r["distance"]
            }
            for r in results
        ],
        "count": len(results)
    }


@memories_router.get("/graph/entity/{entity_name}/memories")
async def get_memories_by_entity(
    entity_name: str,
    user_id: str,
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """Find all memories that mention a specific entity"""
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    cypher_query = """
    MATCH (entity {name: $entity_name})-[:MENTIONED_IN]->(m:Memory)
    WHERE m.user_id = $user_id
    RETURN DISTINCT m.id as memory_id
    LIMIT $limit
    """

    results = query_graph(user_id, cypher_query, {"entity_name": entity_name, "limit": limit})
    memory_ids = [UUID(r["memory_id"]) for r in results if r.get("memory_id")]

    if not memory_ids:
        return {"entity": entity_name, "memories": [], "count": 0}

    memories = db.query(Memory).filter(
        Memory.id.in_(memory_ids),
        Memory.user_id == user.id,
        Memory.state != MemoryState.deleted
    ).options(
        joinedload(Memory.categories),
        joinedload(Memory.app)
    ).all()

    return {
        "entity": entity_name,
        "memories": [
            {
                "id": str(memory.id),
                "content": memory.content,
                "created_at": memory.created_at.isoformat(),
                "app_name": memory.app.name if memory.app else None,
                "categories": [cat.name for cat in memory.categories]
            }
            for memory in memories
        ],
        "count": len(memories)
    }


@memories_router.get("/graph/visualize")
async def get_graph_visualization(
    user_id: str,
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db)
):
    """Get graph data in format suitable for visualization"""
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    try:
        memory_client = get_memory_client()
        if not memory_client or not hasattr(memory_client, 'graph'):
            return {
                "error": "Graph store not available",
                "nodes": [],
                "edges": []
            }

        node_result = query_graph(
            user_id,
            """
            MATCH (n)
            WHERE n.user_id = $user_id
            RETURN id(n) as id, n.name as name, labels(n) as types
            LIMIT $limit
            """,
            {"limit": limit}
        )

        nodes = [
            {
                "id": record["id"],
                "label": record.get("name", "Unknown"),
                "type": record["types"][0] if record.get("types") else "Unknown"
            }
            for record in node_result
        ]

        edge_result = query_graph(
            user_id,
            """
            MATCH (a)-[r]->(b)
            WHERE a.user_id = $user_id
            RETURN id(a) as source, id(b) as target, type(r) as relationship
            LIMIT $limit
            """,
            {"limit": limit}
        )

        edges = [
            {
                "source": record["source"],
                "target": record["target"],
                "label": record.get("relationship", "RELATED_TO")
            }
            for record in edge_result
        ]

        return {
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges)
        }

    except Exception as e:
        logging.error(f"Error getting graph visualization: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "nodes": [],
            "edges": []
        }


@memories_router.get("/graph/context/{topic}")
async def get_topic_context(
    topic: str,
    user_id: str,
    depth: int = Query(2, ge=1, le=3, description="How many relationship hops to traverse"),
    include_memories: bool = Query(True, description="Include related memories"),
    db: Session = Depends(get_db)
):
    """Get comprehensive context about a topic by traversing the knowledge graph.
    Perfect for AI coding agents to understand full context about a system/feature."""
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    cypher_query = f"""
    MATCH (start)
    WHERE start.user_id = $user_id
    AND toLower(start.name) CONTAINS toLower($topic)
    WITH start
    MATCH path = (start)-[*0..{depth}]-(related)
    WHERE related.user_id = $user_id
    RETURN DISTINCT
        start.name as root_entity,
        related.name as related_entity,
        labels(related) as entity_type,
        length(path) as distance
    ORDER BY distance, related_entity
    LIMIT 100
    """

    results = query_graph(user_id, cypher_query, {"topic": topic})

    context = {
        "topic": topic,
        "root_entities": list(set([r["root_entity"] for r in results])),
        "related_entities": {},
        "memories": []
    }

    for r in results:
        distance = r["distance"]
        if distance not in context["related_entities"]:
            context["related_entities"][distance] = []
        context["related_entities"][distance].append({
            "name": r["related_entity"],
            "type": r["entity_type"][0] if r["entity_type"] else "Unknown"
        })

    if include_memories and context["root_entities"]:
        for entity_name in context["root_entities"][:5]:
            entity_memories = await get_memories_by_entity(entity_name, user_id, limit=10, db=db)
            context["memories"].extend(entity_memories["memories"])

    return context


# ===========================================================================
# APPS ENDPOINTS
# ===========================================================================

def get_app_or_404(db: Session, app_id: UUID) -> App:
    app = db.query(App).filter(App.id == app_id).first()
    if not app:
        raise HTTPException(status_code=404, detail="App not found")
    return app


@apps_router.get("/")
async def list_apps(
    name: Optional[str] = None,
    is_active: Optional[bool] = None,
    sort_by: str = 'name',
    sort_direction: str = 'asc',
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    memory_counts = db.query(
        Memory.app_id,
        func.count(Memory.id).label('memory_count')
    ).filter(
        Memory.state.in_([MemoryState.active, MemoryState.paused, MemoryState.archived])
    ).group_by(Memory.app_id).subquery()

    access_counts = db.query(
        MemoryAccessLog.app_id,
        func.count(func.distinct(MemoryAccessLog.memory_id)).label('access_count')
    ).group_by(MemoryAccessLog.app_id).subquery()

    query = db.query(
        App,
        func.coalesce(memory_counts.c.memory_count, 0).label('total_memories_created'),
        func.coalesce(access_counts.c.access_count, 0).label('total_memories_accessed')
    )

    query = query.outerjoin(
        memory_counts,
        App.id == memory_counts.c.app_id
    ).outerjoin(
        access_counts,
        App.id == access_counts.c.app_id
    )

    if name:
        query = query.filter(App.name.ilike(f"%{name}%"))

    if is_active is not None:
        query = query.filter(App.is_active == is_active)

    if sort_by == 'name':
        sort_field = App.name
    elif sort_by == 'memories':
        sort_field = func.coalesce(memory_counts.c.memory_count, 0)
    elif sort_by == 'memories_accessed':
        sort_field = func.coalesce(access_counts.c.access_count, 0)
    else:
        sort_field = App.name

    if sort_direction == 'desc':
        query = query.order_by(desc(sort_field))
    else:
        query = query.order_by(sort_field)

    total = query.count()
    apps = query.offset((page - 1) * page_size).limit(page_size).all()

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "apps": [
            {
                "id": app[0].id,
                "name": app[0].name,
                "is_active": app[0].is_active,
                "total_memories_created": app[1],
                "total_memories_accessed": app[2]
            }
            for app in apps
        ]
    }


@apps_router.get("/{app_id}")
async def get_app_details(
    app_id: UUID,
    db: Session = Depends(get_db)
):
    app = get_app_or_404(db, app_id)

    access_stats = db.query(
        func.count(MemoryAccessLog.id).label("total_memories_accessed"),
        func.min(MemoryAccessLog.accessed_at).label("first_accessed"),
        func.max(MemoryAccessLog.accessed_at).label("last_accessed")
    ).filter(MemoryAccessLog.app_id == app_id).first()

    return {
        "is_active": app.is_active,
        "total_memories_created": db.query(Memory)
            .filter(Memory.app_id == app_id)
            .count(),
        "total_memories_accessed": access_stats.total_memories_accessed or 0,
        "first_accessed": access_stats.first_accessed,
        "last_accessed": access_stats.last_accessed
    }


@apps_router.get("/{app_id}/memories")
async def list_app_memories(
    app_id: UUID,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    get_app_or_404(db, app_id)
    query = db.query(Memory).filter(
        Memory.app_id == app_id,
        Memory.state.in_([MemoryState.active, MemoryState.paused, MemoryState.archived])
    )
    query = query.options(joinedload(Memory.categories))
    total = query.count()
    memories = query.order_by(Memory.created_at.desc()).offset((page - 1) * page_size).limit(page_size).all()

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "memories": [
            {
                "id": memory.id,
                "content": memory.content,
                "created_at": memory.created_at,
                "state": memory.state.value,
                "app_id": memory.app_id,
                "categories": [category.name for category in memory.categories],
                "metadata_": memory.metadata_
            }
            for memory in memories
        ]
    }


@apps_router.get("/{app_id}/accessed")
async def list_app_accessed_memories(
    app_id: UUID,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    query = db.query(
        Memory,
        func.count(MemoryAccessLog.id).label("access_count")
    ).join(
        MemoryAccessLog,
        Memory.id == MemoryAccessLog.memory_id
    ).filter(
        MemoryAccessLog.app_id == app_id
    ).group_by(
        Memory.id
    ).order_by(
        desc("access_count")
    )

    query = query.options(joinedload(Memory.categories))

    total = query.count()
    results = query.offset((page - 1) * page_size).limit(page_size).all()

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "memories": [
            {
                "memory": {
                    "id": memory.id,
                    "content": memory.content,
                    "created_at": memory.created_at,
                    "state": memory.state.value,
                    "app_id": memory.app_id,
                    "app_name": memory.app.name if memory.app else None,
                    "categories": [category.name for category in memory.categories],
                    "metadata_": memory.metadata_
                },
                "access_count": count
            }
            for memory, count in results
        ]
    }


@apps_router.put("/{app_id}")
async def update_app_details(
    app_id: UUID,
    is_active: bool,
    db: Session = Depends(get_db)
):
    app = get_app_or_404(db, app_id)
    app.is_active = is_active
    db.commit()
    return {"status": "success", "message": "Updated app details successfully"}


# ===========================================================================
# STATS ENDPOINTS
# ===========================================================================

@stats_router.get("/")
async def get_profile(
    user_id: str,
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    total_memories = db.query(Memory).filter(
        Memory.user_id == user.id,
        Memory.state != MemoryState.deleted
    ).count()

    apps = db.query(App).filter(App.owner == user)
    total_apps = apps.count()

    return {
        "total_memories": total_memories,
        "total_apps": total_apps,
        "apps": apps.all()
    }


# ===========================================================================
# Include sub-routers
# ===========================================================================

router.include_router(memories_router)
router.include_router(apps_router)
router.include_router(stats_router)
