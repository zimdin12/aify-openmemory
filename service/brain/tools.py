"""
Brain Tools — low-level database access layer for the Memory Brain Agent.

8 primitive async functions that talk directly to Qdrant, Neo4j, and SQLite.
The brain agent calls these tools in a loop to fulfill natural-language requests.

All tools are sync functions (called from the async agent via run_in_executor
or directly since the underlying clients are sync). The agent loop handles
async wrapping.

These tools reuse existing connection logic from enhanced_memory.py and memory.py.
"""

import json
import logging
import os
import queue
import re
import threading
import uuid
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional

from service.database import SessionLocal, engine
from service.memory.client import get_memory_client

from sqlalchemy import text as sql_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graph extraction queue — serializes Neo4j writes to avoid race conditions
# when multiple vector_store calls fire graph extractions concurrently.
# ---------------------------------------------------------------------------
_graph_queue: queue.Queue = queue.Queue()
_graph_worker_started = False
_graph_worker_lock = threading.Lock()


def _graph_worker():
    """Process graph extractions one at a time from the queue."""
    while True:
        try:
            item = _graph_queue.get(timeout=60)
            if item is None:
                break
            fn, args, kwargs = item
            try:
                fn(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Graph extraction failed in worker: {e}")
            finally:
                _graph_queue.task_done()
        except queue.Empty:
            continue


def _ensure_graph_worker():
    """Start the graph worker thread if not already running."""
    global _graph_worker_started
    if _graph_worker_started:
        return
    with _graph_worker_lock:
        if _graph_worker_started:
            return
        t = threading.Thread(target=_graph_worker, daemon=True, name="graph-worker")
        t.start()
        _graph_worker_started = True

# ---------------------------------------------------------------------------
# Audit table (created on import if it doesn't exist)
# ---------------------------------------------------------------------------

_AUDIT_DDL = """
CREATE TABLE IF NOT EXISTS brain_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts DATETIME DEFAULT CURRENT_TIMESTAMP,
    tool TEXT NOT NULL,
    user_id TEXT,
    input TEXT,
    output TEXT,
    error TEXT
)
"""

def _ensure_audit_table():
    """Create brain_audit table if it doesn't exist."""
    try:
        with engine.connect() as conn:
            conn.execute(sql_text(_AUDIT_DDL))
            conn.commit()
    except Exception as e:
        logger.warning(f"Failed to create brain_audit table: {e}")

_ensure_audit_table()


def _log_audit(tool: str, user_id: str, input_data: Any, output_data: Any = None, error: str = None):
    """Log a brain tool invocation to the audit table."""
    try:
        with engine.connect() as conn:
            conn.execute(
                sql_text(
                    "INSERT INTO brain_audit (tool, user_id, input, output, error) "
                    "VALUES (:tool, :user_id, :input, :output, :error)"
                ),
                {
                    "tool": tool,
                    "user_id": user_id,
                    "input": json.dumps(input_data, default=str)[:4000],
                    "output": json.dumps(output_data, default=str)[:4000] if output_data is not None else None,
                    "error": str(error)[:2000] if error else None,
                },
            )
            conn.commit()
    except Exception as e:
        logger.warning(f"Audit log failed: {e}")


# ---------------------------------------------------------------------------
# Helper: get Neo4j driver from memory client
# ---------------------------------------------------------------------------

def _get_neo4j_driver():
    """Get Neo4j driver from memory client's graph store."""
    client = get_memory_client()
    if not client or not hasattr(client, 'graph') or not client.graph:
        return None
    graph = client.graph
    if not hasattr(graph, 'graph'):
        return None
    driver = getattr(graph.graph, 'driver', None)
    if not driver:
        driver = getattr(graph.graph, '_driver', None)
    return driver


# ---------------------------------------------------------------------------
# Helper: get Qdrant client
# ---------------------------------------------------------------------------

def _get_qdrant_client():
    """Get Qdrant client for direct operations."""
    from qdrant_client import QdrantClient
    host = os.environ.get("QDRANT_HOST", "mem0_store")
    port = int(os.environ.get("QDRANT_PORT", 6333))
    return QdrantClient(host=host, port=port)


def _get_collection_name():
    return os.environ.get("QDRANT_COLLECTION", "openmemory")


# ===========================================================================
# Tool 1: vector_search
# ===========================================================================

def vector_search(query: str, limit: int = 10, user_id: str = None) -> List[Dict]:
    """
    Embed query via Ollama, search Qdrant, return [{id, text, score}, ...].
    """
    client = get_memory_client()
    if not client:
        raise RuntimeError("Memory client unavailable")

    embeddings = client.embedding_model.embed(query, "search")
    filters = {"user_id": user_id} if user_id else {}

    hits = client.vector_store.search(
        query=query,
        vectors=embeddings,
        limit=limit,
        filters=filters,
    )

    results = []
    for hit in hits:
        results.append({
            "id": hit.id,
            "text": hit.payload.get("data", ""),
            "score": round(hit.score, 4),
        })

    _log_audit("vector_search", user_id, {"query": query, "limit": limit}, {"count": len(results)})
    return results


# ===========================================================================
# Tool 2: vector_store
# ===========================================================================

def vector_store(text: str, user_id: str, memory_id: str = None) -> str:
    """
    Embed text, store in Qdrant via mem0, trigger async graph extraction.
    Returns the stored memory ID.
    """
    from service.memory.enhanced import EnhancedMemoryManager

    client = get_memory_client()
    if not client:
        raise RuntimeError("Memory client unavailable")

    # Dedup check first
    emb = client.embedding_model.embed(text, "search")
    filters = {"user_id": user_id}
    hits = client.vector_store.search(query=text, vectors=emb, limit=1, filters=filters)
    if hits and hits[0].score >= 0.85:
        existing_id = hits[0].id
        _log_audit("vector_store", user_id, {"text": text[:200]},
                   {"action": "skipped_duplicate", "existing_id": existing_id, "score": hits[0].score})
        return f"DUPLICATE: Already exists (id={existing_id}, score={hits[0].score:.3f})"

    # Store via mem0 (infer=False to avoid LLM re-extraction, graph=False since we trigger separately)
    response = client.add(
        [{"role": "user", "content": text}],
        user_id=user_id,
        metadata={},
        infer=False,
        graph=False,
    )

    # Extract the stored ID
    stored_id = None
    if isinstance(response, dict):
        results = response.get("results", [])
        for r in results:
            if r.get("id"):
                stored_id = r["id"]
                break
    elif isinstance(response, list):
        for r in response:
            if isinstance(r, dict) and r.get("id"):
                stored_id = r["id"]
                break

    if not stored_id:
        stored_id = str(uuid.uuid4())

    # Also create SQLite entry
    try:
        from service.database.models import Memory, MemoryState
        from service.database.utils import get_user_and_app
        db = SessionLocal()
        try:
            user, app = get_user_and_app(db, user_id=user_id, app_id="openmemory")
            mem_uuid = uuid.UUID(stored_id)
            existing = db.query(Memory).filter(Memory.id == mem_uuid).first()
            if not existing:
                memory = Memory(
                    id=mem_uuid,
                    user_id=user.id,
                    app_id=app.id,
                    content=text,
                    state=MemoryState.active,
                )
                db.add(memory)
                db.commit()
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"SQLite entry creation failed (non-fatal): {e}")

    # Queue async graph extraction (serialized to avoid Neo4j race conditions)
    try:
        _ensure_graph_worker()
        _graph_queue.put((
            EnhancedMemoryManager._background_graph_extract,
            (client, text, user_id),
            {"context": None},
        ))
    except Exception as e:
        logger.warning(f"Graph extraction queue failed: {e}")

    _log_audit("vector_store", user_id, {"text": text[:200]}, {"stored_id": stored_id})
    return stored_id


# ===========================================================================
# Tool 3: vector_delete
# ===========================================================================

def vector_delete(ids: List[str]) -> int:
    """Delete vectors from Qdrant by ID, return count deleted."""
    client = get_memory_client()
    if not client:
        raise RuntimeError("Memory client unavailable")

    deleted = 0
    for mid in ids:
        try:
            client.delete(memory_id=mid)
            deleted += 1
        except Exception as e:
            logger.warning(f"Failed to delete vector {mid}: {e}")

    # Also mark as deleted in SQLite
    try:
        db = SessionLocal()
        try:
            from service.database.models import Memory, MemoryState
            for mid in ids:
                try:
                    mem = db.query(Memory).filter(Memory.id == uuid.UUID(mid)).first()
                    if mem:
                        mem.state = MemoryState.deleted
                        mem.deleted_at = datetime.now(UTC)
                except Exception:
                    pass
            db.commit()
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"SQLite deletion sync failed: {e}")

    _log_audit("vector_delete", None, {"ids": ids}, {"deleted": deleted})
    return deleted


# ===========================================================================
# Tool 4: graph_query (read-only)
# ===========================================================================

_MUTATING_CYPHER = re.compile(
    r'\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP)\b', re.IGNORECASE
)

def graph_query(cypher: str, user_id: str = None) -> List[Dict]:
    """
    Execute read-only Cypher on Neo4j, return rows as dicts.
    Raises ValueError if query contains mutating keywords.
    """
    if _MUTATING_CYPHER.search(cypher):
        raise ValueError(
            f"graph_query is read-only. Use graph_mutate for write operations. "
            f"Detected mutating keyword in: {cypher[:100]}"
        )

    driver = _get_neo4j_driver()
    if not driver:
        raise RuntimeError("Neo4j driver unavailable")

    results = []
    with driver.session() as session:
        records = session.run(cypher)
        for record in records:
            results.append(dict(record))

    _log_audit("graph_query", user_id, {"cypher": cypher}, {"rows": len(results)})
    return results


# ===========================================================================
# Tool 5: graph_mutate (write)
# ===========================================================================

def graph_mutate(cypher: str, user_id: str = None) -> int:
    """
    Execute write Cypher on Neo4j, return affected count.
    Logs every mutation to brain_audit.
    """
    driver = _get_neo4j_driver()
    if not driver:
        raise RuntimeError("Neo4j driver unavailable")

    with driver.session() as session:
        result = session.run(cypher)
        summary = result.consume()
        affected = (
            summary.counters.nodes_created
            + summary.counters.nodes_deleted
            + summary.counters.relationships_created
            + summary.counters.relationships_deleted
            + summary.counters.properties_set
        )

    _log_audit("graph_mutate", user_id, {"cypher": cypher}, {"affected": affected})
    return affected


# ===========================================================================
# Tool 6: sql_query (read-only)
# ===========================================================================

_MUTATING_SQL = re.compile(
    r'\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE)\b', re.IGNORECASE
)

def sql_query(sql: str, params: Dict = None) -> List[Dict]:
    """
    Execute read-only SQL on SQLite, return rows as dicts.
    Raises ValueError if SQL contains mutating keywords.
    """
    if _MUTATING_SQL.search(sql):
        raise ValueError(
            f"sql_query is read-only. Use sql_mutate for write operations. "
            f"Detected mutating keyword in: {sql[:100]}"
        )

    with engine.connect() as conn:
        result = conn.execute(sql_text(sql), params or {})
        rows = [dict(row._mapping) for row in result]

    _log_audit("sql_query", None, {"sql": sql, "params": params}, {"rows": len(rows)})
    return rows


# ===========================================================================
# Tool 7: sql_mutate (write)
# ===========================================================================

def sql_mutate(sql: str, params: Dict = None) -> int:
    """
    Execute write SQL on SQLite, return affected count.
    Logs to brain_audit.
    """
    with engine.connect() as conn:
        result = conn.execute(sql_text(sql), params or {})
        affected = result.rowcount
        conn.commit()

    _log_audit("sql_mutate", None, {"sql": sql, "params": params}, {"affected": affected})
    return affected


# ===========================================================================
# Tool 8: embed
# ===========================================================================

def embed(text: str) -> List[float]:
    """Call Ollama embedding endpoint, return raw vector (1024 floats)."""
    client = get_memory_client()
    if not client:
        raise RuntimeError("Memory client unavailable")

    vector = client.embedding_model.embed(text, "search")
    if isinstance(vector, list) and len(vector) > 0 and isinstance(vector[0], list):
        vector = vector[0]

    _log_audit("embed", None, {"text": text[:200]}, {"dims": len(vector)})
    return vector


# ===========================================================================
# Tool registry — used by the agent to look up tools by name
# ===========================================================================

TOOL_DEFINITIONS = [
    {
        "name": "vector_search",
        "description": "Search memories by semantic similarity. Returns scored results from Qdrant vector store.",
        "parameters": {
            "query": {"type": "string", "description": "Search query text", "required": True},
            "limit": {"type": "integer", "description": "Max results (default 10)", "required": False},
            "user_id": {"type": "string", "description": "User ID to filter by", "required": False},
        },
        "function": vector_search,
    },
    {
        "name": "vector_store",
        "description": "Store a memory fact in the vector store. Auto-embeds, deduplicates (skips if cosine >= 0.85), and triggers async graph extraction. Returns the stored memory ID or DUPLICATE message.",
        "parameters": {
            "text": {"type": "string", "description": "The fact to store (self-contained, with subject)", "required": True},
            "user_id": {"type": "string", "description": "User ID", "required": True},
        },
        "function": vector_store,
    },
    {
        "name": "vector_delete",
        "description": "Delete memories from vector store and SQLite by their IDs. Returns count of deleted items.",
        "parameters": {
            "ids": {"type": "array", "description": "List of memory IDs to delete", "required": True},
        },
        "function": vector_delete,
    },
    {
        "name": "graph_query",
        "description": "Execute a READ-ONLY Cypher query on the Neo4j knowledge graph. Returns rows as dictionaries. Use for exploring entities and relationships. Example: MATCH (n)-[r]->(m) WHERE toLower(n.name) CONTAINS 'steven' RETURN n.name, type(r), m.name LIMIT 10",
        "parameters": {
            "cypher": {"type": "string", "description": "Read-only Cypher query", "required": True},
            "user_id": {"type": "string", "description": "User ID for context", "required": False},
        },
        "function": graph_query,
    },
    {
        "name": "graph_mutate",
        "description": "Execute a WRITE Cypher query on Neo4j (CREATE, MERGE, DELETE, SET). Returns count of affected items. Use for adding/removing graph entities and relationships.",
        "parameters": {
            "cypher": {"type": "string", "description": "Write Cypher query", "required": True},
            "user_id": {"type": "string", "description": "User ID for context", "required": False},
        },
        "function": graph_mutate,
    },
    {
        "name": "sql_query",
        "description": "Execute a READ-ONLY SQL query on SQLite metadata database. Returns rows as dictionaries. Tables: memories (id, content, state, user_id, created_at), users, apps, brain_audit.",
        "parameters": {
            "sql": {"type": "string", "description": "Read-only SQL query", "required": True},
            "params": {"type": "object", "description": "Query parameters (optional)", "required": False},
        },
        "function": sql_query,
    },
    {
        "name": "sql_mutate",
        "description": "Execute a WRITE SQL query on SQLite (INSERT, UPDATE, DELETE). Returns count of affected rows.",
        "parameters": {
            "sql": {"type": "string", "description": "Write SQL query", "required": True},
            "params": {"type": "object", "description": "Query parameters (optional)", "required": False},
        },
        "function": sql_mutate,
    },
    {
        "name": "embed",
        "description": "Generate an embedding vector (1024 floats) for a text string using the Ollama embedding model. Use when you need raw vectors for custom similarity comparisons.",
        "parameters": {
            "text": {"type": "string", "description": "Text to embed", "required": True},
        },
        "function": embed,
    },
]

# Quick lookup by name
TOOLS = {t["name"]: t["function"] for t in TOOL_DEFINITIONS}
