"""
Memory Brain Agent REST API.

Single endpoint: POST /api/v1/brain
The agent determines intent (search, store, delete, update) from the request.

GET  /api/v1/brain/audit  — brain audit log
GET  /api/v1/brain/status — brain agent status
"""

import logging
import os
from typing import Any, Dict, List, Optional

from service.brain.agent import brain_agent, BrainResult
from service.database import engine
from fastapi import APIRouter, Query
from pydantic import BaseModel
from sqlalchemy import text as sql_text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/brain", tags=["brain"])


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class BrainRequest(BaseModel):
    request: str
    user_id: str


class BrainResponse(BaseModel):
    answer: str
    steps: int
    tools_called: List[str]
    user_id: str
    success: bool
    error: Optional[str] = None
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("", response_model=BrainResponse)
@router.post("/", response_model=BrainResponse)
async def brain_handle(req: BrainRequest):
    """
    Talk to the Memory Agent in natural language.

    The agent determines what to do from your request:
    - Questions -> searches vector store + knowledge graph, synthesizes answer
    - "Remember X" -> stores new facts with dedup
    - "Delete X" -> finds and removes matching memories
    - "Update X to Y" -> deletes old, stores new
    - Complex requests -> chains multiple operations

    Examples:
    - "What GPU does Steven use?"
    - "Steven has a girlfriend called Mirjam"
    - "Delete all memories about dark mode"
    - "Steven switched from UE5 to Godot"
    """
    result = brain_agent.run(
        request=req.request,
        user_id=req.user_id,
    )
    return BrainResponse(
        answer=result.answer,
        steps=result.steps,
        tools_called=result.tools_called,
        user_id=result.user_id,
        success=result.success,
        error=result.error,
        elapsed_seconds=result.elapsed_seconds,
    )


@router.get("/audit")
async def brain_audit(
    user_id: str = Query(None, description="Filter by user ID"),
    limit: int = Query(50, description="Number of rows to return"),
):
    """Return the last N brain_audit rows."""
    try:
        with engine.connect() as conn:
            if user_id:
                result = conn.execute(
                    sql_text(
                        "SELECT id, ts, tool, user_id, input, output, error "
                        "FROM brain_audit WHERE user_id = :user_id "
                        "ORDER BY id DESC LIMIT :limit"
                    ),
                    {"user_id": user_id, "limit": limit},
                )
            else:
                result = conn.execute(
                    sql_text(
                        "SELECT id, ts, tool, user_id, input, output, error "
                        "FROM brain_audit ORDER BY id DESC LIMIT :limit"
                    ),
                    {"limit": limit},
                )
            rows = [dict(row._mapping) for row in result]
        return {"rows": rows, "count": len(rows)}
    except Exception as e:
        logger.error(f"Audit query failed: {e}")
        return {"rows": [], "count": 0, "error": str(e)}


@router.get("/status")
async def brain_status():
    """Return brain agent configuration and status."""
    from service.brain.tools import TOOL_DEFINITIONS
    return {
        "model": brain_agent.model,
        "max_steps": brain_agent.max_steps,
        "tools_available": [t["name"] for t in TOOL_DEFINITIONS],
    }
