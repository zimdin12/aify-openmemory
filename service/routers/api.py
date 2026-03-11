"""
API Router - Your service endpoints go here.

This is the main file an AI agent should modify when building a service.
Add your domain-specific endpoints below.

Example: If building a llama.cpp service, you would add:
  - POST /completions - Text completion
  - POST /chat/completions - Chat completion (OpenAI-compatible)
  - POST /embeddings - Generate embeddings
  - GET /models - List available models
  - POST /models/load - Load a model
"""

from fastapi import APIRouter

router = APIRouter(tags=["api"])


@router.get("/")
async def root():
    """Service API root. Lists available operations."""
    return {
        "message": "Service API is running. Add your endpoints here.",
        "hint": "Edit service/routers/api.py to add domain-specific endpoints.",
    }


# ---------------------------------------------------------------------------
# TODO: Add your service endpoints below
# ---------------------------------------------------------------------------
#
# @router.post("/your-endpoint")
# async def your_endpoint(request: YourRequestModel):
#     """Description of what this endpoint does."""
#     pass
