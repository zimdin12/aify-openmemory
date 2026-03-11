"""
Streaming HTTP reverse proxy for routing requests to sub-containers.
Supports SSE/chunked streaming (important for LLM inference).
"""

import logging

import httpx
from fastapi import Request
from starlette.responses import StreamingResponse

logger = logging.getLogger(__name__)

# Long-lived client for connection pooling
_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=10.0),
            follow_redirects=True,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
    return _client


async def close_client():
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


async def proxy_request(request: Request, target_url: str) -> StreamingResponse:
    """
    Forward an HTTP request to a target URL, streaming the response back.
    Preserves method, headers, body, query params, and streaming.
    """
    client = get_client()

    headers = dict(request.headers)
    for h in ["host", "transfer-encoding", "connection"]:
        headers.pop(h, None)

    body = await request.body()

    req = client.build_request(
        method=request.method,
        url=target_url,
        headers=headers,
        content=body if body else None,
        params=dict(request.query_params),
    )

    response = await client.send(req, stream=True)

    resp_headers = dict(response.headers)
    for h in ["transfer-encoding", "connection", "content-encoding"]:
        resp_headers.pop(h, None)

    async def stream_body():
        try:
            async for chunk in response.aiter_raw():
                yield chunk
        finally:
            try:
                await response.aclose()
            except Exception:
                pass  # Client may have disconnected

    return StreamingResponse(
        stream_body(),
        status_code=response.status_code,
        headers=resp_headers,
    )
