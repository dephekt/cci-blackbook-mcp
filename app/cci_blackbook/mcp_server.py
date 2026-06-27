from __future__ import annotations

from fastmcp import FastMCP
from starlette.middleware import Middleware
from starlette.responses import JSONResponse

from .auth import BearerAuthMiddleware, expected_token
from .service import BlackBookService
from .settings import load_settings

settings = load_settings()
service = BlackBookService(settings)
mcp = FastMCP("CCI Black Book")


@mcp.tool
def ask_blackbook(
    question: str,
    crop_context: str | None = None,
    facility_context: str | None = None,
    max_citations: int = 6,
) -> dict:
    """Return a bounded cited evidence pack for a grow question."""
    return service.ask(
        question,
        crop_context=crop_context,
        facility_context=facility_context,
        max_citations=max_citations,
    )


@mcp.tool
def blackbook_search(query: str, limit: int = 10, mode: str = "hybrid") -> dict:
    """Search the CCI Black Book with fts, vector, or hybrid retrieval."""
    return service.search(query, limit=limit, mode=mode)


@mcp.tool
def blackbook_read_citation(chunk_id: str) -> dict:
    """Read one bounded citation chunk by chunk_id."""
    return service.read_citation(chunk_id)


@mcp.tool
def blackbook_status() -> dict:
    """Return source, index, and embedding backend status without secrets."""
    return service.status()


@mcp.custom_route("/healthz", methods=["GET"])
async def healthz(_request):
    status = service.status()
    auth_configured = bool(expected_token())
    healthy = bool(status["source"]["exists"] and auth_configured)
    return JSONResponse(
        {
            "ok": healthy,
            "service": "cci-blackbook-mcp",
            "source_exists": status["source"]["exists"],
            "index_ready": status["index"].get("ready", False),
            "auth_configured": auth_configured,
        },
        status_code=200 if healthy else 503,
    )


app = mcp.http_app(
    path="/mcp",
    middleware=[
        Middleware(BearerAuthMiddleware),
    ],
)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port, log_level=settings.log_level)


if __name__ == "__main__":
    main()
