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
    sources: list[str] | str | None = None,
) -> dict:
    """Return a bounded cited evidence pack for a grow question.

    sources: optional source id (or list of ids) to scope the answer to specific
    book(s); default searches the whole corpus. Ids come from blackbook_status().
    """
    return service.ask(
        question,
        crop_context=crop_context,
        facility_context=facility_context,
        max_citations=max_citations,
        sources=sources,
    )


@mcp.tool
def blackbook_search(
    query: str,
    limit: int = 10,
    mode: str = "hybrid",
    sources: list[str] | str | None = None,
) -> dict:
    """Search the indexed corpus.

    mode: "hybrid" (BM25 + text-dense + image-dense, RRF-fused; default), "vector"
    (both dense spaces), "fts", "text" (text-dense only), or "image" (page-image
    dense only). Results include text chunks (unit_type="text") and page images
    (unit_type="image"), each labeled with its source_id/source_title.

    sources: optional source id or list of ids to scope the query to specific book(s)
    (default: all). Ids come from blackbook_status()["sources"][*].source_id.
    """
    return service.search(query, limit=limit, mode=mode, sources=sources)


@mcp.tool
def blackbook_read_citation(chunk_id: str) -> dict:
    """Read one bounded citation unit by id.

    Accepts a namespaced text id ("aroya-guide-to-drying:p0012-c000") or page-image
    id ("aroya-guide-to-drying:p0012-img"). A bare id (no source prefix) resolves only
    when a single source is indexed.
    """
    return service.read_citation(chunk_id)


@mcp.tool
def blackbook_status() -> dict:
    """Return source-directory, indexed-sources (with per-book counts), index, and
    embedding backend status without secrets."""
    return service.status()


@mcp.custom_route("/healthz", methods=["GET"])
async def healthz(_request):
    status = service.status()
    auth_configured = bool(expected_token())
    sd = status["source_dir"]
    source_present = sd["exists"] and sd["pdf_count"] > 0
    healthy = bool(source_present and auth_configured)
    return JSONResponse(
        {
            "ok": healthy,
            "service": "cci-blackbook-mcp",
            "source_dir_exists": sd["exists"],
            "pdf_count": sd["pdf_count"],
            "sources_indexed": len(status.get("sources", [])),
            "index_ready": status["index"].get("ready", False),
            "auth_configured": auth_configured,
            "voyage_configured": status.get("voyage_configured", False),
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
