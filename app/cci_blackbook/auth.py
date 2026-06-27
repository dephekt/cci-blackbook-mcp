from __future__ import annotations

import os
from collections.abc import Sequence
from secrets import compare_digest

try:
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response
except ModuleNotFoundError:  # Allows pure auth helpers to be unit-tested without web deps.
    BaseHTTPMiddleware = object
    Request = object
    Response = object
    JSONResponse = None


TOKEN_ENV_NAMES = ("CCI_BLACKBOOK_MCP_TOKEN", "CCI_MCP_BEARER_TOKEN")
PUBLIC_PATHS = ("/healthz",)


class BearerAuthMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        *,
        token_env_names: Sequence[str] = TOKEN_ENV_NAMES,
        public_paths: Sequence[str] = PUBLIC_PATHS,
    ):
        super().__init__(app)
        self.token_env_names = tuple(token_env_names)
        self.public_paths = tuple(public_paths)

    async def dispatch(self, request: Request, call_next) -> Response:
        if _is_public_path(request.url.path, self.public_paths):
            return await call_next(request)

        expected = expected_token(self.token_env_names)
        if not expected:
            if JSONResponse is None:
                raise RuntimeError("starlette is required for BearerAuthMiddleware")
            return JSONResponse(
                {"error": "mcp bearer token is not configured"},
                status_code=503,
            )

        if not is_authorized(request.headers.get("authorization", ""), expected):
            if JSONResponse is None:
                raise RuntimeError("starlette is required for BearerAuthMiddleware")
            return JSONResponse(
                {"error": "missing or invalid bearer token"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )

        return await call_next(request)


def expected_token(token_env_names: Sequence[str] = TOKEN_ENV_NAMES) -> str:
    for name in token_env_names:
        value = os.environ.get(name)
        if value:
            return value
    return ""


def is_authorized(header_value: str, expected: str) -> bool:
    prefix = "Bearer "
    if not header_value.startswith(prefix):
        return False
    token = header_value[len(prefix) :].strip()
    return bool(token and expected and compare_digest(token, expected))


def _is_public_path(path: str, public_paths: Sequence[str]) -> bool:
    normalized = path.rstrip("/") or "/"
    return any(normalized == public_path.rstrip("/") for public_path in public_paths)
