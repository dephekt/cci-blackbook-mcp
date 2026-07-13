from __future__ import annotations

import hashlib
import inspect
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .render import ImageUnit, page_image_tokens
from .settings import Settings, voyage_configured


class EmbeddingProvider(Protocol):
    def embed_passages(self, texts: list[str]) -> list[np.ndarray]:
        ...

    def embed_query(self, text: str) -> np.ndarray:
        ...

    def status(self) -> dict:
        ...


class HashEmbeddingProvider:
    """Small deterministic fallback used for tests and degraded operation."""

    def __init__(self, *, dim: int = 384, detail: str = "deterministic lexical fallback"):
        self.dim = dim
        self.detail = detail

    def embed_passages(self, texts: list[str]) -> list[np.ndarray]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(text)

    def status(self) -> dict:
        return {
            "backend": "hash",
            "model": "hashing-vectorizer",
            "accelerator": "cpu",
            "detail": self.detail,
            "dim": self.dim,
        }

    def _embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[idx] += sign
        return _normalize(vector)


class FastEmbedProvider:
    def __init__(
        self,
        *,
        model_name: str,
        cache_dir: Path,
        prefer_openvino: bool,
        openvino_device: str,
        render_device: Path,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.prefer_openvino = prefer_openvino
        self.openvino_device = openvino_device
        self.render_device = render_device
        self._model = None
        self._provider_detail = "not initialized"
        self._accelerator = "cpu"
        self._available_providers = _available_onnx_providers()
        self._active_session_providers: list[str] = []

    def embed_passages(self, texts: list[str]) -> list[np.ndarray]:
        model = self._ensure_model()
        return [_normalize(np.asarray(v, dtype=np.float32)) for v in model.embed(_prefix("passage", texts))]

    def embed_query(self, text: str) -> np.ndarray:
        model = self._ensure_model()
        return _normalize(np.asarray(next(model.embed([f"query: {text}"])), dtype=np.float32))

    def status(self) -> dict:
        return {
            "backend": "fastembed",
            "model": self.model_name,
            "accelerator": self._accelerator,
            "detail": self._provider_detail,
            "available_onnx_providers": self._available_providers,
            "active_session_providers": self._active_session_providers,
            "render_device_present": self.render_device.exists(),
        }

    def _ensure_model(self):
        if self._model is not None:
            return self._model

        from fastembed import TextEmbedding

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        kwargs = {
            "model_name": self.model_name,
            "cache_dir": str(self.cache_dir / "fastembed"),
        }

        openvino_available = "OpenVINOExecutionProvider" in self._available_providers
        wants_openvino = self.prefer_openvino and self.render_device.exists() and openvino_available
        if wants_openvino and _accepts_kwarg(TextEmbedding, "providers"):
            kwargs["providers"] = [
                ("OpenVINOExecutionProvider", {"device_type": self.openvino_device}),
                "CPUExecutionProvider",
            ]
            self._accelerator = "openvino"
            self._provider_detail = f"requested OpenVINOExecutionProvider/{self.openvino_device}"
        elif self.prefer_openvino:
            reasons = []
            if not self.render_device.exists():
                reasons.append(f"{self.render_device} is missing")
            if not openvino_available:
                reasons.append("OpenVINOExecutionProvider is unavailable")
            if not _accepts_kwarg(TextEmbedding, "providers"):
                reasons.append("fastembed provider override is unsupported")
            self._provider_detail = "OpenVINO not selected: " + ", ".join(reasons)
        else:
            self._provider_detail = "CPU ONNX Runtime"

        self._model = TextEmbedding(**kwargs)
        self._active_session_providers = _active_session_providers(self._model)
        if "OpenVINOExecutionProvider" in self._active_session_providers:
            self._accelerator = "openvino"
            self._provider_detail = f"active OpenVINOExecutionProvider/{self.openvino_device}"
        elif wants_openvino:
            self._accelerator = "cpu"
            self._provider_detail = (
                "OpenVINO requested but ONNX session is using "
                + ", ".join(self._active_session_providers or ["unknown provider"])
            )
        return self._model


def build_embedding_provider(
    *,
    backend: str,
    model_name: str,
    cache_dir: Path,
    openvino_device: str,
    render_device: Path,
) -> EmbeddingProvider:
    if backend == "hash":
        return HashEmbeddingProvider(detail="selected by CCI_EMBEDDING_BACKEND=hash")
    if backend not in {"auto", "fastembed", "openvino"}:
        return HashEmbeddingProvider(detail=f"unknown backend {backend!r}; using fallback")
    return FastEmbedProvider(
        model_name=model_name,
        cache_dir=cache_dir,
        prefer_openvino=backend in {"auto", "openvino"},
        openvino_device=openvino_device,
        render_device=render_device,
    )


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def _prefix(prefix: str, texts: list[str]) -> list[str]:
    return [f"{prefix}: {text}" for text in texts]


def _available_onnx_providers() -> list[str]:
    try:
        import onnxruntime as ort
    except Exception:
        return []
    try:
        return list(ort.get_available_providers())
    except Exception:
        return []


def _accepts_kwarg(callable_obj, kwarg: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    if kwarg in signature.parameters:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())


def _active_session_providers(model) -> list[str]:
    session = getattr(getattr(model, "model", None), "model", None)
    get_providers = getattr(session, "get_providers", None)
    if get_providers is None:
        return []
    try:
        return list(get_providers())
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Dense providers (Voyage dual-space) + pure, unit-testable helpers.
# ---------------------------------------------------------------------------


class VoyageUnavailable(RuntimeError):
    """Raised when a Voyage call fails or returns an unexpected shape. At ingest the
    service maps this to a loud IngestFailed; at query time it is caught and the
    affected dense ranker is dropped (retrieval degrades to the remaining rankers)."""


class DenseEmbeddingProvider(Protocol):
    text_dim: int
    image_dim: int

    def embed_text_documents(self, documents: list[list[str]]) -> list[list[np.ndarray]]:
        ...

    def embed_text_query(self, text: str) -> np.ndarray:
        ...

    def embed_image_units(self, units: list[ImageUnit]) -> list[np.ndarray]:
        ...

    def embed_image_query(self, text: str) -> np.ndarray:
        ...

    def status(self) -> dict:
        ...


def _unpack_one_document(response: Any, n: int) -> list[np.ndarray]:
    """context-4 returns response.results[0].embeddings (one inner list per document)."""
    if not response.results:
        raise VoyageUnavailable("context-4 returned no results")
    vecs = response.results[0].embeddings
    if len(vecs) != n:
        raise VoyageUnavailable(f"context-4 returned {len(vecs)} vectors for {n} chunks")
    return [_normalize(np.asarray(v, dtype=np.float32)) for v in vecs]


def _unpack_multimodal(response: Any, n: int) -> list[np.ndarray]:
    """multimodal returns response.embeddings (one vector per input element)."""
    vecs = response.embeddings
    if len(vecs) != n:
        raise VoyageUnavailable(f"multimodal returned {len(vecs)} vectors for {n} inputs")
    return [_normalize(np.asarray(v, dtype=np.float32)) for v in vecs]


def mm_element(unit: ImageUnit) -> list:
    """One multimodal input element. NEVER send an empty string alongside the image —
    a text-less figure page is embedded as [image] alone."""
    text = (unit.ocr_text or "").strip()
    return [text, unit.image] if text else [unit.image]


def pack_multimodal_batches(
    units: list[ImageUnit],
    *,
    token_budget: int,
    max_inputs: int,
    pixels_per_token: int,
    chars_per_token: float,
) -> Iterator[list[ImageUnit]]:
    """Pure reference batcher: yields batches that respect both <=max_inputs and
    <=token_budget, and never split a unit. The streaming ingest in service.py
    mirrors this logic inline (it cannot materialize every page image at once)."""
    batch: list[ImageUnit] = []
    batch_tokens = 0
    for unit in units:
        cost = page_image_tokens(
            unit.image.width, unit.image.height, len(unit.ocr_text),
            pixels_per_token=pixels_per_token, chars_per_token=chars_per_token,
        )
        if batch and (len(batch) >= max_inputs or batch_tokens + cost > token_budget):
            yield batch
            batch, batch_tokens = [], 0
        batch.append(unit)
        batch_tokens += cost
    if batch:
        yield batch


def _hash_embed(text: str, dim: int) -> np.ndarray:
    """Deterministic lexical hash vector (shared by the stub provider)."""
    vector = np.zeros(dim, dtype=np.float32)
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[idx] += sign
    return _normalize(vector)


class VoyageProvider:
    def __init__(self, settings: Settings):
        self.s = settings
        self._vo = None
        self.text_dim = self.image_dim = settings.voyage_output_dim

    def _client(self):
        if self._vo is None:
            import voyageai  # lazy → offline tests never import the SDK

            self._vo = voyageai.Client(
                max_retries=self.s.voyage_max_retries, timeout=self.s.voyage_timeout
            )
        return self._vo  # reads VOYAGE_API_KEY from env

    def embed_text_documents(self, documents: list[list[str]]) -> list[list[np.ndarray]]:
        out: list[list[np.ndarray]] = []
        for doc in documents:  # one document per request (invariant)
            resp = self._guard(lambda doc=doc: self._client().contextualized_embed(
                inputs=[doc], model=self.s.voyage_text_model, input_type="document",
                output_dimension=self.s.voyage_output_dim, output_dtype=self.s.voyage_output_dtype,
            ))
            out.append(_unpack_one_document(resp, len(doc)))
        return out

    def embed_text_query(self, text: str) -> np.ndarray:
        resp = self._guard(lambda: self._client().contextualized_embed(
            inputs=[[text]], model=self.s.voyage_text_model, input_type="query",
            output_dimension=self.s.voyage_output_dim, output_dtype=self.s.voyage_output_dtype,
        ))
        return _unpack_one_document(resp, 1)[0]

    def embed_image_units(self, units: list[ImageUnit]) -> list[np.ndarray]:
        resp = self._guard(lambda: self._client().multimodal_embed(
            inputs=[mm_element(u) for u in units], model=self.s.voyage_image_model,
            input_type="document", output_dimension=self.s.voyage_output_dim,
            output_dtype=self.s.voyage_output_dtype,
        ))
        return _unpack_multimodal(resp, len(units))

    def embed_image_query(self, text: str) -> np.ndarray:
        resp = self._guard(lambda: self._client().multimodal_embed(
            inputs=[[text]], model=self.s.voyage_image_model, input_type="query",
            output_dimension=self.s.voyage_output_dim, output_dtype=self.s.voyage_output_dtype,
        ))
        return _unpack_multimodal(resp, 1)[0]

    def _guard(self, fn):
        import voyageai

        try:
            return fn()
        except voyageai.error.VoyageError as exc:  # translate SDK errors loudly
            raise VoyageUnavailable(f"{type(exc).__name__}: {exc}") from exc

    def status(self) -> dict:
        return {
            "backend": "voyage",
            "text_model": self.s.voyage_text_model,
            "image_model": self.s.voyage_image_model,
            "dim": self.s.voyage_output_dim,
            "configured": voyage_configured(),  # never any key material
        }


class StubDenseProvider:
    """Offline, deterministic dual-space provider for tests and the synthetic smoke.

    Text uses the lexical hash (shared word tokens match, like the old bge tests).
    An image unit's vector is a hash of "{page}:{ocr_text}:{thumbnail_digest}", so a
    text-less figure page still gets a distinct, deterministic, storable+searchable
    vector — which is what makes the coverage trap testable without the real model."""

    def __init__(self, dim: int = 1024, *, fail_on: tuple[str, int] | None = None):
        self.dim = dim
        self.text_dim = self.image_dim = dim
        self._fail_on = fail_on
        self._calls: dict[str, int] = {}

    def _maybe_fail(self, method: str) -> None:
        self._calls[method] = self._calls.get(method, 0) + 1
        if self._fail_on and self._fail_on[0] == method and self._calls[method] == self._fail_on[1]:
            raise VoyageUnavailable(f"stub forced failure on {method} call {self._fail_on[1]}")

    def embed_text_documents(self, documents: list[list[str]]) -> list[list[np.ndarray]]:
        self._maybe_fail("embed_text_documents")
        return [[_hash_embed(chunk, self.dim) for chunk in doc] for doc in documents]

    def embed_text_query(self, text: str) -> np.ndarray:
        self._maybe_fail("embed_text_query")
        return _hash_embed(text, self.dim)

    def embed_image_units(self, units: list[ImageUnit]) -> list[np.ndarray]:
        self._maybe_fail("embed_image_units")
        return [self._embed_image(u) for u in units]

    def embed_image_query(self, text: str) -> np.ndarray:
        self._maybe_fail("embed_image_query")
        return _hash_embed(text, self.dim)

    def _embed_image(self, unit: ImageUnit) -> np.ndarray:
        return _hash_embed(f"{unit.page}:{unit.ocr_text}:{self._image_signature(unit.image)}", self.dim)

    @staticmethod
    def _image_signature(image: Any) -> str:
        thumb = image.convert("L").resize((8, 8))
        return hashlib.blake2b(bytes(thumb.tobytes()), digest_size=8).hexdigest()

    def status(self) -> dict:
        return {"backend": "stub", "text_model": "stub", "image_model": "stub",
                "dim": self.dim, "configured": True}


def build_dense_provider(settings: Settings) -> DenseEmbeddingProvider:
    if settings.embedding_backend == "voyage":
        return VoyageProvider(settings)
    if settings.embedding_backend == "stub":
        return StubDenseProvider(dim=settings.voyage_output_dim)
    raise ValueError(f"unknown embedding_backend {settings.embedding_backend!r}")
