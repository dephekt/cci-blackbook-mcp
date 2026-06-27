from __future__ import annotations

import hashlib
import inspect
import re
from pathlib import Path
from typing import Protocol

import numpy as np


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
