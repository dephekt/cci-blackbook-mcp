from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np

from .chunking import normalize_text


@dataclass(frozen=True)
class RenderedPage:
    page: int
    ocr_text: str          # normalize_text(page.get_text("text")); may be ""
    char_count: int
    width: int
    height: int
    ink_coverage: float    # fraction of sampled pixels with luma < ink_luma_threshold, 0..1
    color_fraction: float  # fraction of sampled pixels with (max-min channel) > color_sat_threshold
    image: Any             # PIL.Image (RGB) in prod; duck-typed .size/.width in tests


@dataclass(frozen=True)
class ImageUnit:
    page: int
    ocr_text: str
    image: Any


def visual_signals(
    image: Any, ink_luma_threshold: int, color_sat_threshold: int
) -> tuple[float, float]:
    """Cheap ink/color coverage on a 200px-wide downsample. Pure w.r.t. fitz —
    only needs a PIL image, so it is unit-testable with PIL.Image.new(...)."""
    width = image.width
    height = image.height
    sample_h = max(1, int(200 * height / width)) if width else 1
    a = np.asarray(image.convert("RGB").resize((200, sample_h)), dtype=np.int16)
    luma = a.mean(axis=2)
    ink = float((luma < ink_luma_threshold).mean())
    sat = a.max(axis=2) - a.min(axis=2)
    color = float((sat > color_sat_threshold).mean())
    return ink, color


def page_image_tokens(
    width: int, height: int, ocr_len: int, *, pixels_per_token: int, chars_per_token: float
) -> int:
    """Voyage limit-token cost of one multimodal image element (image pixels +
    interleaved OCR text). Pure/scalar, so unit-testable without fitz."""
    return ceil(width * height / pixels_per_token) + ceil(ocr_len / chars_per_token)


def _downscale_to(image: Any, max_pixels: int) -> Any:
    """Scale down (never up) so total pixels stay under max_pixels (Voyage 16 MP cap)."""
    w, h = image.width, image.height
    if max_pixels <= 0 or w * h <= max_pixels:
        return image
    scale = (max_pixels / float(w * h)) ** 0.5
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return image.resize((new_w, new_h))


def render_pages(
    path: Path,
    *,
    dpi: int,
    max_pixels: int,
    ink_luma_threshold: int,
    color_sat_threshold: int,
) -> Iterator[RenderedPage]:
    """Stream one RenderedPage at a time. The pixmap/PIL image for page N is dropped
    as the loop advances to N+1, so peak RAM is a single page, not the whole book."""
    import fitz
    from PIL import Image

    doc = fitz.open(str(path))
    try:
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for i, page in enumerate(doc, start=1):
            text = normalize_text(page.get_text("text") or "")  # reads the HiddenHorzOCR layer
            pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            img = _downscale_to(img, max_pixels)
            ink, color = visual_signals(img, ink_luma_threshold, color_sat_threshold)
            yield RenderedPage(
                page=i,
                ocr_text=text,
                char_count=len(text),
                width=img.width,
                height=img.height,
                ink_coverage=ink,
                color_fraction=color,
                image=img,
            )
    finally:
        doc.close()
