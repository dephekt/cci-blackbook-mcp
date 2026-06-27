from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class PageText:
    page: int
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    page: int
    chunk_index: int
    text: str
    char_start: int
    char_end: int


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def chunk_page(
    page: int,
    text: str,
    *,
    chunk_chars: int = 1800,
    overlap_chars: int = 250,
) -> list[Chunk]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be non-negative")
    if overlap_chars >= chunk_chars:
        raise ValueError("overlap_chars must be smaller than chunk_chars")

    chunks: list[Chunk] = []
    cursor = 0
    chunk_index = 0

    while cursor < len(normalized):
        target_end = min(len(normalized), cursor + chunk_chars)
        end = _nearest_boundary(normalized, cursor, target_end)
        chunk_text = normalized[cursor:end].strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    chunk_id=f"p{page:04d}-c{chunk_index:03d}",
                    page=page,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    char_start=cursor,
                    char_end=end,
                )
            )
            chunk_index += 1

        if end >= len(normalized):
            break
        cursor = max(cursor + 1, end - overlap_chars)
        cursor = _advance_to_word_boundary(normalized, cursor)

    return chunks


def chunk_pages(
    pages: Iterable[PageText],
    *,
    chunk_chars: int = 1800,
    overlap_chars: int = 250,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        chunks.extend(
            chunk_page(
                page.page,
                page.text,
                chunk_chars=chunk_chars,
                overlap_chars=overlap_chars,
            )
        )
    return chunks


def _nearest_boundary(text: str, start: int, target_end: int) -> int:
    if target_end >= len(text):
        return len(text)

    search_floor = max(start + 1, target_end - 300)
    for marker in ("\n\n", "\n", ". ", "; ", ", "):
        boundary = text.rfind(marker, search_floor, target_end)
        if boundary != -1:
            return boundary + len(marker)

    space = text.rfind(" ", search_floor, target_end)
    if space != -1:
        return space + 1
    return target_end


def _advance_to_word_boundary(text: str, cursor: int) -> int:
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    return cursor
