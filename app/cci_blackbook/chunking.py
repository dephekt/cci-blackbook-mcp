from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from math import ceil


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


def estimate_tokens(text: str, chars_per_token: float) -> int:
    return max(1, ceil(len(text) / chars_per_token))


def group_chunks_into_documents(
    chunks: list[Chunk],
    *,
    token_budget: int,
    chars_per_token: float,
    max_chunk_tokens: int,
) -> list[list[int]]:
    """Page-aligned greedy packing of chunk INDICES into documents <= token_budget.

    A page's chunks are never split across documents (contextualization stays
    within a page's neighborhood at minimum), every chunk lands in exactly one
    document, and order is preserved so returned indices zip back to chunk_ids
    positionally. Raises if any single chunk exceeds max_chunk_tokens.
    """
    if not chunks:
        return []

    # Collapse into per-page groups (chunk_pages emits a page's chunks contiguously).
    pages: list[list] = []  # [page, [indices], token_sum]
    for idx, chunk in enumerate(chunks):
        tok = estimate_tokens(chunk.text, chars_per_token)
        if tok > max_chunk_tokens:
            raise ValueError(
                f"chunk {chunk.chunk_id} is ~{tok} tokens, exceeds max_chunk_tokens {max_chunk_tokens}"
            )
        if pages and pages[-1][0] == chunk.page:
            pages[-1][1].append(idx)
            pages[-1][2] += tok
        else:
            pages.append([chunk.page, [idx], tok])

    documents: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0
    for _page, indices, page_tokens in pages:
        if current and current_tokens + page_tokens > token_budget:
            documents.append(current)
            current, current_tokens = [], 0
        current.extend(indices)
        current_tokens += page_tokens
    if current:
        documents.append(current)
    return documents
