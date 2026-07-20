from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from math import ceil

from .sources import build_text_unit_id


@dataclass(frozen=True)
class PageText:
    page: int
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str          # namespaced: "<source_id>:p0012-c000"
    source_id: str
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
    source_id: str,
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
                    chunk_id=build_text_unit_id(source_id, page, chunk_index),
                    source_id=source_id,
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
    source_id: str,
    pages: Iterable[PageText],
    *,
    chunk_chars: int = 1800,
    overlap_chars: int = 250,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        chunks.extend(
            chunk_page(
                source_id,
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
    chunk_tokens: list[int],
    max_chunk_tokens: int,
) -> list[list[int]]:
    """Page-aligned greedy packing of chunk INDICES into documents <= token_budget.

    `chunk_tokens[i]` is the REAL token count of `chunks[i]` (from the embedding model's
    tokenizer), so a document is packed to the model's true context-window budget rather
    than a chars/token estimate that under-counts dense content. A page's chunks are never
    split across documents (contextualization stays within a page's neighborhood at
    minimum), every chunk lands in exactly one document, and order is preserved so returned
    indices zip back to chunk_ids positionally. The collapse key is (source_id, page) so a
    book's chunks are never contextualized with another book's, even if this is ever handed
    a multi-source list. Raises if any single chunk exceeds max_chunk_tokens.
    """
    if not chunks:
        return []
    if len(chunk_tokens) != len(chunks):
        raise ValueError("chunk_tokens must align 1:1 with chunks")

    # Collapse into per-(source, page) groups (a source's chunks arrive contiguously).
    pages: list[list] = []  # [(source_id, page), [indices], token_sum]
    for idx, chunk in enumerate(chunks):
        tok = chunk_tokens[idx]
        if tok > max_chunk_tokens:
            raise ValueError(
                f"chunk {chunk.chunk_id} is {tok} tokens, exceeds max_chunk_tokens {max_chunk_tokens}"
            )
        key = (chunk.source_id, chunk.page)
        if pages and pages[-1][0] == key:
            pages[-1][1].append(idx)
            pages[-1][2] += tok
        else:
            pages.append([key, [idx], tok])

    documents: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0
    current_source: str | None = None
    for key, indices, page_tokens in pages:
        source_id = key[0]
        # Start a new document when the source changes (never contextualize one book's
        # chunks with another's) or the budget would overflow. A page's chunks are one
        # `pages` entry, so they are never split across documents.
        if current and (source_id != current_source or current_tokens + page_tokens > token_budget):
            documents.append(current)
            current, current_tokens = [], 0
        current.extend(indices)
        current_tokens += page_tokens
        current_source = source_id
    if current:
        documents.append(current)
    return documents
