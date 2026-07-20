from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Stdlib-only (no fitz/voyageai/render imports) so importing this module never pulls in
# the Voyage SDK — preserving the offline-import guarantee. Owns corpus discovery,
# slug/title derivation, and ALL unit-id construction/parsing.


@dataclass(frozen=True)
class SourceMeta:
    id: str          # slug, ^[a-z0-9][a-z0-9-]*$, never contains ':'
    title: str
    path: Path
    size: int
    mtime_ns: int
    ordinal: int     # 0-based position in filename-sorted corpus (stable display order)


_SLUG_STRIP = re.compile(r"[^a-z0-9]+")
_TEXT_LOCAL = re.compile(r"^p(\d+)-c(\d+)$")
_IMG_LOCAL = re.compile(r"^p(\d+)-img$")
_SID_RE = re.compile(r"[a-z0-9][a-z0-9-]*")


def slugify(stem: str) -> str:
    slug = _SLUG_STRIP.sub("-", stem.lower()).strip("-")
    return slug or "source"


def titleize(stem: str) -> str:
    words = [w for w in re.split(r"[\s_\-]+", stem.strip()) if w]
    if not words:
        return "Source"
    # Capitalize all-lowercase words; preserve any word already carrying uppercase
    # (acronyms/mixed-case: CCI, pH) verbatim.
    return " ".join(w if any(c.isupper() for c in w) else w[:1].upper() + w[1:] for w in words)


def discover_sources(source_dir: Path) -> list[SourceMeta]:
    if not source_dir.is_dir():
        return []
    files = sorted(
        (
            p
            for p in source_dir.iterdir()
            if p.is_file() and not p.name.startswith(".") and p.suffix.lower() == ".pdf"
        ),
        key=lambda p: p.name,
    )
    taken: set[str] = set()
    out: list[SourceMeta] = []
    for ordinal, p in enumerate(files):
        base = slugify(p.stem)
        sid, n = base, 2
        while sid in taken:  # deterministic collision disambiguation
            sid, n = f"{base}-{n}", n + 1
        taken.add(sid)
        st = p.stat()
        out.append(SourceMeta(sid, titleize(p.stem), p, st.st_size, st.st_mtime_ns, ordinal))
    return out


def build_text_unit_id(source_id: str, page: int, chunk_index: int) -> str:
    return f"{source_id}:p{page:04d}-c{chunk_index:03d}"


def build_image_unit_id(source_id: str, page: int) -> str:
    return f"{source_id}:p{page:04d}-img"


def parse_unit_id(uid: str) -> tuple[str, int, str] | None:
    """('source_id', page, 'text'|'image') or None. Split on the FIRST ':' (a slug
    never contains ':', and neither does the local part)."""
    if ":" not in uid:
        return None
    source_id, local = uid.split(":", 1)
    if not _SID_RE.fullmatch(source_id):
        return None
    if (m := _TEXT_LOCAL.match(local)):
        return (source_id, int(m.group(1)), "text")
    if (m := _IMG_LOCAL.match(local)):
        return (source_id, int(m.group(1)), "image")
    return None
