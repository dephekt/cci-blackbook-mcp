from __future__ import annotations

from dataclasses import dataclass

from .render import RenderedPage


@dataclass(frozen=True)
class PageDecision:
    page: int
    kept: bool
    reason: str            # kept:content | kept:visual | kept:forced | dropped:blank | dropped:forced
    char_count: int
    ink_coverage: float
    color_fraction: float


def classify_page(
    rp: RenderedPage,
    *,
    blank_min_chars: int,
    blank_max_ink: float,
    blank_max_color: float,
    force_keep: set[int] | frozenset[int],
    force_drop: set[int] | frozenset[int],
    disabled: bool,
) -> PageDecision:
    """Keep-biased, conjunctive blank filter. A page is dropped ONLY when it is
    text-poor AND ink-poor AND color-poor — a lined "Notes:" template (faint rules,
    grayscale) is dropped, while a figure/photo/UI screenshot trips ink and/or color
    and is kept even at 0 OCR chars. Operator force lists win over the heuristic."""
    signals = dict(
        char_count=rp.char_count,
        ink_coverage=rp.ink_coverage,
        color_fraction=rp.color_fraction,
    )
    if rp.page in force_keep:
        return PageDecision(rp.page, True, "kept:forced", **signals)
    if rp.page in force_drop:
        return PageDecision(rp.page, False, "dropped:forced", **signals)
    if disabled:
        return PageDecision(rp.page, True, "kept:content", **signals)
    if rp.char_count >= blank_min_chars:
        return PageDecision(rp.page, True, "kept:content", **signals)
    if rp.ink_coverage >= blank_max_ink or rp.color_fraction >= blank_max_color:
        return PageDecision(rp.page, True, "kept:visual", **signals)
    return PageDecision(rp.page, False, "dropped:blank", **signals)


def summarize(decisions: list[PageDecision]) -> dict:
    kept = sum(1 for d in decisions if d.kept)
    by_reason: dict[str, int] = {}
    for d in decisions:
        by_reason[d.reason] = by_reason.get(d.reason, 0) + 1
    dropped_pages = [d.page for d in decisions if not d.kept]
    return {
        "pages": len(decisions),
        "kept": kept,
        "dropped": len(decisions) - kept,
        "by_reason": by_reason,
        "dropped_pages": dropped_pages[:200],
    }
