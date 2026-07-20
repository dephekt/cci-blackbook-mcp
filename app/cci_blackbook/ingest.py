from __future__ import annotations

import argparse
import json
import sys

from .service import BlackBookService, IndexUnavailable, IngestFailed
from .settings import load_settings, voyage_configured


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or refresh the CCI Black Book index.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild every source; stop the MCP first when replacing a legacy index",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run a live Voyage smoke test on SYNTHETIC data only (no real PDF); validates "
        "connectivity, both embedding spaces, and the text-less image path",
    )
    args = parser.parse_args()

    if args.smoke:
        _run_smoke()
        return

    service = BlackBookService()
    try:
        result = service.ensure_index(force=args.force)
    except (IngestFailed, IndexUnavailable) as exc:
        print(f"ingest failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(result, indent=2, sort_keys=True))


def _run_smoke() -> None:
    """Synthetic live check through the real Voyage provider. No private content is
    sent — only made-up strings and a drawn image. Safe to run before opt-out."""
    import numpy as np
    from PIL import Image, ImageDraw

    from .embeddings import VoyageProvider
    from .render import ImageUnit

    settings = load_settings()
    if not voyage_configured():
        print("smoke failed: VOYAGE_API_KEY is not set", file=sys.stderr)
        raise SystemExit(1)

    provider = VoyageProvider(settings)

    def cos(a, b) -> float:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    doc = ["vapor pressure deficit controls transpiration", "reverse osmosis lowers source water ppm"]
    text_vecs = provider.embed_text_documents([doc])[0]
    assert len(text_vecs) == 2, "context-4 chunk count mismatch"
    assert text_vecs[0].shape[0] == settings.voyage_output_dim, "unexpected text dim"
    tq = provider.embed_text_query("how do I control transpiration?")
    assert cos(tq, text_vecs[0]) > cos(tq, text_vecs[1]), "text query did not rank the VPD chunk first"

    img = Image.new("RGB", (320, 240), (255, 255, 255))
    ImageDraw.Draw(img).rectangle([30, 30, 290, 210], outline=(0, 0, 0), width=4)
    units = [ImageUnit(1, "seedling figure", img), ImageUnit(2, "", img)]  # page 2 = text-less -> [image]
    img_vecs = provider.embed_image_units(units)
    assert len(img_vecs) == 2, "multimodal count mismatch"
    assert img_vecs[0].shape[0] == settings.voyage_output_dim, "unexpected image dim"
    assert np.isfinite(img_vecs[1]).all(), "text-less [image] path returned a bad vector"
    iq = provider.embed_image_query("seedling diagram")
    assert iq.shape[0] == settings.voyage_output_dim, "unexpected image query dim"

    print(json.dumps({
        "smoke": "ok",
        "text_dim": int(text_vecs[0].shape[0]),
        "image_dim": int(img_vecs[0].shape[0]),
        "text_query_ranks_vpd_first": True,
        "textless_image_ok": True,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
