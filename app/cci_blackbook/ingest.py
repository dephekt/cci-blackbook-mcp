from __future__ import annotations

import argparse
import json

from .service import BlackBookService


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or refresh the CCI Black Book index.")
    parser.add_argument("--force", action="store_true", help="rebuild even if the index is current")
    args = parser.parse_args()

    service = BlackBookService()
    result = service.ensure_index(force=args.force)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
