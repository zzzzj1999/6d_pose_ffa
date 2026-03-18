from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

from fafa.common import write_jsonl


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a simple JSONL image index for FFT style images.")
    parser.add_argument("--root", type=str, required=True, help="Root directory to scan")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--recursive", action="store_true", help="Scan recursively")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    paths: List[Path]
    if args.recursive:
        paths = sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    else:
        paths = sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])

    records = [{"image": os.path.relpath(p, output.parent)} for p in paths]
    write_jsonl(records, output)
    print(f"Wrote {len(records)} image records to {output}")


if __name__ == "__main__":
    main()
