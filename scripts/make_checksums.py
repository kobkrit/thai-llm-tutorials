#!/usr/bin/env python3
"""Regenerate data/kobeval_th/checksums.json.

Run this ONLY when intentionally publishing a new benchmark version, and bump
BENCHMARK_VERSION when you do. Silently regenerating after an edit defeats the
purpose: the checksum exists so that published numbers stay comparable.

Run:  python3 scripts/make_checksums.py
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from kobeval.data import sha256_of  # noqa: E402

VERSION = "kobeval-th-v1"
DATA = pathlib.Path(__file__).resolve().parent.parent / "data" / "kobeval_th"
FILES = ["th_know.jsonl", "th_math.jsonl", "th_instr.jsonl", "th_safe.jsonl"]


def main() -> int:
    files = {}
    counts = {}
    for name in FILES:
        path = DATA / name
        if not path.exists():
            print(f"missing: {path}")
            return 1
        files[name] = sha256_of(path)
        counts[name] = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())

    manifest = {
        "version": VERSION,
        "n_items": sum(counts.values()),
        "items_per_slice": counts,
        "license": "CC BY-NC-SA 4.0",
        "algorithm": "sha256",
        "files": files,
    }

    out = DATA / "checksums.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    for name, digest in files.items():
        print(f"  {name:16s} {counts[name]:3d} items  {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
