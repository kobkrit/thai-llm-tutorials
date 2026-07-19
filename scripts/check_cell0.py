#!/usr/bin/env python3
"""Assert that the setup cell is byte-identical across every notebook.

This is the guard that keeps the series comparable. If post 7 quietly pins a
different transformers version or drops the fp16 override, every number in that
post stops being comparable with the other nine -- and nothing else in the repo
would notice.

Run:  python3 scripts/check_cell0.py
Exit: 0 if all notebooks match notebooks/cell0_setup.ipy, 1 otherwise.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"
CELL0_PATH = NOTEBOOKS / "cell0_setup.ipy"


def cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    return src if isinstance(src, str) else "".join(src)


def main() -> int:
    if not CELL0_PATH.exists():
        print(f"missing canonical setup cell: {CELL0_PATH}")
        return 1

    canonical = CELL0_PATH.read_text(encoding="utf-8").rstrip("\n")
    canonical_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    print(f"canonical cell0: {CELL0_PATH.name}  sha256={canonical_hash[:16]}...  "
          f"({len(canonical.splitlines())} lines)\n")

    notebooks = sorted(NOTEBOOKS.glob("*.ipynb"))
    if not notebooks:
        print("no notebooks found")
        return 1

    failures = 0
    for path in notebooks:
        try:
            nb = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"  FAIL  {path.name}: invalid notebook JSON: {exc}")
            failures += 1
            continue

        code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
        if not code_cells:
            print(f"  FAIL  {path.name}: no code cells")
            failures += 1
            continue

        # "Cell 0" is the first CODE cell; the notebook's literal first cell is
        # the markdown header carrying the Colab badge.
        actual = cell_source(code_cells[0]).rstrip("\n")
        actual_hash = hashlib.sha256(actual.encode("utf-8")).hexdigest()

        if actual_hash == canonical_hash:
            print(f"  ok    {path.name}")
        else:
            failures += 1
            print(f"  FAIL  {path.name}: setup cell differs (sha256={actual_hash[:16]}...)")
            canon_lines = canonical.splitlines()
            actual_lines = actual.splitlines()
            for i, (a, b) in enumerate(zip(canon_lines, actual_lines)):
                if a != b:
                    print(f"          first difference at line {i + 1}:")
                    print(f"            canonical: {a!r}")
                    print(f"            notebook : {b!r}")
                    break
            else:
                print(f"          line count differs: canonical={len(canon_lines)} "
                      f"notebook={len(actual_lines)}")

    print()
    if failures:
        print(f"FAIL: {failures}/{len(notebooks)} notebook(s) have a divergent setup cell")
        print("Fix by regenerating: python3 scripts/make_notebook.py --slug <slug> ...")
        return 1
    print(f"PASS: all {len(notebooks)} notebook(s) share a byte-identical setup cell")
    return 0


if __name__ == "__main__":
    sys.exit(main())
