#!/usr/bin/env python3
"""Independently re-solve every TH-MATH item and assert it matches the JSONL.

The solutions below were written by reading each Thai prompt and translating it
into arithmetic from scratch -- they are NOT derived from the ``answer`` field.
That is the whole point: if the JSONL and this file disagree, one of them has a
bug, and the benchmark is not shipped until they agree.

Run:  python3 scripts/verify_math.py
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

DATA = pathlib.Path(__file__).resolve().parent.parent / "data" / "kobeval_th" / "th_math.jsonl"


def solve_008() -> int:
    """พ่อแก่กว่าลูก 28 ปี; อีก 6 ปี พ่อ = 3 เท่าของลูก. Solve by search."""
    for child in range(1, 200):
        father = child + 28
        if father + 6 == 3 * (child + 6):
            return child
    raise AssertionError("no solution")


SOLUTIONS = {
    "TH-MATH-001": lambda: 500 - 12 * 15,
    "TH-MATH-002": lambda: 240 - (240 * 3) // 8,
    "TH-MATH-003": lambda: 60 * 150 // 60,  # 2h30m = 150 minutes
    "TH-MATH-004": lambda: 45 - (45 * 5) // 9,
    "TH-MATH-005": lambda: (5 * 24 * 20) - (5 * 360),
    "TH-MATH-006": lambda: 180 // 15,
    "TH-MATH-007": lambda: 200 - (200 * 3) // 5,
    "TH-MATH-008": solve_008,
    # Percentages use exact integer arithmetic, never float. int(480 * 0.45)
    # is 215, not 216, because 0.45 is not representable in binary floating
    # point -- a trap worth keeping visible in a teaching repo.
    "TH-MATH-009": lambda: 850 * (100 - 20) // 100,
    "TH-MATH-010": lambda: 2 * (18 + 25),
    "TH-MATH-011": lambda: (6 * 12) // 9,
    "TH-MATH-012": lambda: 156 // 12,
    "TH-MATH-013": lambda: 1000 - (129 + 245 + 316),
    "TH-MATH-014": lambda: 350 // 14,
    "TH-MATH-015": lambda: 8 * 7 - 43,
    "TH-MATH-016": lambda: (96 - 6) // 3,
    "TH-MATH-017": lambda: 1200 * 107 // 100,
    "TH-MATH-018": lambda: 3 * 20,
    "TH-MATH-019": lambda: 1250 // 25,
    "TH-MATH-020": lambda: (96 // 12) * 30,
    "TH-MATH-021": lambda: 128 + (128 + 44) + (128 + 44) // 2,
    "TH-MATH-022": lambda: 25 * 72,
    "TH-MATH-023": lambda: math.ceil(500 / 18),
    "TH-MATH-024": lambda: 240 * 35,
    "TH-MATH-025": lambda: 150 + 8 * 9,
    "TH-MATH-026": lambda: 480 * (100 - 55) // 100,
    "TH-MATH-027": lambda: (3 * 48) // 6,
    "TH-MATH-028": lambda: (13 * 60 + 20) - (8 * 60 + 45),
    "TH-MATH-029": lambda: 20000 * 3 * 2 // 100,
    "TH-MATH-030": lambda: 4 * 65 - 187,
}


def main() -> int:
    items = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]

    failures: list[str] = []
    ids_seen: set[str] = set()

    for item in items:
        item_id = item["id"]
        if item_id in ids_seen:
            failures.append(f"{item_id}: duplicate id")
        ids_seen.add(item_id)

        if item_id not in SOLUTIONS:
            failures.append(f"{item_id}: no independent solution written")
            continue

        expected = SOLUTIONS[item_id]()
        stated = item["answer"]

        if not isinstance(stated, int):
            failures.append(f"{item_id}: answer {stated!r} is not an int")
        elif expected != stated:
            failures.append(f"{item_id}: JSONL says {stated}, independent solve says {expected}")
        else:
            print(f"  ok  {item_id}  answer={stated}")

    missing = set(SOLUTIONS) - ids_seen
    if missing:
        failures.append(f"solutions written for ids not in the JSONL: {sorted(missing)}")

    print()
    if failures:
        print(f"FAIL: {len(failures)} problem(s)")
        for f in failures:
            print("   -", f)
        return 1

    print(f"PASS: all {len(items)} TH-MATH answers verified by independent solve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
