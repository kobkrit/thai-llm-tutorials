#!/usr/bin/env python3
"""Execute a course notebook headlessly on a real GPU and report the first failure.

Why this exists
---------------
The development loop was: run in Colab -> screenshot the traceback -> fix ->
re-run. Every round trip needed a human. This runs the same notebooks on the
GTX 1660 in the `web` box, which is Turing sm_75 -- the *same architecture as a
Colab T4*, so it reproduces the no-bfloat16 / no-FlashAttention-2 constraints
that most of our bugs came from.

The 1660 has 6 GB against the T4's 16 GB, so notebooks run under SMOKE
overrides: tiny datasets, 2-3 optimizer steps, short sequences. That is enough
to catch the failures we actually hit -- wrong dataset columns, chat-template
misuse on a base model, NameError, missing fonts, bad API arguments -- none of
which were memory bugs. A clean smoke run does NOT prove the full T4 config
fits; it proves the code is correct.

Usage
-----
    python3 tools/smoke_run.py 01                # one notebook
    python3 tools/smoke_run.py 01 02 04          # several
    python3 tools/smoke_run.py --all
    python3 tools/smoke_run.py 01 --full         # no shrinking (needs a big GPU)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
NB_DIR = REPO / "notebooks"
OUT_DIR = REPO / ".smoke"

# Literal rewrites applied to the notebook source before execution. Regex rather
# than injected globals because most notebooks define and *use* a config value
# inside the same cell, so a later override would arrive too late.
SHRINK: list[tuple[str, str]] = [
    (r"\bN_TRAIN\s*=\s*[^\n]+", "N_TRAIN = 16"),
    (r"\bN_DOMAIN_DOCS\s*=\s*\d+", "N_DOMAIN_DOCS = 120"),
    (r"\bN_GENERAL_DOCS\s*=\s*\d+", "N_GENERAL_DOCS = 120"),
    (r"\bMAX_SEQ_LEN\s*=\s*\d+", "MAX_SEQ_LEN = 192"),
    (r"\bMAX_STEPS\s*=\s*\d+", "MAX_STEPS = 2"),
    (r"\bN_PER_CLASS\s*=\s*\d+", "N_PER_CLASS = 40"),
    (r"\bN_AUG\s*=\s*\d+", "N_AUG = 8"),
    (r"\bN_EXAM\s*=\s*\d+", "N_EXAM = 4"),
    (r"\bN_GSM\s*=\s*\d+", "N_GSM = 4"),
    (r"\bN_REQ\s*=\s*\d+", "N_REQ = 4"),
    (r"\bGEN_TOKENS\s*=\s*\d+", "GEN_TOKENS = 16"),
    (r"\bFAST_MODE\s*=\s*\w+", "FAST_MODE = True"),
    (r"\bRUN_VLLM\s*=\s*\w+", "RUN_VLLM = False"),
    (r"per_device_train_batch_size\s*=\s*\d+", "per_device_train_batch_size=1"),
    (r"gradient_accumulation_steps\s*=\s*\d+", "gradient_accumulation_steps=1"),
    (r"num_train_epochs\s*=\s*[\d.]+", "num_train_epochs=1"),
    (r"max_new_tokens\s*=\s*\d+", "max_new_tokens=16"),
    (r"num_generations\s*=\s*\d+", "num_generations=2"),
    (r"max_completion_length\s*=\s*\d+", "max_completion_length=64"),
    # KobEval slices are 30 items each; 4 is plenty to prove the path runs.
    (r"slices=KOBEVAL_SLICES", 'slices=["TH-KNOW"]'),
    (r"\bn=30\b", "n=4"),
]

# Cell 0 pip-installs pinned packages; the server venv already has them and a
# reinstall on every run would dominate the wall clock.
SKIP_IN_CELL0 = [
    (r"^!pip install.*?(?=\n[^ \t!])", "# [smoke] pip install skipped\n", re.S | re.M),
    (r"^!apt-get[^\n]*", "# [smoke] apt-get skipped", re.M),
    (r"^!git clone[^\n]*", "pass  # [smoke] clone skipped", re.M),
    (r"^!pip install -q -e[^\n]*", "# [smoke] editable install skipped", re.M),
]


def shrink_source(src: str, is_cell0: bool, full: bool) -> str:
    if is_cell0:
        for pat, repl, *flags in SKIP_IN_CELL0:
            src = re.sub(pat, repl, src, flags=flags[0] if flags else 0)
        # the repo is already present on the server
        src = src.replace('REPO_DIR = "/content/thai-llm-tutorials"',
                          f'REPO_DIR = "{REPO}"')
    if not full:
        for pat, repl in SHRINK:
            src = re.sub(pat, repl, src)
    return src


def run(nb_id: str, full: bool, timeout: int, start_at: int) -> dict:
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    matches = sorted(NB_DIR.glob(f"{nb_id}_*.ipynb"))
    if not matches:
        return {"notebook": nb_id, "status": "not_found"}
    path = matches[0]

    nb = nbformat.read(path, as_version=4)
    code_idx = [i for i, c in enumerate(nb.cells) if c.cell_type == "code"]
    for n, i in enumerate(code_idx):
        nb.cells[i].source = shrink_source(nb.cells[i].source, n == 0, full)
    if start_at:
        keep = set(code_idx[:1]) | set(code_idx[start_at:])
        nb.cells = [c for i, c in enumerate(nb.cells)
                    if c.cell_type != "code" or i in keep]

    client = NotebookClient(nb, timeout=timeout, kernel_name="python3",
                            allow_errors=False, resources={"metadata": {"path": str(REPO)}})
    t0 = time.time()
    result = {"notebook": path.name, "shrunk": not full}
    try:
        client.execute()
        result |= {"status": "pass", "seconds": round(time.time() - t0, 1)}
    except CellExecutionError as e:
        failed, tb = None, ""
        for n, c in enumerate(nb.cells):
            for o in c.get("outputs", []):
                if o.get("output_type") == "error":
                    failed = n
                    tb = "\n".join(o.get("traceback", []))
                    break
            if failed is not None:
                break
        clean = re.sub(r"\x1b\[[0-9;]*m", "", tb)
        lines = [l for l in clean.split("\n") if l.strip()]
        result |= {
            "status": "fail",
            "seconds": round(time.time() - t0, 1),
            "cell": failed,
            "error": str(e).split("\n")[-1][:200],
            "traceback_tail": lines[-14:],
            "source": (nb.cells[failed].source[:600] if failed is not None else ""),
        }
    except Exception as e:  # kernel death, timeout, ...
        result |= {"status": "error", "error": f"{type(e).__name__}: {e}"[:300],
                   "seconds": round(time.time() - t0, 1)}

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / f"{path.stem}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    (OUT_DIR / f"{path.stem}.ipynb").write_text(json.dumps(nb, ensure_ascii=False))
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", help="notebook prefixes, e.g. 01 04")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--full", action="store_true", help="no shrinking")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--start-at", type=int, default=0, help="skip to Nth code cell")
    a = ap.parse_args()

    ids = [f"{i:02d}" for i in range(1, 11)] if a.all else a.ids
    if not ids:
        ap.error("give notebook ids or --all")

    worst = 0
    for i in ids:
        r = run(i, a.full, a.timeout, a.start_at)
        icon = {"pass": "PASS", "fail": "FAIL", "error": "ERR ", "not_found": "MISS"}[r["status"]]
        print(f"\n{'='*72}\n{icon}  {r.get('notebook', i)}  ({r.get('seconds', 0)}s)\n{'='*72}")
        if r["status"] != "pass":
            worst = 1
            print(f"cell {r.get('cell')}: {r.get('error','')}")
            for l in r.get("traceback_tail", []):
                print("  ", l[:160])
            if r.get("source"):
                print("--- cell source (head) ---")
                for l in r["source"].split("\n")[:18]:
                    print("  ", l[:150])
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
