#!/usr/bin/env python3
"""Run a course notebook on Colab in short chunks over a persistent kernel.

Two failure modes killed the naive approaches, and this avoids both:

1. `colab exec -f notebook.ipynb` holds one websocket for the whole 40-minute
   run and dies with ``RuntimeError: Connection was lost`` -- training was fine,
   the pipe was not.
2. Detaching the work with nohup frees the kernel, so Colab decides the runtime
   is idle and reclaims the entire VM, taking the background job with it.

The kernel keeps its globals between `colab exec` calls (verified), so the
notebook can be executed a few cells at a time. Every connection is short, and
the kernel is genuinely busy throughout, so nothing is reclaimed.

Run from the repo root with ~/.local/bin on PATH:

    python3 tools/colab_chunk.py 02                # whole notebook, 1 cell/call
    python3 tools/colab_chunk.py 02 --from 6       # resume at code cell 6
    python3 tools/colab_chunk.py 02 --session llm --gpu T4
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
CLONE = "/content/thai-llm-tutorials"

BOOTSTRAP = f'''
import os, subprocess, sys
if not os.path.isdir("{CLONE}"):
    subprocess.run(["git","clone","-q","--depth","1",
                    "https://github.com/kobkrit/thai-llm-tutorials.git","{CLONE}"], check=False)
subprocess.run(["git","-C","{CLONE}","fetch","-q","--depth","1","origin","main"], check=False)
subprocess.run(["git","-C","{CLONE}","reset","-q","--hard","origin/main"], check=False)
os.chdir("/content")
print("BOOTSTRAP_OK")
'''


def sh(args: list[str], timeout: int) -> tuple[int, str]:
    p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def ensure_session(session: str, gpu: str) -> bool:
    _, out = sh(["colab", "sessions"], 120)
    if f"[{session}]" in out:
        return True
    sh(["colab", "stop", "-s", session], 120)
    print(f"[chunk] creating session '{session}' on {gpu} ...", flush=True)
    _, out = sh(["colab", "new", "-s", session, "--gpu", gpu], 900)
    return "READY" in out


ARTIFACTS = ["results.json", "results_table.json", "results_before_after.json",
             "samples.json", "tokens.json", "heldout_scores.json"]


def harvest(session: str, dest: pathlib.Path) -> None:
    """Pull result files off the VM after every cell.

    `colab download` returns 0-byte files in this CLI version, so read the JSON
    through `colab exec` + base64 instead. Colab reclaims runtimes without
    warning, so making finished work durable after each cell is what stops a
    lost session from destroying a completed run.
    """
    dest.mkdir(parents=True, exist_ok=True)
    names = ", ".join(repr(n) for n in ARTIFACTS)
    reader = ("import os,base64,json\n"
              "for _n in [%s]:\n"
              "    if os.path.exists(_n):\n"
              "        print('@@F', _n, base64.b64encode(open(_n,'rb').read()).decode())\n" % names)
    try:
        rc, out = exec_src(session, reader, 600)
    except Exception:
        return   # a failed harvest must not kill the whole run
    if "Connection was lost" in out or "not found" in out:
        return
    import base64 as _b64
    for line in out.splitlines():
        if line.startswith("@@F "):
            try:
                _, name, b = line.split(" ", 2)
                (dest / name).write_bytes(_b64.b64decode(b))
            except Exception:
                pass


def exec_src(session: str, src: str, timeout: int) -> tuple[int, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        return sh(["colab", "exec", "-s", session, "-f", path,
                   "--timeout", str(timeout)], timeout + 120)
    finally:
        pathlib.Path(path).unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("nb")
    ap.add_argument("--session", default="llm")
    ap.add_argument("--gpu", default="T4")
    ap.add_argument("--from", dest="start", type=int, default=0)
    ap.add_argument("--cell-timeout", type=int, default=1500)
    ap.add_argument("--retries", type=int, default=2)
    a = ap.parse_args()

    nb_path = sorted((REPO / "notebooks").glob(f"{a.nb}_*.ipynb"))[0]
    cells = [c for c in json.loads(nb_path.read_text())["cells"] if c["cell_type"] == "code"]
    print(f"[chunk] {nb_path.name}: {len(cells)} code cells, starting at {a.start}", flush=True)

    if not ensure_session(a.session, a.gpu):
        print("[chunk] could not obtain a runtime"); return 3
    rc, out = exec_src(a.session, BOOTSTRAP, 600)
    if "BOOTSTRAP_OK" not in out:
        print("[chunk] bootstrap failed:\n" + out[-1500:]); return 3

    log = REPO / f".smoke/chunk_{a.nb}.log"
    log.parent.mkdir(exist_ok=True)
    log.write_text("")

    for i, cell in enumerate(cells):
        if i < a.start:
            continue
        src = "".join(cell["source"])
        # Cell 0 pip-installs and clones; the kernel already has the repo, but the
        # installs are still needed once, so it runs as-is the first time.
        src = src.replace('REPO_DIR = "/content/thai-llm-tutorials"', f'REPO_DIR = "{CLONE}"')
        head = next((l for l in src.split("\n") if l.strip() and not l.strip().startswith("#")), "")[:70]
        print(f"\n[chunk] cell {i}/{len(cells)-1}: {head}", flush=True)

        for attempt in range(1, a.retries + 1):
            t0 = time.time()
            rc, out = exec_src(a.session, src, a.cell_timeout)
            dt = time.time() - t0
            with log.open("a") as fh:
                fh.write(f"\n===== cell {i} (attempt {attempt}, {dt:.0f}s) =====\n{out}\n")

            if "Connection was lost" in out or "not found" in out or "appears to be lost" in out:
                # A long cell often drops the websocket while the KERNEL stays alive.
                # Probe the session for a global that cell 0 always defines (DTYPE):
                # if it survived we merely lost the pipe -> re-exec this same cell on
                # the same session and keep all state. Only if the probe shows the
                # kernel is fresh/gone do we conclude the run cannot continue.
                print(f"[chunk]   connection lost after {dt:.0f}s; probing kernel ...", flush=True)
                time.sleep(15)
                try:
                    _, probe = exec_src(
                        a.session,
                        "print('STATE_OK' if 'DTYPE' in dir() else 'STATE_LOST')", 180)
                except Exception:
                    probe = ""
                if "STATE_OK" in probe and attempt < a.retries:
                    print(f"[chunk]   kernel alive, state survived -> re-exec cell {i} "
                          f"(attempt {attempt + 1})", flush=True)
                    continue
                print(f"[chunk]   kernel state is gone; giving up "
                      f"(probe={'STATE_OK' if 'STATE_OK' in probe else 'lost'})", flush=True)
                ensure_session(a.session, a.gpu)
                exec_src(a.session, BOOTSTRAP, 600)
                print("[chunk]   NOTE: kernel state was lost; rerun from --from 0", flush=True)
                return 4

            err = re.search(r"^(\w*(?:Error|Exception))\b.*", out, re.M)
            if err and "HF_TOKEN" not in err.group(0):
                print(f"[chunk]   FAILED in {dt:.0f}s: {err.group(0)[:160]}", flush=True)
                tail = [l for l in out.split("\n") if l.strip()][-12:]
                for l in tail:
                    print("   ", l[:150])
                print(f"[chunk] full log: {log}")
                return 1

            if any(k in src for k in ("write_results", "json.dump", "results.json",
                                      "results_table", "samples.json", ".png")):
                harvest(a.session, REPO / ".smoke" / f"nb{a.nb}")
            print(f"[chunk]   ok ({dt:.0f}s)" + (f" | {out.strip().splitlines()[-1][:90]}"
                                                 if out.strip() else ""), flush=True)
            break

    print(f"\n[chunk] ALL {len(cells)} CELLS COMPLETED\n[chunk] log: {log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
