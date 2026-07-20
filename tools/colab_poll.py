#!/usr/bin/env python3
"""Read the state of a detached notebook run (short connection, safe to repeat)."""
import json, os, subprocess

alive = subprocess.run(["pgrep", "-f", "_runner.py"], capture_output=True).returncode == 0
print("RUNNER:", "ALIVE" if alive else "NOT RUNNING")

if os.path.exists("/content/status.json"):
    s = json.load(open("/content/status.json"))
    print("DONE:", s.get("done"), "| OK:", s.get("ok"), "|", s.get("notebook"))
    if s.get("error"):
        e = s["error"]
        print(f"FAILED cell {e.get('cell')}: {e.get('ename')}: {e.get('evalue')}")
        for l in e.get("traceback", []):
            print("   ", l[:150])
        print("--- source head ---")
        for l in (e.get("source_head") or "").split("\n")[:14]:
            print("   ", l[:140])
else:
    print("status.json not written yet")

if os.path.exists("/content/nb.log"):
    log = open("/content/nb.log", errors="replace").read()
    print(f"--- nb.log tail ({len(log)} bytes) ---")
    print(log[-2200:])
