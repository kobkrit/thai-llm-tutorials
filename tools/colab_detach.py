#!/usr/bin/env python3
"""Launch a notebook run *detached on the Colab VM*, so no long websocket is held.

Why
---
`colab exec -f notebook.ipynb` keeps a websocket open for the whole run. A
20-40 minute training notebook reliably dies with
``RuntimeError: Connection was lost`` -- the code was fine, the pipe was not.

This sends a launcher that starts the notebook under `nohup` on the VM and
returns in seconds. Progress is then read with short poll connections, so no
single connection has to survive the run.

Usage (from the repo root, with PATH including ~/.local/bin):

    colab exec -s llm -f tools/colab_detach.py --timeout 600     # NB=02 default
    colab exec -s llm -f tools/colab_poll.py   --timeout 300     # repeat as needed
"""
import os
import pathlib
import subprocess
import sys
import textwrap

REPO = "/content/thai-llm-tutorials"
NB = os.environ.get("NB", "02")

if not os.path.isdir(REPO):
    subprocess.run(["git", "clone", "-q", "--depth", "1",
                    "https://github.com/kobkrit/thai-llm-tutorials.git", REPO], check=False)
subprocess.run(["git", "-C", REPO, "fetch", "-q", "--depth", "1", "origin", "main"], check=False)
subprocess.run(["git", "-C", REPO, "reset", "-q", "--hard", "origin/main"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nbclient", "nbformat"], check=False)

runner = pathlib.Path("/content/_runner.py")
runner.write_text(textwrap.dedent(f'''
    import glob, json, traceback
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    path = sorted(glob.glob("{REPO}/notebooks/{NB}_*.ipynb"))[0]
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=7200, kernel_name="python3",
                            allow_errors=False,
                            resources={{"metadata": {{"path": "/content"}}}})
    status = {{"notebook": path.split("/")[-1], "done": False, "error": None}}
    try:
        client.execute()
        status["ok"] = True
    except CellExecutionError:
        status["ok"] = False
        for i, cell in enumerate(nb.cells):
            hit = next((o for o in cell.get("outputs", []) if o.get("output_type") == "error"), None)
            if hit:
                status["error"] = {{
                    "cell": i,
                    "ename": hit.get("ename"),
                    "evalue": str(hit.get("evalue"))[:400],
                    "traceback": [l for l in hit.get("traceback", [])][-14:],
                    "source_head": cell.get("source", "")[:500],
                }}
                break
    except Exception as e:
        status["ok"] = False
        status["error"] = {{"ename": type(e).__name__, "evalue": str(e)[:400],
                           "traceback": traceback.format_exc().split(chr(10))[-12:]}}
    finally:
        status["done"] = True
        try:
            nbformat.write(nb, "/content/executed.ipynb")
        except Exception:
            pass
        json.dump(status, open("/content/status.json", "w"), ensure_ascii=False, indent=1)
    print("RUNNER_DONE", status.get("ok"))
'''))

for stale in ("/content/status.json", "/content/nb.log"):
    try:
        os.remove(stale)
    except OSError:
        pass

subprocess.Popen([sys.executable, "-u", str(runner)],
                 stdout=open("/content/nb.log", "w"),
                 stderr=subprocess.STDOUT,
                 start_new_session=True)
print(f"LAUNCHED notebook {NB} detached; poll with tools/colab_poll.py")
