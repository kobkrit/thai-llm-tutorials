"""Loading and integrity-checking the KobEval-TH benchmark."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
from typing import Dict, List

__all__ = ["SLICES", "load_slice", "load_slices", "data_dir", "sha256_of", "verify_checksums"]

SLICES = ["TH-KNOW", "TH-MATH", "TH-INSTR", "TH-SAFE"]

_FILENAMES = {
    "TH-KNOW": "th_know.jsonl",
    "TH-MATH": "th_math.jsonl",
    "TH-INSTR": "th_instr.jsonl",
    "TH-SAFE": "th_safe.jsonl",
}

_RAW_BASE = "https://raw.githubusercontent.com/kobkrit/thai-llm-tutorials/main/data/kobeval_th"


def data_dir() -> pathlib.Path:
    """Locate data/kobeval_th.

    Resolution order:
      1. ``$KOBEVAL_DATA`` if set (lets Colab point at /content/... after a wget),
      2. ``<repo>/data/kobeval_th`` relative to this file (editable install / clone),
      3. ``./data/kobeval_th`` relative to the current working directory,
      4. ``./kobeval_th`` -- the layout you get from an unzipped download.
    """
    env = os.environ.get("KOBEVAL_DATA")
    if env:
        return pathlib.Path(env).expanduser().resolve()

    candidates = [
        pathlib.Path(__file__).resolve().parent.parent / "data" / "kobeval_th",
        pathlib.Path.cwd() / "data" / "kobeval_th",
        pathlib.Path.cwd() / "kobeval_th",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return candidates[0]


def _download(slice_name: str, target: pathlib.Path) -> None:
    """Last-resort fetch so a bare `!wget kobeval` in Colab still works."""
    import urllib.request

    url = f"{_RAW_BASE}/{_FILENAMES[slice_name]}"
    target.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, target)


def load_slice(slice_name: str, allow_download: bool = True) -> List[dict]:
    """Load one slice as a list of dicts, in file order (order is part of the spec)."""
    if slice_name not in _FILENAMES:
        raise KeyError(f"unknown slice {slice_name!r}; expected one of {SLICES}")

    path = data_dir() / _FILENAMES[slice_name]
    if not path.exists():
        if not allow_download:
            raise FileNotFoundError(
                f"{path} not found. Clone the repo, or set KOBEVAL_DATA to the "
                f"directory holding {_FILENAMES[slice_name]}."
            )
        _download(slice_name, path)

    items = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} is not valid JSON: {exc}") from exc

    if len(items) != 30:
        raise ValueError(
            f"{slice_name} must contain exactly 30 items, found {len(items)}. "
            "The benchmark is versioned; a changed count means a changed benchmark."
        )
    return items


def load_slices(slice_names: List[str] | None = None) -> Dict[str, List[dict]]:
    """Load several slices at once. ``None`` means all four."""
    return {name: load_slice(name) for name in (slice_names or SLICES)}


def sha256_of(path: pathlib.Path) -> str:
    """SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checksums(strict: bool = False) -> dict:
    """Check the on-disk slices against data/kobeval_th/checksums.json.

    Any mismatch means results are not comparable with published numbers -- the
    most likely cause being that someone edited an item to make their model look
    better. Returns a report; raises only when ``strict``.
    """
    directory = data_dir()
    manifest_path = directory / "checksums.json"
    if not manifest_path.exists():
        report = {"ok": False, "reason": "checksums.json not found", "files": {}}
        if strict:
            raise FileNotFoundError(report["reason"])
        return report

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files: dict[str, dict] = {}
    ok = True
    for filename, expected in manifest.get("files", {}).items():
        path = directory / filename
        if not path.exists():
            files[filename] = {"ok": False, "reason": "missing"}
            ok = False
            continue
        actual = sha256_of(path)
        matched = actual == expected
        ok = ok and matched
        files[filename] = {"ok": matched, "expected": expected, "actual": actual}

    report = {"ok": ok, "version": manifest.get("version"), "files": files}
    if strict and not ok:
        raise ValueError(f"KobEval-TH checksum mismatch: {report}")
    return report
