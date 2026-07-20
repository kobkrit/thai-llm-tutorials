"""Plots for the tutorial series.

matplotlib is imported lazily so that importing ``kobeval`` on a machine without
it (or in a headless CI job) still works.

Thai glyphs: matplotlib's default DejaVu Sans has no Thai coverage and renders
every Thai character as a tofu box. ``use_thai_font()`` tries the fonts that
actually exist on Colab and falls back to English labels rather than shipping a
chart full of boxes.
"""

from __future__ import annotations

import json
import pathlib
from typing import Sequence

__all__ = ["plot_before_after", "use_thai_font", "plot_slice_bars"]

_THAI_FONT_CANDIDATES = [
    "Noto Sans Thai",
    "Sarabun",
    "TH Sarabun New",
    "Loma",           # fonts-thai-tlwg
    "Garuda",
    "Norasi",
    "Waree",
    "Kinnari",
    "Umpush",
    "TlwgTypist",
    "Thonburi",       # macOS
    "Tahoma",
]

# Filename fragments of Thai-capable font files, used when rescanning the disk.
_THAI_FILE_HINTS = (
    "thai", "loma", "garuda", "norasi", "waree", "kinnari", "umpush",
    "purisa", "sawasdee", "tlwg", "sarabun", "thonburi", "notosansthai",
)


def _rescan_system_fonts() -> None:
    """Register fonts that appeared after matplotlib built its cache.

    This is the whole bug: Cell 0 runs ``apt-get install fonts-thai-tlwg``, but
    matplotlib has already built ``fontManager.ttflist`` from a cache written
    before those files existed. Nothing re-reads it, so the Thai fonts are
    invisible, ``use_thai_font()`` finds no match, and every Thai label renders
    as a tofu box with only a UserWarning to show for it.

    ``findSystemFonts`` walks the filesystem directly rather than the cache, so
    it sees the new files.
    """
    from matplotlib import font_manager

    known = {f.fname for f in font_manager.fontManager.ttflist}
    for path in font_manager.findSystemFonts(fontext="ttf"):
        if path in known:
            continue
        if any(hint in path.lower() for hint in _THAI_FILE_HINTS):
            try:
                font_manager.fontManager.addfont(path)
            except Exception:  # a broken font file must not kill the plot
                pass


def use_thai_font(rescan: bool = True) -> str | None:
    """Select an installed Thai-capable font for matplotlib.

    Returns the font name, or None if none is available. On a fresh Colab the
    fonts come from Cell 0:

        !apt-get install -y fonts-thai-tlwg > /dev/null

    Because that install happens after matplotlib is importable, we rescan the
    font directories before looking (see :func:`_rescan_system_fonts`).
    """
    try:
        import matplotlib
        from matplotlib import font_manager
    except ImportError:
        return None

    if rescan:
        try:
            _rescan_system_fonts()
        except Exception:
            pass

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _THAI_FONT_CANDIDATES:
        if name in available:
            # Keep DejaVu behind it so Latin/maths glyphs the Thai font lacks
            # (β, ≥, …) still resolve instead of becoming tofu themselves.
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name
    return None


def plot_before_after(
    before: dict,
    after: dict,
    slices: Sequence[str] | None = None,
    title: str = "Before vs After",
    out_path: str | pathlib.Path | None = "before_after.png",
    results_json: str | pathlib.Path | None = "results.json",
    show: bool = True,
):
    """Grouped bar chart of two ``evaluate()`` reports, with Wilson error bars.

    The error bars are the point of this chart. On 30-item slices a bar going
    from 40% to 47% looks like progress until you draw the intervals and see
    them overlap almost completely. Drawing them makes the honest reading the
    default reading.

    Also writes ``results_json`` so the blog's React widgets read real numbers
    rather than hand-copied ones.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    use_thai_font()

    names = list(slices) if slices else [s for s in before["slices"] if s in after["slices"]]
    x = np.arange(len(names))
    width = 0.36

    def series(rep):
        acc, lo, hi = [], [], []
        for name in names:
            s = rep["slices"][name]
            acc.append(100 * s["accuracy"])
            lo.append(100 * (s["accuracy"] - s["ci_low"]))
            hi.append(100 * (s["ci_high"] - s["accuracy"]))
        return np.array(acc), np.array([lo, hi])

    before_acc, before_err = series(before)
    after_acc, after_err = series(after)

    fig, ax = plt.subplots(figsize=(1.9 * len(names) + 3.2, 4.6))
    ax.bar(x - width / 2, before_acc, width, yerr=before_err, capsize=4,
           label=before.get("model", "before"), color="#94a3b8")
    ax.bar(x + width / 2, after_acc, width, yerr=after_err, capsize=4,
           label=after.get("model", "after"), color="#2563eb")

    for xi, (b, a) in enumerate(zip(before_acc, after_acc)):
        ax.text(xi - width / 2, b + 1.5, f"{b:.0f}", ha="center", fontsize=9)
        ax.text(xi + width / 2, a + 1.5, f"{a:.0f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"{title}\n(error bars = Wilson 95% CI, n=30 per slice)", fontsize=11)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

    if results_json:
        payload = {
            "chart": "before_after",
            "title": title,
            "slices": names,
            "before": {
                "model": before.get("model", "before"),
                "accuracy": {n: before["slices"][n]["accuracy"] for n in names},
                "ci": {n: [before["slices"][n]["ci_low"], before["slices"][n]["ci_high"]] for n in names},
                "th_ratio": before["overall"]["th_ratio"],
                "mean_ppl": before["overall"].get("mean_ppl"),
            },
            "after": {
                "model": after.get("model", "after"),
                "accuracy": {n: after["slices"][n]["accuracy"] for n in names},
                "ci": {n: [after["slices"][n]["ci_low"], after["slices"][n]["ci_high"]] for n in names},
                "th_ratio": after["overall"]["th_ratio"],
                "mean_ppl": after["overall"].get("mean_ppl"),
            },
        }
        path = pathlib.Path(results_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if show:
        plt.show()
    return fig, ax


def plot_slice_bars(report: dict, out_path: str | pathlib.Path | None = None, show: bool = True):
    """Single-model per-slice accuracy with Wilson intervals -- used in Cell 1."""
    import matplotlib.pyplot as plt
    import numpy as np

    use_thai_font()
    names = list(report["slices"])
    acc = np.array([100 * report["slices"][n]["accuracy"] for n in names])
    lo = np.array([100 * (report["slices"][n]["accuracy"] - report["slices"][n]["ci_low"]) for n in names])
    hi = np.array([100 * (report["slices"][n]["ci_high"] - report["slices"][n]["accuracy"]) for n in names])

    fig, ax = plt.subplots(figsize=(1.6 * len(names) + 2.5, 4.2))
    ax.bar(names, acc, yerr=[lo, hi], capsize=4, color="#2563eb")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"{report.get('model', 'model')} -- KobEval-TH baseline\n(Wilson 95% CI, n=30)", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax
