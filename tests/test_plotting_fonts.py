"""Regression tests for the Thai-font tofu bug.

Symptom on Colab: every Thai label rendered as [] boxes with only a
UserWarning ("Glyph 3637 ... missing from font DejaVu Sans") to show for it.
Cause: Cell 0 apt-installs fonts-thai-tlwg *after* matplotlib has built
fontManager.ttflist from a cache, so the new fonts are invisible.
"""
import matplotlib
matplotlib.use("Agg")

from kobeval.plotting import (
    use_thai_font,
    _rescan_system_fonts,
    _THAI_FONT_CANDIDATES,
    _THAI_FILE_HINTS,
)


def test_rescan_is_idempotent_and_safe():
    """Rescanning twice must not raise or duplicate registrations."""
    from matplotlib import font_manager
    _rescan_system_fonts()
    n1 = len(font_manager.fontManager.ttflist)
    _rescan_system_fonts()
    n2 = len(font_manager.fontManager.ttflist)
    assert n2 == n1, "second rescan should register nothing new"


def test_use_thai_font_keeps_dejavu_as_fallback():
    """Thai font first, DejaVu behind it so beta/maths glyphs still resolve."""
    name = use_thai_font()
    if name is None:
        import pytest
        pytest.skip("no Thai font installed on this machine")
    stack = matplotlib.rcParams["font.sans-serif"]
    assert stack[0] == name
    assert "DejaVu Sans" in stack, "Latin/maths fallback must remain"
    assert matplotlib.rcParams["font.family"] == ["sans-serif"]


def test_tlwg_fonts_are_discoverable_by_hint():
    """The apt package ships Loma/Garuda/etc; their filenames must match a hint."""
    for fname in ("Loma.ttf", "Garuda-Bold.ttf", "NotoSansThai-Regular.ttf",
                  "TH Sarabun New.ttf", "Waree.ttf"):
        assert any(h in fname.lower() for h in _THAI_FILE_HINTS), fname


def test_candidate_list_covers_the_apt_package():
    """fonts-thai-tlwg provides these; at least these must be candidates."""
    for f in ("Loma", "Garuda", "Norasi", "Waree"):
        assert f in _THAI_FONT_CANDIDATES
