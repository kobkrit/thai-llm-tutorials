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


def test_font_family_is_a_list_for_per_glyph_fallback():
    """font.family must be a LIST, not "sans-serif" + a font.sans-serif list.

    Only the list form makes matplotlib fall back per glyph. The other form
    picks one family and renders missing glyphs as .notdef with no warning,
    which is how beta in "DPO (beta=0.1)" became a tofu box while Thai looked
    fine and nothing was logged.
    """
    name = use_thai_font()
    if name is None:
        import pytest
        pytest.skip("no Thai font installed on this machine")
    fam = matplotlib.rcParams["font.family"]
    assert isinstance(fam, list) and len(fam) >= 2, f"family must be a list: {fam}"
    assert fam[0] == name
    assert "DejaVu Sans" in fam, "Latin/Greek/maths fallback must remain"


def test_greek_beta_renders_with_thai_font(tmp_path):
    """Regression: beta must not be tofu when a Thai font is active.

    Thai fonts routinely lack Greek, and matplotlib does not warn when a glyph
    silently falls through to .notdef, so this asserts on rendered pixels.
    """
    import pytest
    import matplotlib.pyplot as plt
    if use_thai_font() is None:
        pytest.skip("no Thai font installed on this machine")

    def ink(text):
        fig, ax = plt.subplots(figsize=(2, 1))
        ax.set_title(text)
        ax.axis("off")
        out = tmp_path / "probe.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        from matplotlib import image as mpimg
        return float((mpimg.imread(out)[..., :3] < 0.5).sum())

    # A real beta and a tofu box both mark pixels, so compare against a
    # codepoint no font has: if beta rendered as .notdef the two would match.
    assert ink("\u03b2") != ink("\U000107a0"), "beta appears to render as .notdef"


def test_tlwg_fonts_are_discoverable_by_hint():
    """The apt package ships Loma/Garuda/etc; their filenames must match a hint."""
    for fname in ("Loma.ttf", "Garuda-Bold.ttf", "NotoSansThai-Regular.ttf",
                  "TH Sarabun New.ttf", "Waree.ttf"):
        assert any(h in fname.lower() for h in _THAI_FILE_HINTS), fname


def test_candidate_list_covers_the_apt_package():
    """fonts-thai-tlwg provides these; at least these must be candidates."""
    for f in ("Loma", "Garuda", "Norasi", "Waree"):
        assert f in _THAI_FONT_CANDIDATES
