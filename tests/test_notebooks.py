"""Structural tests for the notebooks.

These encode the rules of the series' format: the setup cell is shared verbatim,
the baseline evaluation happens before anything else, and the T4 constraints are
never violated by a stray torch_dtype="auto".
"""

import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

ROOT = pathlib.Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"
CELL0 = (NOTEBOOKS / "cell0_setup.ipy").read_text(encoding="utf-8").rstrip("\n")
PATHS = sorted(NOTEBOOKS.glob("*.ipynb"))


def _source(cell) -> str:
    src = cell.get("source", "")
    return src if isinstance(src, str) else "".join(src)


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_notebooks_exist():
    assert PATHS, "no notebooks found"


def test_notebooks_are_valid_json_and_nbformat4():
    for path in PATHS:
        nb = _load(path)
        assert nb["nbformat"] == 4, path.name
        assert isinstance(nb["cells"], list) and nb["cells"], path.name


def test_every_cell_is_well_formed():
    for path in PATHS:
        for i, cell in enumerate(_load(path)["cells"]):
            assert cell["cell_type"] in {"markdown", "code"}, (path.name, i)
            assert "source" in cell, (path.name, i)
            if cell["cell_type"] == "code":
                assert cell.get("outputs") == [], f"{path.name} cell {i} ships saved outputs"
                assert cell.get("execution_count") is None, (path.name, i)


def test_first_cell_is_markdown_header_with_colab_badge():
    for path in PATHS:
        cells = _load(path)["cells"]
        assert cells[0]["cell_type"] == "markdown", path.name
        src = _source(cells[0])
        assert "colab-badge.svg" in src, path.name
        assert f"blob/main/notebooks/{path.stem}.ipynb" in src, (
            f"{path.name}: Colab badge does not point at itself"
        )


def test_colab_badge_uses_the_github_url_form():
    """The badge must resolve via colab.research.google.com/github/... ."""
    for path in PATHS:
        src = _source(_load(path)["cells"][0])
        assert "colab.research.google.com/github/kobkrit/thai-llm-tutorials" in src, path.name


def test_setup_cell_is_byte_identical_everywhere():
    for path in PATHS:
        code_cells = [c for c in _load(path)["cells"] if c["cell_type"] == "code"]
        assert _source(code_cells[0]).rstrip("\n") == CELL0, (
            f"{path.name}: first code cell differs from cell0_setup.ipy"
        )


def test_setup_cell_teaches_the_bf16_lesson():
    assert "SUPPORTS_BF16" in CELL0
    assert "torch.cuda.is_bf16_supported()" in CELL0
    assert "sdpa" in CELL0
    assert 'torch_dtype="auto"' in CELL0, "the auto-dtype landmine must be called out"


def test_setup_cell_asserts_a_gpu_is_present():
    assert "assert torch.cuda.is_available()" in CELL0


def test_setup_cell_sets_the_seed():
    for line in ["random.seed(SEED)", "np.random.seed(SEED)", "torch.manual_seed(SEED)"]:
        assert line in CELL0, line
    assert "SEED = 42" in CELL0


def test_setup_cell_imports_kobeval():
    assert "from kobeval import evaluate, compare, plot_before_after, th_ratio, wilson_ci" in CELL0


def _pip_install_specs(source: str) -> list[str]:
    """Extract the quoted package specs from the `!pip install` block.

    Scoped to the pip command and its backslash continuations, so that ordinary
    quoted Thai strings elsewhere in the cell are never mistaken for packages.
    """
    lines = source.splitlines()
    specs: list[str] = []
    for i, line in enumerate(lines):
        if "pip install" not in line:
            continue
        block = [line]
        j = i
        while block[-1].rstrip().endswith("\\") and j + 1 < len(lines):
            j += 1
            block.append(lines[j])
        specs.extend(re.findall(r'"([^"]+)"', "\n".join(block)))
    return specs


def test_setup_cell_pins_every_install():
    """Unpinned installs make the series unreproducible six months later."""
    specs = _pip_install_specs(CELL0)
    assert specs, "no pip install block found in the setup cell"
    for spec in specs:
        assert "==" in spec, f"unpinned dependency: {spec!r}"


def test_setup_cell_does_not_reinstall_torch():
    """Reinstalling torch over Colab's build is the top cause of broken notebooks."""
    for spec in _pip_install_specs(CELL0):
        assert not spec.startswith("torch=="), spec


def test_second_code_cell_is_the_baseline_evaluation():
    """Cell 1 always measures before anything is changed."""
    for path in PATHS:
        code_cells = [c for c in _load(path)["cells"] if c["cell_type"] == "code"]
        assert len(code_cells) >= 2, path.name
        src = _source(code_cells[1])
        assert "BASELINE" in src, path.name
        assert "evaluate(" in src, path.name
        assert "seed=42" in src, path.name


def _strip_comments(source: str) -> str:
    """Drop `#` comments so the T4 guards inspect executable code, not prose.

    The setup cell deliberately *mentions* torch_dtype="auto" and
    flash_attention_2 in its teaching comments; only real code must be checked.
    Naive splitting on '#' is sufficient here because no code line in these
    notebooks contains a '#' inside a string literal.
    """
    return "\n".join(line.split("#", 1)[0] for line in source.splitlines())


def test_code_never_uses_bf16_or_flash_attention_on_t4():
    """The T4 landmines, guarded mechanically against every code cell."""
    for path in PATHS:
        for i, cell in enumerate(_load(path)["cells"]):
            if cell["cell_type"] != "code":
                continue
            code = _strip_comments(_source(cell))
            assert 'torch_dtype="auto"' not in code, f"{path.name} cell {i}: auto dtype on T4"
            assert "flash_attention_2" not in code, f"{path.name} cell {i}: FA2 is Ampere+ only"
            assert "bf16=True" not in code, f"{path.name} cell {i}: bf16=True is invalid on T4"


def test_model_loading_is_explicit_about_dtype_and_attention():
    """Wherever a model is loaded, dtype and attention must be stated explicitly."""
    for path in PATHS:
        for cell in _load(path)["cells"]:
            if cell["cell_type"] != "code":
                continue
            code = _strip_comments(_source(cell))
            if "AutoModelForCausalLM.from_pretrained(" not in code:
                continue
            assert "torch_dtype=DTYPE" in code, path.name
            assert "attn_implementation=ATTN_IMPL" in code, path.name


def test_notebook_has_thai_limitations_section():
    for path in PATHS:
        text = "".join(_source(c) for c in _load(path)["cells"] if c["cell_type"] == "markdown")
        assert "ข้อจำกัดของการทดลองนี้" in text, path.name


def test_notebook_has_ten_section_outline():
    for path in PATHS:
        header = _source(_load(path)["cells"][0])
        for n in range(1, 11):
            assert f"{n}. " in header, (path.name, n)


def test_notebook_mentions_the_evaluation_spine():
    for path in PATHS:
        text = "".join(_source(c) for c in _load(path)["cells"] if c["cell_type"] == "markdown")
        assert "KobEval-TH" in text, path.name
        assert "th_ratio" in text, path.name


def _run_all():
    mod = sys.modules[__name__]
    tests = [n for n in dir(mod) if n.startswith("test_")]
    failed = 0
    for name in sorted(tests):
        try:
            getattr(mod, name)()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
