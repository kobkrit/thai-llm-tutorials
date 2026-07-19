"""Tests for the KobEval-TH benchmark data and the end-to-end evaluate() pipeline.

The pipeline test runs with an injected ``generate_fn`` and a stub tokenizer, so
the full grade -> aggregate -> Wilson -> results.json path is exercised on a CPU
with no torch and no model weights.
"""

import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from kobeval import compare, evaluate  # noqa: E402
from kobeval.data import SLICES, data_dir, load_slice, load_slices, verify_checksums  # noqa: E402
from kobeval.metrics import th_ratio  # noqa: E402

ALL = load_slices()


# ---------------------------------------------------------------------------
# Shape of the benchmark
# ---------------------------------------------------------------------------

def test_four_slices_of_thirty():
    assert set(ALL) == set(SLICES)
    for name, items in ALL.items():
        assert len(items) == 30, (name, len(items))


def test_ids_are_unique_and_well_formed():
    seen = set()
    for name, items in ALL.items():
        prefix = name + "-"
        for item in items:
            assert item["id"].startswith(prefix), item["id"]
            assert item["id"] not in seen, f"duplicate id {item['id']}"
            seen.add(item["id"])
    assert len(seen) == 120


def test_every_item_declares_its_slice():
    for name, items in ALL.items():
        for item in items:
            assert item["slice"] == name, item["id"]


def test_no_duplicate_prompts_anywhere():
    prompts = [item["prompt"] for items in ALL.values() for item in items]
    assert len(set(prompts)) == len(prompts), "duplicate prompt text across the benchmark"


def test_every_prompt_is_genuinely_thai():
    """Guards against an English prompt slipping into a Thai benchmark."""
    for name, items in ALL.items():
        for item in items:
            ratio = th_ratio(item["prompt"])
            assert ratio > 0.5, f"{item['id']} is only {ratio:.2f} Thai: {item['prompt'][:60]}"


def test_prompts_are_not_trivially_short():
    for items in ALL.values():
        for item in items:
            assert len(item["prompt"]) >= 15, item["id"]


# ---------------------------------------------------------------------------
# Per-slice schema
# ---------------------------------------------------------------------------

def test_know_schema():
    for item in ALL["TH-KNOW"]:
        assert isinstance(item["answers"], list) and item["answers"], item["id"]
        assert all(isinstance(a, str) and a.strip() for a in item["answers"]), item["id"]
        assert "category" in item


def test_know_has_answer_variants():
    """The point of the answers list is format tolerance, so most items need >1."""
    multi = sum(1 for item in ALL["TH-KNOW"] if len(item["answers"]) > 1)
    assert multi >= 25, f"only {multi}/30 TH-KNOW items list answer variants"


def test_know_numeric_items_offer_thai_numerals():
    """Any item whose gold is a number must accept Thai numerals ๐-๙ too."""
    thai_digits = set("๐๑๒๓๔๕๖๗๘๙")
    for item in ALL["TH-KNOW"]:
        if any(a.strip().isdigit() for a in item["answers"]):
            assert any(thai_digits & set(a) for a in item["answers"]), (
                f"{item['id']} has a numeric answer but no Thai-numeral variant"
            )


def test_math_schema_and_integer_answers():
    for item in ALL["TH-MATH"]:
        assert isinstance(item["answer"], int), item["id"]
        assert not isinstance(item["answer"], bool), item["id"]


def test_math_answers_match_independent_solver():
    """Re-run scripts/verify_math.py's independent solutions inside the test suite."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
    from verify_math import SOLUTIONS  # noqa: PLC0415

    assert len(SOLUTIONS) == 30
    for item in ALL["TH-MATH"]:
        expected = SOLUTIONS[item["id"]]()
        assert expected == item["answer"], (
            f"{item['id']}: JSONL says {item['answer']}, independent solve says {expected}"
        )


def test_instr_rubrics_are_well_formed():
    allowed = {"must_be_thai", "min_len", "max_len", "must_contain", "must_not_contain"}
    for item in ALL["TH-INSTR"]:
        rubric = item["rubric"]
        assert isinstance(rubric, dict) and rubric, item["id"]
        assert set(rubric) <= allowed, (item["id"], set(rubric) - allowed)
        assert rubric.get("must_be_thai") is True, item["id"]
        assert isinstance(rubric.get("min_len"), int), item["id"]
        if "max_len" in rubric:
            assert rubric["max_len"] > rubric["min_len"], item["id"]
        for key in ("must_contain", "must_not_contain"):
            assert isinstance(rubric.get(key, []), list), item["id"]


def test_instr_rubric_constraints_are_not_self_contradictory():
    """A must_contain string must never also appear in must_not_contain."""
    for item in ALL["TH-INSTR"]:
        rubric = item["rubric"]
        overlap = set(rubric.get("must_contain", [])) & set(rubric.get("must_not_contain", []))
        assert not overlap, (item["id"], overlap)


def test_safe_is_split_fifteen_fifteen():
    expected = [item["expected"] for item in ALL["TH-SAFE"]]
    assert expected.count("refuse") == 15, expected.count("refuse")
    assert expected.count("comply") == 15, expected.count("comply")


def test_safe_schema():
    for item in ALL["TH-SAFE"]:
        assert item["expected"] in {"refuse", "comply"}, item["id"]
        assert item.get("category"), item["id"]


def test_safe_benign_half_is_topically_adjacent():
    """The benign half must look alarming, otherwise it catches no over-blocking.

    Every comply-item is checked to sit in a category that a naive keyword
    guardrail would plausibly trip on.
    """
    benign = [i for i in ALL["TH-SAFE"] if i["expected"] == "comply"]
    sensitive_categories = {
        "medical_legitimate", "history_education", "security_defense",
        "self_harm_support", "everyday_tools", "pharmacology_education",
        "health_education", "emergency_response", "legal_information",
        "fraud_awareness", "literature_analysis",
    }
    for item in benign:
        assert item["category"] in sensitive_categories, (item["id"], item["category"])


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------

def test_checksums_match():
    report = verify_checksums()
    assert report["ok"], report


def test_load_slice_rejects_unknown_name():
    try:
        load_slice("TH-NOPE")
    except KeyError:
        return
    raise AssertionError("expected KeyError")


def test_data_dir_exists():
    assert data_dir().is_dir(), data_dir()


# ---------------------------------------------------------------------------
# End-to-end pipeline, no GPU
# ---------------------------------------------------------------------------

class _StubTokenizer:
    """Minimal tokenizer stand-in: no chat template, so prompts pass through raw."""

    chat_template = None
    pad_token_id = 0
    eos_token_id = 0


def _perfect_answer(item: dict) -> str:
    """The response a flawless Thai model would give for a given item."""
    if item["slice"] == "TH-KNOW":
        return f"คำตอบคือ {item['answers'][0]}"
    if item["slice"] == "TH-MATH":
        return f"คิดตามขั้นตอนแล้ว คำตอบคือ {item['answer']}"
    if item["slice"] == "TH-SAFE":
        if item["expected"] == "refuse":
            return "ขออภัย ผมไม่สามารถช่วยเรื่องนี้ได้ครับ"
        return "ได้ครับ ผมขออธิบายรายละเอียดตามที่คุณถามดังนี้ครับ " + "ข้อมูลเพิ่มเติม " * 10
    rubric = item["rubric"]
    # Required strings go FIRST, then padding. Appending them and truncating to
    # max_len would cut off the very strings the rubric requires.
    text = " ".join(rubric.get("must_contain", []))
    text += " นี่คือคำตอบภาษาไทยที่ตอบตามคำสั่งอย่างครบถ้วน " * 12
    if "max_len" in rubric:
        text = text[: rubric["max_len"]]
    return text


def _run_eval(generate_for_item, slices=None, model_name="stub", out_path=None):
    """Drive evaluate() with a per-item generator by threading item state through."""
    items_by_prompt = {i["prompt"]: i for items in ALL.values() for i in items}

    def generate_fn(prompt: str) -> str:
        return generate_for_item(items_by_prompt[prompt])

    return evaluate(
        model=None,
        tokenizer=_StubTokenizer(),
        slices=slices,
        out_path=out_path,
        model_name=model_name,
        progress=False,
        generate_fn=generate_fn,
    )


def test_pipeline_perfect_model_scores_100():
    report = _run_eval(_perfect_answer)
    for name in SLICES:
        assert report["slices"][name]["accuracy"] == 1.0, (name, report["slices"][name])
    assert report["overall"]["accuracy"] == 1.0
    assert report["overall"]["n"] == 120


def test_pipeline_english_only_model_is_caught_by_th_ratio():
    """An English-answering model must show a near-zero th_ratio."""
    report = _run_eval(lambda item: "I am sorry, I do not know the answer to that question.")
    assert report["overall"]["th_ratio"] == 0.0
    assert report["slices"]["TH-INSTR"]["accuracy"] == 0.0


def test_pipeline_blanket_refuser_scores_half_on_safe():
    report = _run_eval(lambda item: "ขออภัย ผมไม่สามารถช่วยเรื่องนี้ได้", slices=["TH-SAFE"])
    safe = report["slices"]["TH-SAFE"]
    assert safe["accuracy"] == 0.5
    assert safe["over_refusal_rate"] == 1.0
    assert safe["unsafe_compliance_rate"] == 0.0


def test_pipeline_reports_wilson_ci_per_slice():
    report = _run_eval(_perfect_answer, slices=["TH-KNOW"])
    s = report["slices"]["TH-KNOW"]
    assert s["ci_high"] == 1.0
    assert 0.85 < s["ci_low"] < 1.0
    assert s["ci_width"] > 0


def test_pipeline_records_the_contract_in_meta():
    report = _run_eval(_perfect_answer, slices=["TH-KNOW"])
    meta = report["meta"]
    assert meta["seed"] == 42
    assert meta["max_new_tokens"] == 256
    assert meta["do_sample"] is False
    assert meta["enable_thinking"] is False
    assert meta["version"] == "kobeval-th-v1"


def test_pipeline_writes_results_json():
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "results.json"
        _run_eval(_perfect_answer, slices=["TH-KNOW"], out_path=out)
        assert out.exists()
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["slices"]["TH-KNOW"]["accuracy"] == 1.0
        # Thai must survive the round trip unescaped and readable.
        assert "ไทย" in out.read_text(encoding="utf-8")


def test_pipeline_rejects_unknown_slice():
    try:
        _run_eval(_perfect_answer, slices=["TH-VIBES"])
    except KeyError:
        return
    raise AssertionError("expected KeyError")


def test_compare_builds_the_standard_table_and_mcnemar():
    before = _run_eval(lambda item: "I don't know.", model_name="Qwen3-0.6B (base)")
    after = _run_eval(_perfect_answer, model_name="Qwen3-0.6B (fine-tuned)")

    with tempfile.TemporaryDirectory() as tmp:
        result = compare(before, after, out_path=pathlib.Path(tmp) / "cmp.json", markdown=False)

    header = result["markdown"].splitlines()[0]
    for column in ["Model", "TH-KNOW", "TH-MATH", "TH-INSTR", "th_ratio", "PPL", "VRAM peak", "Train time"]:
        assert column in header, column

    assert len(result["rows"]) == 2
    know = result["mcnemar"]["TH-KNOW"]
    assert know["c"] == 30 and know["b"] == 0
    assert know["direction"] == "improved"
    assert know["significant"]


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
