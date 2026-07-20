"""Regression tests: base models must not be prompted with a chat template.

Observed on a real T4 run of notebook 01: TH-KNOW read 0.0% before AND after
CPT, with th_ratio 0.00 and perplexity ~489,000 (higher than the 151,936 vocab,
i.e. worse than uniform). Cause: Qwen3-0.6B-Base ships a 4 KB chat template it
was never trained on, and the runner keyed off "has a chat template" to decide
how to prompt. Every generation was fluent nonsense, and it looked like a
legitimate null result rather than a broken measurement.
"""
from kobeval.runner import _build_prompt, _truncate_fewshot, is_base_model


class _Fake:
    def __init__(self, name):
        self.name_or_path = name
        self.config = type("c", (), {"_name_or_path": name})


def test_base_model_detected_by_name_not_chat_template():
    assert is_base_model(_Fake("Qwen/Qwen3-0.6B-Base"), None) is True
    assert is_base_model(_Fake("Qwen/Qwen3-0.6B"), None) is False
    assert is_base_model(_Fake("/content/checkpoints/qwen3-0.6b-base/"), None) is True


def test_base_prompt_is_fewshot_and_has_no_chat_markup():
    p = _build_prompt(None, {"prompt": "เมืองหลวงของไทยคือ"}, False, None, base_model=True)
    assert "<|im_start|>" not in p, "base model must never see chat markup"
    assert p.count("คำถาม:") == 4, "3 exemplars + the real question"
    assert p.rstrip().endswith("คำตอบ:"), "must end mid-pattern so the model completes it"


def test_fewshot_exemplars_are_not_benchmark_items():
    """Exemplars must not leak KobEval-TH content."""
    from kobeval.runner import _FEWSHOT_TH
    from kobeval.data import load_slices
    bench = {i["prompt"].strip() for i in load_slices(["TH-KNOW"])["TH-KNOW"]}
    for q, _ in _FEWSHOT_TH:
        assert q.strip() not in bench, f"exemplar leaked from the benchmark: {q}"


def test_truncation_keeps_only_the_first_answer():
    assert _truncate_fewshot("กรุงเทพมหานคร\nคำถาม: อะไร\nคำตอบ: x") == "กรุงเทพมหานคร"
    assert _truncate_fewshot("อ่าวไทย\n\nอย่างอื่น") == "อ่าวไทย"
    assert _truncate_fewshot("  เชียงใหม่  ") == "เชียงใหม่"
