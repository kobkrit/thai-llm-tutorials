"""The KobEval-TH evaluation runner.

THE EVALUATION CONTRACT
=======================
Every notebook in this series calls ``evaluate()`` with the same settings, and
those settings are frozen here rather than passed in per-notebook, because a
benchmark whose decoding parameters drift between posts measures nothing:

    do_sample        = False   (greedy -- no temperature, no top_p, no top_k)
    max_new_tokens   = 256
    enable_thinking  = False   (Qwen3's thinking mode is off unless overridden)
    torch.manual_seed(42)      (re-seeded before EVERY generation, not once)
    dtype            = whatever the caller loaded; on a T4 that must be float16

Greedy decoding makes the run reproducible; re-seeding per item means an item's
result does not depend on how many items ran before it, so you can evaluate a
single slice and get identical numbers to a full run.
"""

from __future__ import annotations

import json
import math
import pathlib
import time
from typing import Callable, Dict, List, Sequence

from .data import SLICES, load_slices
from .metrics import (
    grade_instr,
    grade_know,
    grade_math,
    grade_safe,
    th_ratio,
)
from .stats import wilson_ci

__all__ = ["evaluate", "compare", "EVAL_CONTRACT"]

EVAL_CONTRACT = {
    "do_sample": False,
    "max_new_tokens": 256,
    "enable_thinking": False,
    "seed": 42,
    "version": "kobeval-th-v1",
}


def _build_prompt(tokenizer, item: dict, enable_thinking: bool, system: str | None) -> str:
    """Render one item through the model's chat template.

    Falls back to the raw prompt for base models (``Qwen3-0.6B-Base`` has no
    chat template) -- that fallback is deliberate and is the reason post 2 can
    compare a base model against an instruct model on the same benchmark.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": item["prompt"]})

    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            # Templates that predate Qwen3 do not accept enable_thinking.
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    return item["prompt"]


def _strip_thinking(text: str) -> str:
    """Remove a Qwen3 <think>...</think> block if one leaked into the output."""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def _generate(model, tokenizer, prompt: str, max_new_tokens: int, seed: int) -> str:
    import torch

    torch.manual_seed(seed)  # re-seed per item: see the contract note above
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[-1]:]
    return _strip_thinking(tokenizer.decode(generated, skip_special_tokens=True))


def _perplexity(model, tokenizer, prompt: str, continuation: str | None) -> float | None:
    """Teacher-forced perplexity.

    If a gold ``continuation`` exists (TH-KNOW, TH-MATH) the loss is masked to
    the continuation tokens only, so the number answers "how surprised is the
    model by the correct Thai answer". For TH-INSTR and TH-SAFE there is no gold
    text, so we fall back to perplexity over the prompt itself, which is a proxy
    for how well the model models Thai at all. The two are NOT comparable across
    slices, only across models on the same slice -- which is the only comparison
    the tutorials actually make.
    """
    import torch

    try:
        if continuation:
            prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
            full_ids = tokenizer(prompt + continuation, return_tensors="pt").input_ids
            n_prompt = prompt_ids.shape[-1]
            if full_ids.shape[-1] <= n_prompt:
                return None
            labels = full_ids.clone()
            labels[:, :n_prompt] = -100  # score only the gold continuation
        else:
            full_ids = tokenizer(prompt, return_tensors="pt").input_ids
            if full_ids.shape[-1] < 2:
                return None
            labels = full_ids.clone()

        full_ids = full_ids.to(model.device)
        labels = labels.to(model.device)
        with torch.no_grad():
            loss = model(input_ids=full_ids, labels=labels).loss
        value = float(loss.item())
        if not math.isfinite(value):
            return None
        return float(math.exp(min(value, 20.0)))  # clamp: exp(>20) is not a number anyone needs
    except Exception:
        return None


def _gold_text(item: dict) -> str | None:
    if "answers" in item:
        return str(item["answers"][0])
    if "answer" in item:
        return str(item["answer"])
    return None


def evaluate(
    model,
    tokenizer,
    slices: Sequence[str] | None = None,
    seed: int = 42,
    max_new_tokens: int = 256,
    enable_thinking: bool = False,
    system: str | None = None,
    compute_ppl: bool = True,
    out_path: str | pathlib.Path | None = "results.json",
    model_name: str = "model",
    extra: dict | None = None,
    progress: bool = True,
    generate_fn: Callable[..., str] | None = None,
) -> dict:
    """Run KobEval-TH and return a report dict (also written to ``out_path``).

    Args:
        model, tokenizer: a loaded HF causal LM and its tokenizer.
        slices: subset of ["TH-KNOW", "TH-MATH", "TH-INSTR", "TH-SAFE"].
        seed: re-applied before every generation.
        max_new_tokens: frozen at 256 by the contract; override only to explain
            what changes when you do.
        enable_thinking: Qwen3 thinking mode. Off by default.
        system: optional system prompt (post 4 uses this to show how much of
            "Thai ability" is really just prompting).
        out_path: where to write results.json; None disables the write.
        model_name: label used in tables and plots.
        extra: merged into the report's "meta" (e.g. train_time_s, vram_peak_gb).
        generate_fn: injectable generator, used by the test suite to exercise
            the whole pipeline without a GPU.

    Returns:
        {
          "model": str,
          "slices": {slice: {accuracy, ci_low, ci_high, n, n_correct,
                             mean_ppl, mean_output_len, th_ratio, ...}},
          "overall": {accuracy, th_ratio, mean_ppl, ...},
          "meta": {...contract, timings, vram...},
          "items": [per-item records]
        }
    """
    slice_names = list(slices) if slices else list(SLICES)
    unknown = [s for s in slice_names if s not in SLICES]
    if unknown:
        raise KeyError(f"unknown slice(s) {unknown}; expected from {SLICES}")

    gen = generate_fn or (lambda prompt: _generate(model, tokenizer, prompt, max_new_tokens, seed))

    # Reset the CUDA peak-memory counter so vram_peak_gb measures THIS eval.
    torch = None
    try:
        import torch as _torch

        torch = _torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass

    data = load_slices(slice_names)
    started = time.time()
    slice_reports: Dict[str, dict] = {}
    all_records: List[dict] = []

    for slice_name in slice_names:
        items = data[slice_name]
        records = []
        for idx, item in enumerate(items, 1):
            prompt = _build_prompt(tokenizer, item, enable_thinking, system)
            output = gen(prompt)

            if slice_name == "TH-KNOW":
                correct, detail = grade_know(item, output), {}
            elif slice_name == "TH-MATH":
                correct, detail = grade_math(item, output), {}
            elif slice_name == "TH-INSTR":
                graded = grade_instr(item, output)
                correct, detail = graded["passed"], graded
            else:
                graded = grade_safe(item, output)
                correct, detail = graded["correct"], graded

            ppl = (
                _perplexity(model, tokenizer, prompt, _gold_text(item))
                if (compute_ppl and generate_fn is None)
                else None
            )

            record = {
                "id": item["id"],
                "slice": slice_name,
                "prompt": item["prompt"],
                "output": output,
                "correct": bool(correct),
                "th_ratio": th_ratio(output),
                "output_len": len(output),
                "ppl": ppl,
                "detail": detail,
            }
            records.append(record)
            all_records.append(record)

            if progress and idx % 10 == 0:
                print(f"  {slice_name}: {idx}/{len(items)}")

        n = len(records)
        n_correct = sum(r["correct"] for r in records)
        ppls = [r["ppl"] for r in records if r["ppl"] is not None]
        lo, hi = wilson_ci(n_correct, n)

        report = {
            "n": n,
            "n_correct": n_correct,
            "accuracy": n_correct / n if n else 0.0,
            "ci_low": lo,
            "ci_high": hi,
            "ci_width": hi - lo,
            "mean_ppl": (sum(ppls) / len(ppls)) if ppls else None,
            "mean_output_len": sum(r["output_len"] for r in records) / n if n else 0.0,
            "th_ratio": sum(r["th_ratio"] for r in records) / n if n else 0.0,
            "empty_outputs": sum(1 for r in records if not r["output"].strip()),
        }

        if slice_name == "TH-SAFE":
            # The two halves fail in opposite directions; averaging them hides both.
            report["unsafe_compliance_rate"] = sum(
                r["detail"].get("unsafe_compliance", False) for r in records
            ) / max(1, sum(1 for r in records if r["detail"].get("expected") == "refuse"))
            report["over_refusal_rate"] = sum(
                r["detail"].get("over_refusal", False) for r in records
            ) / max(1, sum(1 for r in records if r["detail"].get("expected") == "comply"))

        if slice_name == "TH-INSTR":
            report["mean_rubric_score"] = sum(
                r["detail"].get("score", 0.0) for r in records
            ) / n if n else 0.0

        slice_reports[slice_name] = report

    elapsed = time.time() - started

    vram_peak_gb = None
    if torch is not None and torch.cuda.is_available():
        vram_peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    total_n = sum(s["n"] for s in slice_reports.values())
    total_correct = sum(s["n_correct"] for s in slice_reports.values())
    all_ppls = [r["ppl"] for r in all_records if r["ppl"] is not None]

    result = {
        "model": model_name,
        "slices": slice_reports,
        "overall": {
            "n": total_n,
            "n_correct": total_correct,
            "accuracy": total_correct / total_n if total_n else 0.0,
            "th_ratio": (sum(r["th_ratio"] for r in all_records) / len(all_records))
            if all_records
            else 0.0,
            "mean_ppl": (sum(all_ppls) / len(all_ppls)) if all_ppls else None,
            "mean_output_len": (sum(r["output_len"] for r in all_records) / len(all_records))
            if all_records
            else 0.0,
        },
        "meta": {
            **EVAL_CONTRACT,
            "seed": seed,
            "max_new_tokens": max_new_tokens,
            "enable_thinking": enable_thinking,
            "system_prompt": system,
            "slices_run": slice_names,
            "eval_time_s": elapsed,
            "vram_peak_gb": vram_peak_gb,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **(extra or {}),
        },
        "items": all_records,
    }

    if out_path:
        write_results(result, out_path)
    return result


def write_results(result: dict, out_path: str | pathlib.Path) -> pathlib.Path:
    """Write results.json (UTF-8, not ASCII-escaped, so the Thai stays readable)."""
    path = pathlib.Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Comparison table -- section 8 of every post
# ---------------------------------------------------------------------------

_TABLE_SLICES = ["TH-KNOW", "TH-MATH", "TH-INSTR"]


def _fmt_pct(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.1f}%"


def compare(
    *reports: dict,
    out_path: str | pathlib.Path | None = "results.json",
    markdown: bool = True,
) -> dict:
    """Build the standard cross-model results table.

    Columns: Model | TH-KNOW | TH-MATH | TH-INSTR | th_ratio | PPL | VRAM peak | Train time

    Accepts any number of reports from ``evaluate()``. Returns a dict with the
    rows, the rendered markdown, and -- when exactly two reports are given -- a
    McNemar test per slice comparing the second against the first, so the post
    can say whether a change is real rather than eyeballing two percentages.
    """
    if not reports:
        raise ValueError("compare() needs at least one report")

    rows = []
    for rep in reports:
        row = {"model": rep.get("model", "model")}
        for slice_name in _TABLE_SLICES:
            s = rep["slices"].get(slice_name)
            row[slice_name] = None if s is None else s["accuracy"]
            row[f"{slice_name}_ci"] = None if s is None else (s["ci_low"], s["ci_high"])
        safe = rep["slices"].get("TH-SAFE")
        row["TH-SAFE"] = None if safe is None else safe["accuracy"]
        row["th_ratio"] = rep["overall"]["th_ratio"]
        row["ppl"] = rep["overall"].get("mean_ppl")
        row["vram_peak_gb"] = rep["meta"].get("vram_peak_gb")
        row["train_time_s"] = rep["meta"].get("train_time_s")
        rows.append(row)

    header = (
        "| Model | TH-KNOW | TH-MATH | TH-INSTR | th_ratio | PPL | VRAM peak | Train time |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    lines = [header]
    for row in rows:
        ppl = "-" if row["ppl"] is None else f"{row['ppl']:.1f}"
        vram = "-" if row["vram_peak_gb"] is None else f"{row['vram_peak_gb']:.2f} GB"
        train = "-" if row["train_time_s"] is None else f"{row['train_time_s'] / 60:.1f} min"
        lines.append(
            f"| {row['model']} | {_fmt_pct(row['TH-KNOW'])} | {_fmt_pct(row['TH-MATH'])} "
            f"| {_fmt_pct(row['TH-INSTR'])} | {row['th_ratio']:.2f} | {ppl} | {vram} | {train} |"
        )
    table_md = "\n".join(lines)

    result = {"rows": rows, "markdown": table_md, "contract": EVAL_CONTRACT}

    if len(reports) == 2:
        from .stats import mcnemar

        before, after = reports
        tests = {}
        for slice_name in SLICES:
            if slice_name not in before["slices"] or slice_name not in after["slices"]:
                continue
            before_by_id = {r["id"]: r["correct"] for r in before["items"] if r["slice"] == slice_name}
            after_by_id = {r["id"]: r["correct"] for r in after["items"] if r["slice"] == slice_name}
            shared = set(before_by_id) & set(after_by_id)
            b = sum(1 for i in shared if before_by_id[i] and not after_by_id[i])
            c = sum(1 for i in shared if not before_by_id[i] and after_by_id[i])
            tests[slice_name] = mcnemar(b, c)
        result["mcnemar"] = tests

    if markdown:
        print(table_md)
        if "mcnemar" in result:
            print("\nMcNemar (baseline -> new):")
            for slice_name, test in result["mcnemar"].items():
                warn = "  [n<25: treat p as indicative]" if test["exact_recommended"] else ""
                print(
                    f"  {slice_name}: fixed={test['c']} broke={test['b']} "
                    f"p={test['p_value']:.4f} {test['direction']}{warn}"
                )

    if out_path:
        path = pathlib.Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return result
