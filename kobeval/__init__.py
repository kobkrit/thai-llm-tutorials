"""kobeval -- the shared evaluation harness for the Thai LLM tutorial series.

    https://kobkrit.com  --  Dr. Kobkrit Viriyayudhakorn

Public API, used identically in all ten notebooks:

    from kobeval import evaluate, compare, plot_before_after, th_ratio, wilson_ci

    report = evaluate(model, tokenizer, slices=["TH-KNOW", "TH-MATH"], seed=42)

Design constraint worth stating up front: everything except ``evaluate`` and the
plotting helpers is pure Python with no torch dependency, so the statistics and
metrics can be imported and tested on a CPU-only machine -- and are, in
tests/test_metrics.py and tests/test_stats.py.
"""

from .data import SLICES, load_slice, load_slices, verify_checksums
from .metrics import (
    exact_match,
    extract_final_int,
    grade_instr,
    grade_know,
    grade_math,
    grade_safe,
    is_refusal,
    normalize_th,
    th_ratio,
    thai_digits_to_arabic,
)
from .plotting import plot_before_after, plot_slice_bars, use_thai_font
from .runner import EVAL_CONTRACT, compare, evaluate, write_results
from .stats import mcnemar, pass_at_k, wilson_ci

__version__ = "0.1.0"
BENCHMARK_VERSION = "kobeval-th-v1"

__all__ = [
    # headline API
    "evaluate",
    "compare",
    "plot_before_after",
    "plot_slice_bars",
    "th_ratio",
    "wilson_ci",
    # statistics
    "mcnemar",
    "pass_at_k",
    # metrics and graders
    "exact_match",
    "extract_final_int",
    "is_refusal",
    "normalize_th",
    "thai_digits_to_arabic",
    "grade_know",
    "grade_math",
    "grade_instr",
    "grade_safe",
    # data
    "SLICES",
    "load_slice",
    "load_slices",
    "verify_checksums",
    # misc
    "EVAL_CONTRACT",
    "write_results",
    "use_thai_font",
    "__version__",
    "BENCHMARK_VERSION",
]
