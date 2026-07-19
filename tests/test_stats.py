"""Tests for kobeval.stats -- pure Python, no GPU, no torch."""

import math
import sys
import pathlib
from itertools import combinations

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from kobeval.stats import mcnemar, pass_at_k, wilson_ci  # noqa: E402


# ---------------------------------------------------------------------------
# wilson_ci
# ---------------------------------------------------------------------------

def test_wilson_known_value():
    """The textbook check: 50/100 at 95% is approximately (0.404, 0.596)."""
    lo, hi = wilson_ci(50, 100)
    assert abs(lo - 0.4038) < 1e-3, lo
    assert abs(hi - 0.5962) < 1e-3, hi


def test_wilson_is_not_the_normal_approximation():
    """At p=0 the normal approximation gives a zero-width interval; Wilson does not.

    This is exactly the case that matters for these tutorials, because a
    baseline Qwen3-0.6B genuinely scores 0/30 on some slices.
    """
    lo, hi = wilson_ci(0, 30)
    assert lo == 0.0
    assert hi > 0.10, "a 0/30 result must still carry real uncertainty"
    assert abs(hi - 0.1135) < 1e-3, hi


def test_wilson_stays_in_unit_interval():
    for n in (1, 5, 30, 100, 1000):
        for k in range(n + 1):
            lo, hi = wilson_ci(k, n)
            assert 0.0 <= lo <= hi <= 1.0, (k, n, lo, hi)


def test_wilson_perfect_score():
    lo, hi = wilson_ci(30, 30)
    assert hi == 1.0
    assert abs(lo - 0.8865) < 1e-3, lo


def test_wilson_symmetry():
    """wilson(k, n) must mirror wilson(n-k, n) about 0.5."""
    for n in (7, 30, 99):
        for k in range(n + 1):
            lo_a, hi_a = wilson_ci(k, n)
            lo_b, hi_b = wilson_ci(n - k, n)
            assert abs(lo_a - (1 - hi_b)) < 1e-12
            assert abs(hi_a - (1 - lo_b)) < 1e-12


def test_wilson_interval_narrows_with_n():
    widths = [wilson_ci(n // 2, n)[1] - wilson_ci(n // 2, n)[0] for n in (10, 50, 200, 1000)]
    assert widths == sorted(widths, reverse=True), widths


def test_wilson_zero_trials_returns_full_interval():
    assert wilson_ci(0, 0) == (0.0, 1.0)


def test_wilson_rejects_bad_input():
    for bad in [(5, 3), (-1, 10)]:
        try:
            wilson_ci(*bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


# ---------------------------------------------------------------------------
# pass_at_k
# ---------------------------------------------------------------------------

def brute_force_pass_at_k(n: int, c: int, k: int) -> float:
    """Enumerate every k-subset of n samples and count those containing a correct one.

    Deliberately naive and obviously right, so it can referee the closed form.
    Samples 0..c-1 are the correct ones.
    """
    correct = set(range(c))
    subsets = list(combinations(range(n), k))
    hits = sum(1 for s in subsets if correct & set(s))
    return hits / len(subsets)


def test_pass_at_k_matches_brute_force():
    checked = 0
    for n in range(1, 13):
        for c in range(0, n + 1):
            for k in range(1, n + 1):
                assert abs(pass_at_k(n, c, k) - brute_force_pass_at_k(n, c, k)) < 1e-9, (n, c, k)
                checked += 1
    assert checked > 300, checked


def test_pass_at_k_edges():
    assert pass_at_k(10, 0, 5) == 0.0        # nothing correct -> never passes
    assert pass_at_k(10, 10, 1) == 1.0       # everything correct -> always passes
    assert pass_at_k(10, 1, 10) == 1.0       # drawing all samples finds the one correct
    assert abs(pass_at_k(10, 1, 1) - 0.1) < 1e-12


def test_pass_at_k_monotonic_in_k():
    values = [pass_at_k(20, 4, k) for k in range(1, 21)]
    assert values == sorted(values), values


def test_pass_at_k_rejects_bad_input():
    for bad in [(0, 0, 1), (10, 11, 2), (10, 5, 0), (10, 5, 11)]:
        try:
            pass_at_k(*bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


# ---------------------------------------------------------------------------
# mcnemar
# ---------------------------------------------------------------------------

def test_mcnemar_no_discordant_pairs():
    r = mcnemar(0, 0)
    assert r["p_value"] == 1.0
    assert r["direction"] == "no change"
    assert not r["significant"]


def test_mcnemar_symmetric_is_not_significant():
    r = mcnemar(10, 10)
    assert r["direction"] == "no change"
    assert r["p_value"] > 0.9


def test_mcnemar_known_value():
    """b=1, c=9: continuity-corrected chi2 = (|1-9|-1)^2/10 = 4.9, p ~= 0.0268."""
    r = mcnemar(1, 9)
    assert abs(r["statistic"] - 4.9) < 1e-9, r["statistic"]
    assert abs(r["p_value"] - 0.0268) < 1e-3, r["p_value"]
    assert r["significant"]
    assert r["direction"] == "improved"


def test_mcnemar_continuity_correction_is_conservative():
    """The corrected p-value must be larger (more conservative) than the raw one."""
    assert mcnemar(2, 12, correction=True)["p_value"] > mcnemar(2, 12, correction=False)["p_value"]


def test_mcnemar_direction():
    assert mcnemar(12, 2)["direction"] == "regressed"
    assert mcnemar(2, 12)["direction"] == "improved"


def test_mcnemar_flags_small_samples():
    """b+c < 25 makes the chi-square approximation shaky and must be flagged."""
    assert mcnemar(2, 10)["exact_recommended"] is True
    assert mcnemar(15, 15)["exact_recommended"] is False


def test_mcnemar_p_value_in_range():
    for b in range(0, 30):
        for c in range(0, 30):
            p = mcnemar(b, c)["p_value"]
            assert 0.0 <= p <= 1.0, (b, c, p)


def _run_all():
    """Run every test in this module without pytest installed."""
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
