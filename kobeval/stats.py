"""Statistical helpers for KobEval-TH.

Pure Python + math only: these must import and run on a CPU-only machine with no
torch installed, because the blog's static builds and CI call them directly.
"""

from __future__ import annotations

import math
from typing import Tuple

__all__ = ["wilson_ci", "mcnemar", "pass_at_k"]


def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    This is deliberately NOT the normal approximation (p +/- z*sqrt(p(1-p)/n)).
    With n=30 per slice -- which is exactly the KobEval-TH slice size -- the
    normal approximation produces intervals that fall outside [0, 1] and
    collapses to zero width at p=0 or p=1, which would let a notebook claim a
    model scored "0.0 +/- 0.0". The Wilson interval stays inside [0, 1] and
    keeps a sensible width at the boundaries.

        centre = (p + z^2/2n) / (1 + z^2/n)
        halfwidth = z/(1 + z^2/n) * sqrt(p(1-p)/n + z^2/4n^2)

    Args:
        successes: number of successes (0 <= successes <= n).
        n: number of trials.
        z: normal quantile; 1.96 gives a 95% interval.

    Returns:
        (low, high), each clamped into [0.0, 1.0].

    >>> lo, hi = wilson_ci(50, 100)
    >>> round(lo, 3), round(hi, 3)
    (0.404, 0.596)
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if not (0 <= successes <= n):
        raise ValueError(f"successes={successes} out of range for n={n}")
    if n == 0:
        # No evidence at all -> the full unit interval, not a divide-by-zero.
        return (0.0, 1.0)

    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    halfwidth = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (max(0.0, centre - halfwidth), min(1.0, centre + halfwidth))


def mcnemar(b: int, c: int, correction: bool = True) -> dict:
    """Continuity-corrected McNemar test for two paired binary classifiers.

    Use this to answer the only question that matters after a fine-tune:
    "did the model actually change, or did I just reshuffle which items it got
    right?" Accuracy going from 12/30 to 14/30 can hide 8 items fixed and 6
    items broken -- McNemar sees the 8 and the 6, a raw accuracy delta does not.

    Args:
        b: items the BASELINE got right and the NEW model got wrong (regressions).
        c: items the BASELINE got wrong and the NEW model got right (fixes).
        correction: apply Edwards' continuity correction (recommended, default).

    Returns:
        dict with keys: b, c, n_discordant, statistic, p_value, significant
        (p < 0.05), and direction ("improved" / "regressed" / "no change").

    Note:
        The chi-square approximation is unreliable when b + c < 25. The returned
        dict therefore carries "exact_recommended": True in that case, which the
        notebooks surface as a warning rather than silently reporting a p-value.
    """
    if b < 0 or c < 0:
        raise ValueError("b and c must be non-negative")

    n_disc = b + c
    if n_disc == 0:
        return {
            "b": b,
            "c": c,
            "n_discordant": 0,
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "direction": "no change",
            "exact_recommended": False,
        }

    diff = abs(b - c)
    if correction:
        diff = max(0.0, diff - 1.0)  # Edwards' continuity correction
    stat = (diff * diff) / n_disc

    # Survival function of chi-square with 1 dof == erfc(sqrt(stat/2)).
    p_value = math.erfc(math.sqrt(stat / 2.0)) if stat > 0 else 1.0

    if c > b:
        direction = "improved"
    elif b > c:
        direction = "regressed"
    else:
        direction = "no change"

    return {
        "b": b,
        "c": c,
        "n_discordant": n_disc,
        "statistic": stat,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "direction": direction,
        "exact_recommended": n_disc < 25,
    }


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021, "Codex").

        pass@k = 1 - C(n - c, k) / C(n, k)

    Args:
        n: total samples drawn for the problem.
        c: number of those samples that were correct.
        k: the k in pass@k (k <= n).

    Returns:
        Probability in [0, 1] that at least one of k samples drawn without
        replacement from the n is correct.

    Implementation note:
        Computed with the stable product form

            1 - prod_{i = n-c+1}^{n} (1 - k/i)

        rather than as a literal ratio of two binomial coefficients. For the
        sampling budgets used in these tutorials the direct ratio is exact in
        Python's arbitrary-precision ints, so both agree -- but the product form
        is what you want if you ever move this to numpy floats.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not (0 <= c <= n):
        raise ValueError(f"c={c} out of range for n={n}")
    if not (1 <= k <= n):
        raise ValueError(f"k={k} out of range for n={n}")

    if n - c < k:
        # Fewer than k incorrect samples exist, so any draw of k must contain
        # a correct one.
        return 1.0
    return 1.0 - math.prod(1.0 - k / i for i in range(n - c + 1, n + 1))
