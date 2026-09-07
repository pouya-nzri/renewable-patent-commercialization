"""
statistics_tests.py
===================
Trend and ordered-alternative tests behind Table 8 and the gradient claims.

Implements exactly what the paper reports:
  * Spearman rho with t-approximation p-value (n = 6 strata)         [paper]
  * Mann-Kendall S with EXACT two-sided permutation p-value; tie-corrected
    normal approximation when ties are present (assignee series)     [paper]
  * quadratic fit a (leading coefficient) and vertex                 [paper]
  * Jonckheere-Terpstra ordered test on run-level replicates         [paper]

These are the same computations used to audit the manuscript, so numbers
produced here will line up with Table 8 when run on the real shares.
"""

from __future__ import annotations
import itertools
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

# Precompute the exact null distribution of Mann-Kendall S for n=6 (no ties).
_N6 = 6
_ALL_S_N6 = None


def _mk_S(x: List[float]) -> int:
    n = len(x)
    return int(sum(np.sign(x[j] - x[i]) for i in range(n) for j in range(i + 1, n)))


def _exact_mk_null_n6() -> np.ndarray:
    global _ALL_S_N6
    if _ALL_S_N6 is None:
        _ALL_S_N6 = np.array([_mk_S(list(p)) for p in itertools.permutations(range(_N6))])
    return _ALL_S_N6


def spearman_trend(tau: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Spearman rho and its t-approximation p-value."""
    rho, p = stats.spearmanr(tau, y)
    return float(rho), float(p)


def mann_kendall(y: List[float]) -> Dict[str, float]:
    """
    Mann-Kendall S with exact two-sided p (n=6, untied) or tie-corrected
    normal approximation when ties are present.
    """
    y = list(y)
    n = len(y)
    S = _mk_S(y)

    # detect ties
    _, counts = np.unique(y, return_counts=True)
    has_ties = np.any(counts > 1)

    if n == 6 and not has_ties:
        null = _exact_mk_null_n6()
        p = float((np.abs(null) >= abs(S)).mean())
        method = "exact_permutation"
    else:
        # tie-corrected variance, continuity-corrected normal approx
        var = (n * (n - 1) * (2 * n + 5)
               - sum(t * (t - 1) * (2 * t + 5) for t in counts)) / 18.0
        if var <= 0:
            z, p = 0.0, 1.0
        else:
            z = (S - np.sign(S)) / np.sqrt(var)
            p = float(2 * (1 - stats.norm.cdf(abs(z))))
        method = "normal_tie_corrected"
    return {"S": int(S), "p": p, "method": method}


def quadratic_fit(tau: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Leading coefficient a and vertex of a degree-2 fit in tau."""
    a, b, c = np.polyfit(tau, y, 2)
    vertex = float(-b / (2 * a)) if a != 0 else float("nan")
    return {"a": float(a), "b": float(b), "c": float(c), "vertex": vertex}


def trend_table(shares: "pd.DataFrame") -> "pd.DataFrame":
    """
    Build the full Table-8 style trend table from a group x tau share matrix
    (rows = groups, columns = tau). Adds a deep-semantic (claims+abstract) row.
    """
    import pandas as pd
    tau = np.array(sorted(shares.columns))
    rows = []

    def _row(name, series):
        rho, rp = spearman_trend(tau, series)
        mk = mann_kendall(list(series))
        q = quadratic_fit(tau, series)
        if abs(rho) >= 0.9 and mk["p"] < 0.05:
            trend = "Increase" if rho > 0 else "Decrease"
        elif abs(q["a"]) > 0.5 and q["a"] < 0:
            trend = f"Inverted-U (peak tau~{q['vertex']:.1f})"
        elif rho > 0:
            trend = "Increase"
        else:
            trend = "Decrease"
        return {
            "group": name, "rho": round(rho, 3), "rho_p": round(rp, 3),
            "S": mk["S"], "mk_p": round(mk["p"], 3), "a": round(q["a"], 2),
            "vertex": round(q["vertex"], 2), "trend": trend, "mk_method": mk["method"],
        }

    for g in shares.index:
        rows.append(_row(g, shares.loc[g].values.astype(float)))

    # deep-semantic sum
    if "claims" in shares.index and "abstract" in shares.index:
        ds = (shares.loc["claims"] + shares.loc["abstract"]).values.astype(float)
        rows.append(_row("deep_semantic(claims+abstract)", ds))

    return pd.DataFrame(rows)


def jonckheere_terpstra(groups_by_stratum: List[np.ndarray]) -> Dict[str, float]:
    """
    Jonckheere-Terpstra test for an ordered alternative across strata,
    using the run-level replicates (not just the 6 stratum means), which is
    the "full replicate power" test the paper mentions.

    groups_by_stratum: list of arrays, one per ordered stratum (D3..D8),
    each holding that stratum's run-level shares for one group.
    """
    k = len(groups_by_stratum)
    JT = 0
    for i in range(k):
        for j in range(i + 1, k):
            a = groups_by_stratum[i]
            b = groups_by_stratum[j]
            # count pairs where later stratum exceeds earlier (Mann-Whitney U form)
            u = 0.0
            for x in a:
                u += np.sum(b > x) + 0.5 * np.sum(b == x)
            JT += u

    # normal approximation
    ns = np.array([len(g) for g in groups_by_stratum], dtype=float)
    N = ns.sum()
    mean_JT = (N ** 2 - np.sum(ns ** 2)) / 4.0
    var_JT = (N ** 2 * (2 * N + 3) - np.sum(ns ** 2 * (2 * ns + 3))) / 72.0
    z = (JT - mean_JT) / np.sqrt(var_JT) if var_JT > 0 else 0.0
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    return {"JT": float(JT), "z": float(z), "p": p}
