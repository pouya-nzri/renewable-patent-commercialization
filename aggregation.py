"""
aggregation.py
==============
Consensus attribution across the 360 model-dataset-fold runs.

Reproduces, from the per-run SHAP-share records:
  * R^2-weighted aggregated share per group           -> Table 6/7   [paper]
  * top-rank frequency (which group wins each run)     -> Table 6      [paper]
  * unweighted mean + bootstrap CIs + rank stability   -> "rho > 0.90" [paper]
  * per-stratum share matrix                           -> Figure 5     [paper]
"""

from __future__ import annotations
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

import config as C


def runs_to_frame(all_runs: List[dict]) -> pd.DataFrame:
    """
    Flatten run records into a tidy DataFrame with one row per run and one
    column per group share (plus tau, encoder, learner, fold, r2, rmse).
    """
    rows = []
    for r in all_runs:
        row = {
            "tau": r["tau"], "encoder": r["encoder"], "learner": r["learner"],
            "fold": r["fold"], "rmse": r["rmse"], "r2": r["r2"],
        }
        for g, v in r["shares"].items():
            row[f"share__{g}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def _share_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("share__")]


def weighted_shares(df: pd.DataFrame) -> pd.Series:
    """R^2-weighted mean share per group over all runs (weights = max(r2,0))."""
    w = np.clip(df["r2"].values, 0, None)
    if w.sum() == 0:
        w = np.ones_like(w)
    out = {}
    for c in _share_cols(df):
        out[c.replace("share__", "")] = float(np.average(df[c].values, weights=w))
    s = pd.Series(out).sort_values(ascending=False)
    return s / s.sum()               # renormalize to sum to 1


def top_rank_frequency(df: pd.DataFrame) -> pd.Series:
    """Count, per group, the runs in which it holds the single highest share."""
    cols = _share_cols(df)
    winners = df[cols].idxmax(axis=1).str.replace("share__", "", regex=False)
    freq = winners.value_counts()
    # ensure every group present
    for c in cols:
        g = c.replace("share__", "")
        if g not in freq.index:
            freq[g] = 0
    return freq.sort_values(ascending=False)


def bootstrap_rank_stability(df: pd.DataFrame, n_boot: int = 1000,
                             seed: int = C.RANDOM_SEED) -> Dict[str, float]:
    """
    Resample runs with replacement; recompute the weighted ranking each time;
    report the mean Spearman correlation of bootstrap rankings with the full
    ranking. The paper claims this exceeds 0.90.
    """
    base = weighted_shares(df)
    base_rank = base.rank(ascending=False)
    rng = np.random.RandomState(seed)
    corrs = []
    n = len(df)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        s = weighted_shares(df.iloc[idx])
        s = s.reindex(base.index)
        r = s.rank(ascending=False)
        rho, _ = stats.spearmanr(base_rank.values, r.values)
        corrs.append(rho)
    return {
        "mean_spearman": float(np.mean(corrs)),
        "p05": float(np.percentile(corrs, 5)),
        "p95": float(np.percentile(corrs, 95)),
    }


def unweighted_vs_weighted_rho(df: pd.DataFrame) -> float:
    w = weighted_shares(df)
    u_raw = {c.replace("share__", ""): df[c].mean() for c in _share_cols(df)}
    u = pd.Series(u_raw)
    u = (u / u.sum()).reindex(w.index)
    rho, _ = stats.spearmanr(w.rank(ascending=False), u.rank(ascending=False))
    return float(rho)


def modality_split(weighted: pd.Series) -> Dict[str, float]:
    """Aggregate weighted shares into semantic vs structural totals. [paper]"""
    sem = sum(weighted.get(g, 0.0) for g in C.SEMANTIC_GROUPS)
    struc = sum(weighted.get(g, 0.0) for g in C.STRUCTURAL_GROUPS)
    tot = sem + struc or 1.0
    return {"semantic": 100 * sem / tot, "structural": 100 * struc / tot}


def per_stratum_shares(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group x tau matrix of R^2-weighted shares (%), the data behind Figure 5's
    commercialization gradient. Columns are tau=3..8, rows are groups.
    """
    mats = {}
    for tau in C.STRATA:
        sub = df[df["tau"] == tau]
        if len(sub) == 0:
            continue
        mats[tau] = weighted_shares(sub) * 100.0
    out = pd.DataFrame(mats)
    return out
