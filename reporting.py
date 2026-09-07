"""
reporting.py
============
Turn the analysis outputs into the exact tables and figure-data files that
appear in the manuscript, written to ./results/ as CSV + LaTeX snippets.

Nothing here fabricates numbers; every value is passed in from a real
computation upstream.
"""

from __future__ import annotations
import os
from typing import Dict

import numpy as np
import pandas as pd

import config as C


def _save(df: pd.DataFrame, name: str, float_fmt: str = "%.4f") -> None:
    csv = os.path.join(C.PATHS.tables_dir, f"{name}.csv")
    tex = os.path.join(C.PATHS.tables_dir, f"{name}.tex")
    df.to_csv(csv, index=False)
    with open(tex, "w") as f:
        f.write(df.to_latex(index=False, float_format=lambda x: float_fmt % x
                            if isinstance(x, (int, float, np.floating)) else x,
                            escape=False))
    print(f"  wrote {csv} and {tex}")


def export_main_attribution(weighted, top_rank, scores_sum, modality) -> None:
    """Table 6/7: top-rank frequency + weighted share + normalized score."""
    rows = []
    for g in weighted.index:
        disp = C.GROUP_DISPLAY.get(g, g)
        rows.append({
            "Feature group": disp,
            "Top-rank freq (of 360)": int(top_rank.get(g, 0)),
            "Weighted share (%)": round(100 * weighted[g], 2),
        })
    df = pd.DataFrame(rows)
    _save(df, "table6_attribution", float_fmt="%.2f")

    with open(os.path.join(C.PATHS.tables_dir, "modality_split.txt"), "w") as f:
        f.write(f"semantic = {modality['semantic']:.1f}%\n")
        f.write(f"structural = {modality['structural']:.1f}%\n")
    print(f"  modality: semantic {modality['semantic']:.1f}% / "
          f"structural {modality['structural']:.1f}%")


def export_per_stratum(shares: pd.DataFrame) -> None:
    """Figure 5 data: group x tau share matrix (%)."""
    out = shares.copy()
    out.index = [C.GROUP_DISPLAY.get(g, g) for g in out.index]
    csv = os.path.join(C.PATHS.figures_dir, "figure5_gradient.csv")
    out.to_csv(csv)
    # column sums (should be ~100 per stratum)
    sums = out.sum(axis=0)
    print(f"  Figure 5 column sums: {sums.round(1).to_dict()}")
    print(f"  wrote {csv}")


def export_trend_table(trend: pd.DataFrame) -> None:
    """Table 8: Spearman + Mann-Kendall + quadratic."""
    t = trend.copy()
    t["group"] = t["group"].apply(lambda g: C.GROUP_DISPLAY.get(g, g))
    _save(t, "table8_trend", float_fmt="%.3f")


def export_ablation(df: pd.DataFrame) -> None:
    _save(df, "table_ablation", float_fmt="%.4f")


def export_memorization(df: pd.DataFrame) -> None:
    _save(df, "table_memorization", float_fmt="%.4f")


def export_perf(df_perf: pd.DataFrame) -> None:
    """Table 5: best configuration per stratum."""
    _save(df_perf, "table5_performance", float_fmt="%.4f")


def export_lift(lift_model: Dict, lift_base: Dict) -> None:
    df = pd.DataFrame({
        "decile": lift_model["deciles"],
        "hybrid_gain": lift_model["gain"],
        "bibliometric_gain": lift_base["gain"],
        "random_gain": lift_model["deciles"],
    })
    csv = os.path.join(C.PATHS.figures_dir, "figure_lift.csv")
    df.to_csv(csv, index=False)
    print(f"  LIFT top-20% capture: hybrid {lift_model['top20_capture']:.1f}% "
          f"vs bibliometric {lift_base['top20_capture']:.1f}%")
    print(f"  wrote {csv}")


def export_consistency_report(df_runs, bootstrap, uw_rho) -> None:
    """A machine-checkable summary a reviewer could reproduce."""
    path = os.path.join(C.PATHS.results_dir, "consistency_report.txt")
    with open(path, "w") as f:
        f.write(f"total runs: {len(df_runs)} (paper: 360)\n")
        f.write(f"unweighted-vs-weighted rank rho: {uw_rho:.3f} (paper: > 0.90)\n")
        f.write(f"bootstrap mean rank rho: {bootstrap['mean_spearman']:.3f} "
                f"[{bootstrap['p05']:.3f}, {bootstrap['p95']:.3f}]\n")
    print(f"  wrote {path}")
