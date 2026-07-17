"""Shared statistics helpers: LMM-with-reference-session fitting and FDR correction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from scripts.common.constants import SESSION_ORDER


def run_lmm_with_ref(
    df_long: pd.DataFrame,
    ref_session: str,
    value_col: str = "value",
    session_col: str = "session",
    group_col: str = "subject_id",
    session_order: list[str] = SESSION_ORDER,
) -> tuple:
    """Fit `value ~ session` mixed model with `ref_session` as the reference
    category, so the fixed effects give ref-vs-other contrasts directly.

    Returns (fitted_result, rows) where rows is a list of dicts, one per
    fixed-effect term (including Intercept), with Estimate/SE/p-value/CI.
    """
    df_long = df_long.copy()
    df_long[session_col] = pd.Categorical(df_long[session_col], categories=session_order)
    reordered = [ref_session, *(s for s in session_order if s != ref_session)]
    df_long[session_col] = df_long[session_col].cat.reorder_categories(reordered)

    formula = f"{value_col} ~ {session_col}"
    model = smf.mixedlm(formula, data=df_long, groups=df_long[group_col])
    result = model.fit()
    conf_int = result.conf_int()

    rows = []
    for param in result.fe_params.index:
        rows.append({
            "Reference Session": ref_session,
            "Effect": param,
            "Estimate": result.fe_params[param],
            "SE": result.bse[param],
            "p-value": result.pvalues[param],
            "CI Lower Bound": conf_int.loc[param][0],
            "CI Upper Bound": conf_int.loc[param][1],
        })
    return result, rows


def fdr_correct(pvals, alpha: float = 0.05, method: str = "fdr_bh"):
    """Wrapper around statsmodels multipletests. Returns (rejected, pvals_corrected)."""
    pvals = np.asarray(pvals, dtype=float)
    if pvals.size == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)
    rejected, pvals_corrected, _, _ = multipletests(pvals, alpha=alpha, method=method)
    return rejected, pvals_corrected


def cohens_d(observed: float, null_distribution: np.ndarray) -> float:
    """Standardized effect size of an observed statistic against a null distribution."""
    mean_null = np.mean(null_distribution)
    std_null = np.std(null_distribution, ddof=1)
    return (observed - mean_null) / std_null


def find_id_column(columns) -> str | None:
    """Case-insensitive search for a subject-ID-like column name."""
    for c in columns:
        if str(c).lower() in ("subject_id", "subjectid", "id", "subject"):
            return c
    return None


def zscore_safe(x: pd.Series) -> pd.Series:
    """z-score that returns all-zeros (instead of NaN/inf) for a constant column."""
    x = pd.to_numeric(x, errors="coerce")
    sd = np.nanstd(x, ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - np.nanmean(x)) / sd
