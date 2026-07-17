#!/usr/bin/env python3
"""Trait analysis: which trait predicts baseline (ses-1) graph metrics?

For each global graph metric, fits an OLS model (HC3 robust SEs) of the
metric on z-scored trait scores at baseline, and FDR-corrects across the
traits within each metric.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.common import config as cfg
from scripts.common import testing
from scripts.common.constants import ALL_GLOBAL_METRICS, SESSION_ORDER
from scripts.common.stats import fdr_correct, find_id_column, zscore_safe

DEFAULT_TRAITS = ["AM", "ME", "PSQI"]


def load_surveys(path, traits) -> pd.DataFrame:
    s = pd.read_csv(path, sep=";")
    sid = find_id_column(s.columns)
    if sid is None:
        raise ValueError(f"Could not find a subject-ID column in {path}.")
    s = s.rename(columns={sid: "subject_id"})
    s["subject_id"] = s["subject_id"].astype("string").str.strip()
    for t in traits:
        s[t] = pd.to_numeric(s[t], errors="coerce")
    return s


def melt_metric(df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    kid = find_id_column(df.columns)
    if kid is None:
        raise ValueError(f"No subject-ID column found for metric '{metric_name}'.")
    df = df.rename(columns={kid: "subject_id"}).copy()
    df["subject_id"] = df["subject_id"].astype("string").str.strip()
    sess_cols = [c for c in df.columns if str(c).startswith("ses-")]
    long = df.melt(id_vars="subject_id", value_vars=sess_cols, var_name="session", value_name=metric_name)
    long["session"] = pd.Categorical(long["session"], categories=SESSION_ORDER, ordered=True)
    long[metric_name] = pd.to_numeric(long[metric_name], errors="coerce")
    return long


def analyze(metrics_dir: Path, metrics: list[str], survey_file, traits: list[str], session: str = "ses-1") -> pd.DataFrame:
    surv = load_surveys(survey_file, traits)
    rows = []

    for metric_name in metrics:
        mpath = metrics_dir / f"{metric_name}_all_subjects.csv"
        if not mpath.exists():
            print(f"Skipping {metric_name}: {mpath} not found.")
            continue
        met = pd.read_csv(mpath, sep=None, engine="python")
        long_df = melt_metric(met, metric_name)
        df = long_df.merge(surv[["subject_id"] + traits], on="subject_id", how="inner")

        d1 = df[df["session"] == session].dropna(subset=[metric_name] + traits).copy()
        if d1.empty:
            print(f"Skipping {metric_name}: no complete rows at {session}.")
            continue

        for t in traits:
            d1[f"z_{t}"] = zscore_safe(d1[t])

        formula = f"{metric_name} ~ " + " + ".join(f"z_{t}" for t in traits)
        fit = smf.ols(formula, data=d1).fit(cov_type="HC3")

        preds = [f"z_{t}" for t in traits if f"z_{t}" in fit.params.index]
        pvals = [float(fit.pvalues[p]) for p in preds]
        rejected, p_fdr = fdr_correct(pvals) if pvals else ([], [])

        for pred, sig, padj in zip(preds, rejected, p_fdr):
            tname = pred.replace("z_", "")
            rows.append({
                "metric": metric_name,
                "session": session,
                "predictor": tname,
                "coef": float(fit.params[pred]),
                "p_raw": float(fit.pvalues[pred]),
                "p_FDR": float(padj),
                "significant": bool(sig),
                "N": int(fit.nobs),
                "R2": float(fit.rsquared),
            })

    return pd.DataFrame(rows).sort_values(["metric", "predictor"]).reset_index(drop=True) if rows else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cfg.add_config_arg(parser)
    parser.add_argument("--survey-csv", help="Semicolon-separated CSV with a subject-ID column and trait columns.")
    parser.add_argument("--metrics-dir", default="outputs/graph_metrics")
    parser.add_argument("--metrics", nargs="*", default=None)
    parser.add_argument("--traits", nargs="*", default=DEFAULT_TRAITS)
    parser.add_argument("--session", default="ses-1")
    parser.add_argument("--output-dir", default="outputs/behavior")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        tmp_dir = Path(tempfile.mkdtemp(prefix="traits_baseline_test_"))
        subjects = [f"sub-{i:02d}" for i in range(6)]
        survey_csv = testing.write_synthetic_survey_csv(tmp_dir / "survey.csv", subjects=subjects, kind="trait")
        metrics_dir = tmp_dir
        testing.write_synthetic_global_metric_csv(tmp_dir / "global_efficiency_all_subjects.csv", subjects=subjects, seed=3)
        metrics = ["global_efficiency"]
        traits = DEFAULT_TRAITS
        output_dir = tmp_dir / "out"
    else:
        if not args.survey_csv:
            parser.error("--survey-csv is required.")
        survey_csv = args.survey_csv
        metrics_dir = Path(args.metrics_dir)
        metrics = args.metrics or ALL_GLOBAL_METRICS
        traits = args.traits
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    result = analyze(metrics_dir, metrics, survey_csv, traits, session=args.session)
    out_csv = output_dir / f"global_graph_metrics_{args.session}_traits.csv"
    result.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    print(result)

    if args.test:
        assert not result.empty, "Test fixture produced no trait-vs-baseline results"
        print("PASS")


if __name__ == "__main__":
    main()
