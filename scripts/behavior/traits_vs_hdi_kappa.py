#!/usr/bin/env python3
"""Trait analysis: which trait predicts the within-subject HDI kappas
(degree/closeness/clustering reorganization after sleep loss)?

For each HDI metric's kappas_<metric>.csv (from compute_hdi.py) and each
kappa column in it, fits an OLS model (HC3 robust SEs) of that kappa on
z-scored trait scores, and FDR-corrects across traits within each
metric/kappa-column combination.
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
from scripts.common.constants import HDI_CCML_METRICS, METRIC_FILE_SLUG
from scripts.common.stats import fdr_correct, find_id_column, zscore_safe

DEFAULT_TRAITS = ["AM", "ME", "PSQI"]


def analyze_metric(metric_name: str, kappa_path, survey_path, traits: list[str]) -> pd.DataFrame:
    kappa_df = pd.read_csv(kappa_path)
    survey_df = pd.read_csv(survey_path, sep=";")

    kid = find_id_column(kappa_df.columns)
    sid = find_id_column(survey_df.columns)
    if kid is None or sid is None:
        raise ValueError(f"Could not find a subject-ID column in {kappa_path} or {survey_path}.")
    kappa_df = kappa_df.rename(columns={kid: "subject_id"})
    survey_df = survey_df.rename(columns={sid: "subject_id"})
    for df in (kappa_df, survey_df):
        df["subject_id"] = df["subject_id"].astype("string").str.strip()

    df = pd.merge(kappa_df, survey_df, on="subject_id", how="inner")
    kappa_cols = [c for c in kappa_df.columns if c != "subject_id"]

    all_rows = []
    print(f"\n### METRIC: {metric_name.upper()} ###")
    for k_col in kappa_cols:
        cols = [k_col] + traits
        sub = df[["subject_id"] + cols].copy()
        for c in cols:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
        sub = sub.dropna(subset=cols)
        print(f"\n=== {k_col} ===")
        if sub.empty:
            print("No data after dropna; skipping.")
            continue

        for t in traits:
            sub[f"z_{t}"] = zscore_safe(sub[t])

        formula = f"{k_col} ~ " + " + ".join(f"z_{t}" for t in traits)
        fit = smf.ols(formula, data=sub).fit(cov_type="HC3")

        tmp_rows, pvals = [], []
        for pred in [f"z_{t}" for t in traits]:
            if pred in fit.params.index:
                tmp_rows.append({
                    "Metric": metric_name, "Kappa_Column": k_col, "Psych": pred.replace("z_", ""),
                    "N": int(fit.nobs), "Test": "OLS",
                    "Effect": float(fit.params[pred]), "p": float(fit.pvalues[pred]), "R2": float(fit.rsquared),
                })
                pvals.append(float(fit.pvalues[pred]))

        rejected, p_fdr = fdr_correct(pvals) if pvals else ([], [])
        for row, sig, padj in zip(tmp_rows, rejected, p_fdr):
            row["p_FDR"] = padj
            row["Sig_FDR"] = bool(sig)
            all_rows.append(row)
            print(f"OLS | {row['Psych']:8} | N={row['N']:3d} | beta = {row['Effect']: .3f} | p = {row['p']:.4f} | p_FDR = {padj:.4f} | Sig_FDR = {sig} | R2={row['R2']:.3f}")

    return pd.DataFrame(all_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cfg.add_config_arg(parser)
    parser.add_argument("--survey-csv", help="Semicolon-separated CSV with a subject-ID column and trait columns.")
    parser.add_argument("--kappa-dir", default="outputs/hdi", help="Directory with kappas_<metric>.csv from compute_hdi.py.")
    parser.add_argument("--metrics", nargs="*", default=None)
    parser.add_argument("--traits", nargs="*", default=DEFAULT_TRAITS)
    parser.add_argument("--output-dir", default="outputs/behavior")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        tmp_dir = Path(tempfile.mkdtemp(prefix="traits_kappa_test_"))
        subjects = [f"sub-{i:02d}" for i in range(6)]
        survey_csv = testing.write_synthetic_survey_csv(tmp_dir / "survey.csv", subjects=subjects, kind="trait")
        kappa_dir = tmp_dir
        testing.write_synthetic_kappa_csv(tmp_dir / "kappas_degree.csv", subjects=subjects)
        metrics = ["Degree_centrality"]
        traits = DEFAULT_TRAITS
        output_dir = tmp_dir / "out"
    else:
        if not args.survey_csv:
            parser.error("--survey-csv is required.")
        survey_csv = args.survey_csv
        kappa_dir = Path(args.kappa_dir)
        metrics = args.metrics or HDI_CCML_METRICS
        traits = args.traits
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for metric in metrics:
        kappa_path = kappa_dir / f"kappas_{METRIC_FILE_SLUG[metric]}.csv"
        if not kappa_path.exists():
            print(f"Skipping {metric}: {kappa_path} not found.")
            continue
        results.append(analyze_metric(metric, kappa_path, survey_csv, traits))

    combined = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    out_csv = output_dir / "kappa_vs_traits.csv"
    combined.to_csv(out_csv, index=False)
    print(f"\nSaved combined results to: {out_csv}")

    if args.test:
        assert not combined.empty, "Test fixture produced no kappa-vs-traits results"
        print("PASS")


if __name__ == "__main__":
    main()
