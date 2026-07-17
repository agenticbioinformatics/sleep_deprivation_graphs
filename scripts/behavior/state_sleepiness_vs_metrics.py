#!/usr/bin/env python3
"""State analysis: does subjective sleepiness (KSS) track global graph
metrics, separately per session?

Fits `graph_metric ~ 0 + C(session) + C(session):sleepiness` (random subject
intercept, and slope where the model converges) per global metric, extracts
the per-session sleepiness slope, and FDR-corrects the 3 session slopes
within each metric.
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
from scripts.common.stats import fdr_correct

KSS_TO_SESSION = {"KSS-B1": "ses-1", "KSS-A1": "ses-2", "KSS-C1": "ses-3"}
SESSION_NICE_NAME = {"ses-1": "Baseline (B)", "ses-2": "Acute (A)", "ses-3": "Chronic (C)"}
FORMULA = "graph_metric ~ 0 + C(session) + C(session):sleepiness"


def load_sleepiness_long(survey_csv) -> pd.DataFrame:
    sleep = pd.read_csv(survey_csv, sep=";")
    if "Subject_ID" not in sleep.columns:
        raise ValueError(f"Expected a 'Subject_ID' column in {survey_csv}, found: {list(sleep.columns)}")
    sleep = sleep.rename(columns={"Subject_ID": "subject_id"})
    sleep["subject_id"] = sleep["subject_id"].astype(str).str.strip()

    kss_cols = list(KSS_TO_SESSION.keys())
    missing = [c for c in kss_cols if c not in sleep.columns]
    if missing:
        raise ValueError(f"Missing expected KSS columns in {survey_csv}: {missing}")

    sleep = sleep[["subject_id"] + kss_cols]
    long = sleep.melt(id_vars=["subject_id"], value_vars=kss_cols, var_name="kss_session", value_name="sleepiness")
    long["session"] = long["kss_session"].map(KSS_TO_SESSION)
    return long


def fit_model(df, re_formula):
    md = smf.mixedlm(FORMULA, df, groups=df["subject_id"], re_formula=re_formula)
    try:
        return md.fit(reml=False, method="lbfgs", maxiter=200)
    except Exception:
        return md.fit(reml=True, method="lbfgs", maxiter=200)


def analyze_one_metric(metric_name, graph_csv_path, sleep_long) -> pd.DataFrame:
    graph = pd.read_csv(graph_csv_path)
    graph["subject_id"] = graph["subject_id"].astype(str).str.strip()
    graph_long = graph.melt(id_vars="subject_id", value_vars=SESSION_ORDER, var_name="session", value_name="graph_metric")

    df = pd.merge(sleep_long, graph_long, on=["subject_id", "session"], how="inner")
    df = df.dropna(subset=["sleepiness", "graph_metric"])
    df["session"] = pd.Categorical(df["session"], categories=SESSION_ORDER, ordered=True)

    dup_counts = df.groupby(["subject_id", "session"]).size()
    if not dup_counts.empty and dup_counts.max() > 1:
        raise ValueError(f"[{metric_name}] duplicate rows per subject-session detected.")
    if df.empty:
        print(f"[{metric_name}] no overlapping subject/session data; skipping.")
        return pd.DataFrame()

    try:
        m = fit_model(df, re_formula="~sleepiness")
    except Exception:
        m = fit_model(df, re_formula="~1")

    params, bse, conf = m.params, m.bse, m.conf_int()
    conf.columns = ["CI_low", "CI_high"]

    rows = []
    for s in SESSION_ORDER:
        term = f"C(session)[{s}]:sleepiness"
        if term in params.index:
            beta, se = params[term], bse[term]
            rows.append({
                "metric": metric_name,
                "session": SESSION_NICE_NAME[s],
                "slope (Δgraph per 1 KSS pt)": beta,
                "SE": se,
                "z": beta / se,
                "p_raw": m.pvalues[term],
                "CI_low": conf.loc[term, "CI_low"],
                "CI_high": conf.loc[term, "CI_high"],
                "n_obs": int(m.model.endog.size),
                "n_subj": df["subject_id"].nunique(),
            })
    print(m.summary())
    slopes = pd.DataFrame(rows)
    if not slopes.empty:
        rejected, p_fdr = fdr_correct(slopes["p_raw"].values)
        slopes["p_FDR"] = p_fdr
        slopes["FDR_sig@0.05"] = rejected
    return slopes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cfg.add_config_arg(parser)
    parser.add_argument("--survey-csv", help="Semicolon-separated CSV with Subject_ID, KSS-B1, KSS-A1, KSS-C1 columns.")
    parser.add_argument("--metrics-dir", default="outputs/graph_metrics")
    parser.add_argument("--metrics", nargs="*", default=None)
    parser.add_argument("--output-dir", default="outputs/behavior")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        tmp_dir = Path(tempfile.mkdtemp(prefix="state_test_"))
        subjects = [f"sub-{i:02d}" for i in range(6)]
        survey_csv = testing.write_synthetic_survey_csv(tmp_dir / "survey.csv", subjects=subjects, kind="state")
        metrics_dir = tmp_dir
        testing.write_synthetic_global_metric_csv(tmp_dir / "global_efficiency_all_subjects.csv", subjects=subjects, seed=2)
        metrics = ["global_efficiency"]
        output_dir = tmp_dir / "out"
    else:
        if not args.survey_csv:
            parser.error("--survey-csv is required.")
        survey_csv = args.survey_csv
        metrics_dir = Path(args.metrics_dir)
        metrics = args.metrics or ALL_GLOBAL_METRICS
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    sleep_long = load_sleepiness_long(survey_csv)

    all_results = []
    for metric_name in metrics:
        graph_csv_path = metrics_dir / f"{metric_name}_all_subjects.csv"
        if not graph_csv_path.exists():
            print(f"Skipping {metric_name}: {graph_csv_path} not found.")
            continue
        slopes = analyze_one_metric(metric_name, graph_csv_path, sleep_long)
        if slopes.empty:
            continue
        out_csv = output_dir / f"{metric_name}_sleepiness_mixed_slopes_FDR.csv"
        slopes.to_csv(out_csv, index=False)
        print(f"\n=== {metric_name} ===\n{slopes.to_string(index=False)}")
        all_results.append(slopes)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined_csv = output_dir / "ALL_global_graph_metrics_KSS_mixed_model.csv"
        combined.to_csv(combined_csv, index=False)
        print(f"\nCombined results: {combined_csv}")
    else:
        combined = pd.DataFrame()
        print("No metrics produced results.")

    if args.test:
        assert not combined.empty, "Test fixture produced no state-vs-sleepiness results"
        print("PASS")


if __name__ == "__main__":
    main()
