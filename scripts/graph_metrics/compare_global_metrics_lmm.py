#!/usr/bin/env python3
"""Linear mixed-model comparison of global graph metrics across sessions.

Fits `value ~ session` (random subject intercept) twice per metric -- once
with ses-1 and once with ses-2 as the reference category -- to get all 3
pairwise session contrasts (TSD vs RW, CSR vs RW, CSR vs TSD), FDR-corrects
across the 3 comparisons per metric, and plots a forest-style CI figure.

Reads the 5 global-metric CSVs written by compute_global_metrics.py
(global_efficiency, average_clustering, average_path_length, modularity)
and compute_community_structure.py (avg_graph_distance) -- both default to
the same --metrics-dir.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import matplotlib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.common import config as cfg
from scripts.common import testing
from scripts.common.constants import ALL_GLOBAL_METRICS
from scripts.common.stats import fdr_correct, run_lmm_with_ref

COMPARISON_TITLES = {("ses-2", "ses-1"): "TSD vs RW", ("ses-3", "ses-1"): "CSR vs RW", ("ses-3", "ses-2"): "CSR vs TSD"}


def fit_lmm_all_metrics(metrics_dir: Path, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for metric_name in metrics:
        path = metrics_dir / f"{metric_name}_all_subjects.csv"
        if not path.exists():
            print(f"Skipping {metric_name}: {path} not found.")
            continue
        df = pd.read_csv(path)
        df_long = df.melt(id_vars="subject_id", var_name="session", value_name="value")

        for ref_session in ["ses-1", "ses-2"]:
            try:
                result, lmm_rows = run_lmm_with_ref(df_long.copy(), ref_session)
                print(f"\n{metric_name} (reference: {ref_session}):\n{result.summary()}")
                for row in lmm_rows:
                    row["Metric"] = metric_name
                    rows.append(row)
            except Exception as e:
                print(f"Error fitting model for {metric_name} with ref {ref_session}: {e}")
    return pd.DataFrame(rows)


def fdr_and_plot(lmm_df: pd.DataFrame, metrics: list[str], output_dir: Path):
    lmm_df = lmm_df[~lmm_df["Effect"].str.contains("Intercept")].copy()
    lmm_df = lmm_df[lmm_df["Metric"].isin(metrics)].copy()
    if lmm_df.empty:
        return lmm_df

    lmm_df["Compared Session"] = lmm_df["Effect"].str.extract(r"session\[T\.(.*?)\]")
    lmm_df["Comparison"] = lmm_df["Compared Session"] + " vs " + lmm_df["Reference Session"]
    lmm_df["p-value (FDR)"] = float("nan")
    lmm_df["Significant (FDR)"] = False

    for comp_ses, ref_ses in COMPARISON_TITLES:
        mask = (lmm_df["Compared Session"] == comp_ses) & (lmm_df["Reference Session"] == ref_ses)
        pvals = lmm_df.loc[mask, "p-value"].values
        if len(pvals) > 0:
            rejected, pvals_corrected = fdr_correct(pvals)
            lmm_df.loc[mask, "p-value (FDR)"] = pvals_corrected
            lmm_df.loc[mask, "Significant (FDR)"] = rejected

    def sig_marker(p):
        if p <= 0.001:
            return "***"
        if p <= 0.01:
            return "**"
        if p <= 0.05:
            return "*"
        if p <= 0.1:
            return "#"
        return ""

    def color(p):
        if p < 0.05:
            return "darkturquoise"
        if p < 0.1:
            return "black"
        return "gray"

    lmm_df["Significance (FDR)"] = lmm_df["p-value (FDR)"].apply(sig_marker)
    lmm_df["Color"] = lmm_df["p-value (FDR)"].apply(color)

    import matplotlib.pyplot as plt

    comparisons = list(COMPARISON_TITLES.keys())
    fig, axes = plt.subplots(1, len(comparisons), figsize=(6 * len(comparisons), 6), sharey=True)
    if len(comparisons) == 1:
        axes = [axes]
    metric_order = metrics[::-1]

    for i, (comp_ses, ref_ses) in enumerate(comparisons):
        df_comp = lmm_df[(lmm_df["Compared Session"] == comp_ses) & (lmm_df["Reference Session"] == ref_ses)].copy()
        df_comp["Metric"] = pd.Categorical(df_comp["Metric"], categories=metric_order, ordered=True)
        df_comp = df_comp.sort_values("Metric")

        ax = axes[i]
        ax.axvline(x=0, color="gray", linestyle="--")
        for _, row in df_comp.iterrows():
            ax.plot([row["CI Lower Bound"], row["CI Upper Bound"]], [row["Metric"], row["Metric"]], color=row["Color"], linewidth=4)
            ax.plot(row["Estimate"], row["Metric"], "o", color=row["Color"], markersize=10)
            ax.text(row["CI Upper Bound"], row["Metric"], f'{row["Estimate"]:.4f}{row["Significance (FDR)"]}', va="center", fontsize=11)
        ax.set_title(COMPARISON_TITLES[(comp_ses, ref_ses)], fontsize=16)
        ax.set_xlabel("Estimate")
        if i == 0:
            ax.set_ylabel("Graph metric")

    fig.tight_layout()
    fig.savefig(output_dir / "lmm_forest_plot.png")
    plt.close(fig)
    return lmm_df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cfg.add_config_arg(parser)
    parser.add_argument("--metrics-dir", default="outputs/graph_metrics", help="Directory with *_all_subjects.csv files from compute_global_metrics.py / compute_community_structure.py.")
    parser.add_argument("--metrics", nargs="*", default=None)
    parser.add_argument("--output-dir", default="outputs/graph_metrics")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    matplotlib.use("Agg")

    if args.test:
        tmp_dir = Path(tempfile.mkdtemp(prefix="lmm_test_"))
        metrics_dir = tmp_dir
        for m in ALL_GLOBAL_METRICS:
            testing.write_synthetic_global_metric_csv(tmp_dir / f"{m}_all_subjects.csv", subjects=[f"sub-{i:02d}" for i in range(6)])
        output_dir = tmp_dir
        metrics = ALL_GLOBAL_METRICS
    else:
        metrics_dir = Path(args.metrics_dir)
        output_dir = Path(args.output_dir)
        metrics = args.metrics or ALL_GLOBAL_METRICS

    output_dir.mkdir(parents=True, exist_ok=True)
    lmm_df = fit_lmm_all_metrics(metrics_dir, metrics)
    lmm_path = output_dir / "lmm_results_all_metrics_with_ref.csv"
    lmm_df.to_csv(lmm_path, index=False)
    print(f"\nSaved: {lmm_path}")

    fdr_df = fdr_and_plot(lmm_df, metrics, output_dir)
    fdr_path = output_dir / "lmm_results_with_FDR.csv"
    fdr_df.to_csv(fdr_path, index=False)
    print(f"Saved: {fdr_path}")

    if args.test:
        assert not fdr_df.empty, "Test fixture produced no FDR-corrected LMM results"
        print("PASS")


if __name__ == "__main__":
    main()
