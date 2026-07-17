#!/usr/bin/env python3
"""Per-region linear mixed-model comparison of nodal graph metrics across
sessions, plus a p-value distribution histogram per metric/comparison.

For each region and each nodal metric, fits `value ~ session` (random
subject intercept) with ses-1 and ses-2 as reference in turn, to get the 3
pairwise session contrasts per region. Reads the per-subject nodal metrics
CSVs written by compute_nodal_metrics.py.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.common import config as cfg
from scripts.common import io, testing
from scripts.common.constants import NODAL_METRICS, NODAL_SESSION_COMPARISONS, SESSION_ORDER
from scripts.common.stats import run_lmm_with_ref

COLS = NODAL_SESSION_COMPARISONS
CONTRAST_MAP = list(zip(["ses-1", "ses-1", "ses-2"], ["session[T.ses-2]", "session[T.ses-3]", "session[T.ses-3]"], COLS))


def load_all_nodal_values(data_root, sessions, metric) -> dict:
    """(session, subject_id) -> pd.Series of `metric` indexed by Node.
    Reads each subject's metrics CSV once (not once per region)."""
    values = {}
    for ses in sessions:
        for sub_path in io.subject_dirs(data_root, ses):
            sub_id = os.path.basename(sub_path)
            metrics_file = io.find_nodal_metrics_file(sub_path)
            if metrics_file is None:
                continue
            df = pd.read_csv(metrics_file).set_index("Node").sort_index()
            if metric in df.columns:
                values[(ses, sub_id)] = df[metric]
    return values


def per_region_lmm(data_root, sessions, metric, n_regions) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = load_all_nodal_values(data_root, sessions, metric)
    pvals_per_region = np.full((n_regions, 3), np.nan)
    estimates_per_region = np.full((n_regions, 3), np.nan)
    lmm_rows = []

    for region in range(n_regions):
        data_long = [
            {"subject_id": sub_id, "session": ses, "value": series.loc[region]}
            for (ses, sub_id), series in values.items()
            if region in series.index
        ]
        df_long = pd.DataFrame(data_long)
        if df_long.empty or df_long["session"].nunique() < 2 or df_long["subject_id"].nunique() < 2:
            continue
        for ref_session in ["ses-1", "ses-2"]:
            try:
                _, rows = run_lmm_with_ref(df_long.copy(), ref_session)
                for row in rows:
                    if row["Effect"] == "Intercept":
                        continue
                    row["Region"] = region
                    lmm_rows.append(row)
            except Exception as e:
                print(f"Error fitting model for {metric} region {region} ref {ref_session}: {e}")

    df_lmm = pd.DataFrame(lmm_rows)
    for i in range(n_regions):
        for ref, contrast, col in CONTRAST_MAP:
            if df_lmm.empty:
                continue
            match = df_lmm[(df_lmm["Region"] == i) & (df_lmm["Reference Session"] == ref) & (df_lmm["Effect"] == contrast)]
            if not match.empty:
                pvals_per_region[i, COLS.index(col)] = match["p-value"].values[0]
                estimates_per_region[i, COLS.index(col)] = match["Estimate"].values[0]

    return pd.DataFrame(pvals_per_region, columns=COLS), pd.DataFrame(estimates_per_region, columns=COLS)


def plot_pvalue_histograms(pvals_by_metric: dict, output_dir: Path):
    import matplotlib.pyplot as plt

    session_pair_labels = ["Ses-1 (control) vs Ses-2 (acute)", "Ses-1 (control) vs Ses-3 (chronic)", "Ses-2 (acute) vs Ses-3 (chronic)"]
    for metric_name, pvals_df in pvals_by_metric.items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for i, ax in enumerate(axes):
            ax.hist(pvals_df[COLS[i]].dropna(), bins=20)
            ax.set_title(f"{metric_name} - {session_pair_labels[i]}", fontsize=10)
            ax.set_xlabel("p-value")
            ax.set_ylabel("Number of regions")
        fig.tight_layout()
        fig.savefig(output_dir / f"{metric_name.lower()}_pvalue_histograms.png")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cfg.add_config_arg(parser)
    parser.add_argument("--data-root")
    parser.add_argument("--metrics", nargs="*", default=None, help=f"Nodal metrics to test (default: {NODAL_METRICS}).")
    parser.add_argument("--sessions", nargs="*", default=None)
    parser.add_argument("--n-regions", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/graph_metrics/nodal_metrics_LMM")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    matplotlib.use("Agg")
    warnings.filterwarnings("ignore")

    if args.test:
        tmp_dir = Path(tempfile.mkdtemp(prefix="nodal_lmm_test_"))
        n_regions = 5
        testing.write_synthetic_nodal_metrics(tmp_dir, n_regions=n_regions)
        data_root = str(tmp_dir)
        sessions = SESSION_ORDER
        metrics = ["Degree_centrality"]
        output_dir = tmp_dir / "out"
    else:
        config = cfg.load_config(args.config)
        data_root = args.data_root or cfg.get(config, "data_root")
        if not data_root:
            parser.error("--data-root is required (or set data_root in config.yaml).")
        sessions = args.sessions or SESSION_ORDER
        n_regions = args.n_regions or cfg.get(config, "n_regions", 89)
        metrics = args.metrics or NODAL_METRICS
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    pvals_by_metric = {}
    for metric in metrics:
        print(f"\nProcessing metric: {metric}")
        pvals_df, estimates_df = per_region_lmm(data_root, sessions, metric, n_regions)
        pvals_path = output_dir / f"{metric.lower()}_pvals_per_region_LMM.csv"
        est_path = output_dir / f"{metric.lower()}_estimates_per_region_LMM.csv"
        pvals_df.to_csv(pvals_path, index=False)
        estimates_df.to_csv(est_path, index=False)
        pvals_by_metric[metric] = pvals_df
        print(f"Saved: {pvals_path}")
        print(f"Saved: {est_path}")

    plot_pvalue_histograms(pvals_by_metric, output_dir)

    if args.test:
        assert pvals_by_metric, "Test fixture produced no per-region LMM results"
        for df in pvals_by_metric.values():
            assert len(df) == n_regions, "pvals dataframe has wrong number of regions"
        print("PASS")


if __name__ == "__main__":
    main()
