#!/usr/bin/env python3
"""Permutation-test robustness checks for the global-metric session
comparisons, as a sanity check against the LMM results.

Two null models, selected via --method:
  unpaired -- pool all sessions' values and resample with replacement into
              3 new "sessions" (breaks both subject pairing and session
              identity).
  paired   -- within each subject pair, randomly flip which value is
              labeled session A vs session B (keeps pairing, breaks the
              session-label / value association).

Both report real paired t-statistics with FDR-corrected p-values against
the permutation null distribution, plotted as 1 row per metric x 1 column
per session-pair comparison.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.utils import resample

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.common import config as cfg
from scripts.common import testing
from scripts.common.constants import ALL_GLOBAL_METRICS
from scripts.common.stats import fdr_correct

SESSION_PAIRS = [("ses-1", "ses-2"), ("ses-1", "ses-3"), ("ses-2", "ses-3")]
COMPARISON_LABELS = ["Acute vs Control", "Chronic vs Control", "Chronic vs Acute"]


def run_unpaired(df_metric: pd.DataFrame, n_iterations: int):
    n_subjects = len(df_metric)
    pooled = np.hstack([df_metric["ses-1"], df_metric["ses-2"], df_metric["ses-3"]])
    perm_stats = np.zeros((n_iterations, 3))
    for i in range(n_iterations):
        new_sample = resample(pooled)
        new_ses = [new_sample[k * n_subjects:(k + 1) * n_subjects] for k in range(3)]
        perm_stats[i, 0], _ = ttest_rel(new_ses[0], new_ses[1])
        perm_stats[i, 1], _ = ttest_rel(new_ses[0], new_ses[2])
        perm_stats[i, 2], _ = ttest_rel(new_ses[1], new_ses[2])

    real_stats, real_pvals = [], []
    for a, b in SESSION_PAIRS:
        t, p = ttest_rel(df_metric[a], df_metric[b])
        real_stats.append(t)
        real_pvals.append(p)
    return perm_stats, real_stats, real_pvals


def run_paired(df_metric: pd.DataFrame, n_iterations: int, rng: np.random.Generator):
    n_subjects = len(df_metric)
    session_data = {ses: df_metric[ses].values for ses in ["ses-1", "ses-2", "ses-3"]}
    perm_stats = np.zeros((n_iterations, 3))
    real_stats, real_pvals = [], []

    for idx, (ses_a, ses_b) in enumerate(SESSION_PAIRS):
        x, y = session_data[ses_a], session_data[ses_b]
        t_real, p_real = ttest_rel(x, y)
        real_stats.append(t_real)
        real_pvals.append(p_real)
        for i in range(n_iterations):
            flip = rng.choice([True, False], size=n_subjects)
            x_perm = np.where(flip, x, y)
            y_perm = np.where(flip, y, x)
            perm_stats[i, idx], _ = ttest_rel(x_perm, y_perm)
    return perm_stats, real_stats, real_pvals


def plot_and_report(metric_name, method, perm_stats, real_stats, real_pvals, output_dir):
    import matplotlib.pyplot as plt

    rejected, real_pvals_fdr = fdr_correct(real_pvals)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for j, (ax, label) in enumerate(zip(axes, COMPARISON_LABELS)):
        ax.hist(perm_stats[:, j], bins=30, alpha=0.7)
        ax.axvline(real_stats[j], color="red", linestyle="dashed", linewidth=2,
                    label=f"Real t-stat\nt = {real_stats[j]:.2f}\np (FDR) = {real_pvals_fdr[j]:.4f}")
        ax.set_title(f"{metric_name}\n{label}\n({method} permutation)", fontsize=13)
        ax.set_xlabel("t-statistic")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / f"{metric_name}_{method}_permutation.png")
    plt.close(fig)

    rows = []
    for label, t_val, p_val, p_fdr, sig in zip(COMPARISON_LABELS, real_stats, real_pvals, real_pvals_fdr, rejected):
        rows.append({"metric": metric_name, "comparison": label, "t": t_val, "p_raw": p_val, "p_FDR": p_fdr, "significant_FDR": bool(sig)})
        print(f"{metric_name} | {label}: t = {t_val:.3f}, p = {p_val:.4f}, FDR p = {p_fdr:.4f} [{'SIGNIFICANT' if sig else 'NON-SIGNIFICANT'}]")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cfg.add_config_arg(parser)
    parser.add_argument("--metrics-dir", default="outputs/graph_metrics")
    parser.add_argument("--metrics", nargs="*", default=None)
    parser.add_argument("--method", choices=["unpaired", "paired"], default="paired")
    parser.add_argument("--n-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/graph_metrics/permutation_tests")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    matplotlib.use("Agg")
    rng = np.random.default_rng(args.seed)

    if args.test:
        tmp_dir = Path(tempfile.mkdtemp(prefix="perm_test_"))
        metrics_dir = tmp_dir
        subjects = [f"sub-{i:02d}" for i in range(6)]
        testing.write_synthetic_global_metric_csv(tmp_dir / "global_efficiency_all_subjects.csv", subjects=subjects, seed=1)
        metrics = ["global_efficiency"]
        n_iterations = 50
        output_dir = tmp_dir
    else:
        metrics_dir = Path(args.metrics_dir)
        metrics = args.metrics or ALL_GLOBAL_METRICS
        n_iterations = args.n_iterations
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for metric_name in metrics:
        path = metrics_dir / f"{metric_name}_all_subjects.csv"
        if not path.exists():
            print(f"Skipping {metric_name}: {path} not found.")
            continue
        df_metric = pd.read_csv(path)
        print(f"\n=== {metric_name} ({args.method}) ===")
        if args.method == "unpaired":
            perm_stats, real_stats, real_pvals = run_unpaired(df_metric, n_iterations)
        else:
            perm_stats, real_stats, real_pvals = run_paired(df_metric, n_iterations, rng)
        all_rows.extend(plot_and_report(metric_name, args.method, perm_stats, real_stats, real_pvals, output_dir))

    results_df = pd.DataFrame(all_rows)
    out_path = output_dir / f"{args.method}_permutation_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    if args.test:
        assert not results_df.empty, "Test fixture produced no permutation results"
        print("PASS")


if __name__ == "__main__":
    main()
