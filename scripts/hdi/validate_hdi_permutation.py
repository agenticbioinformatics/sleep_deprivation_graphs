#!/usr/bin/env python3
"""Permutation validation of the within-subject Hub Disruption Index (HDI).

Consolidates ~8 near-duplicate exploratory permutation cells from the
original notebook (each hand-tuned to one metric/session-pair combination)
into 4 distinct, reusable null models, each generalized across
--metric/--session-pair:

  group-mean-diff          kappa_within (per-subject regression slope) minus
                            kappa_groupmean (slope against the group-average
                            baseline); null shuffles node order of the test
                            session's vector.
  sign-flip-within-pairs   per-subject regression slope of delta vs baseline;
                            null randomly flips the sign of delta per subject
                            (breaks the delta's direction, keeps its
                            magnitude and the baseline pairing).
  subject-shuffle           mean kappa; null permutes which subject's test
                            vector is paired with which subject's baseline
                            vector (breaks subject correspondence).
  node-shuffle-within-pairs single kappa; null shuffles node order of the
                            test vector within the same subject (breaks
                            node-to-node correspondence, keeps subject
                            pairing).

The original notebook used 3 different, inconsistent p-value formulas across
these cells (one-tailed in one, two-tailed with different centering in
others, no p-value at all in another). This script applies one consistent
two-tailed permutation p-value everywhere: the fraction of null draws at
least as extreme as the real statistic, relative to the null distribution's
own mean, with a +1 continuity correction.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.common import config as cfg
from scripts.common import io, testing
from scripts.common.constants import HDI_CCML_METRICS, HDI_SESSION_PAIR_LABELS, HDI_SESSION_PAIRS, METRIC_FILE_SLUG
from scripts.common.stats import cohens_d, fdr_correct


def _slope(model: LinearRegression, x: np.ndarray, y: np.ndarray) -> float:
    model.fit(x.reshape(-1, 1), y)
    return float(model.coef_[0])


def load_paired_vectors(data_root, metric_column, test_ses, ref_ses):
    common_ids = io.common_subjects(data_root, [ref_ses, test_ses])
    ref_all = np.array([io.load_nodal_metric_vector(os.path.join(data_root, ref_ses, sid), metric_column) for sid in common_ids])
    test_all = np.array([io.load_nodal_metric_vector(os.path.join(data_root, test_ses, sid), metric_column) for sid in common_ids])
    return common_ids, ref_all, test_all


def null_model_group_mean_diff(ref_all, test_all, n_iterations, rng):
    model = LinearRegression()
    group_ref_mean = np.mean(ref_all, axis=0)
    n_subjects = len(ref_all)

    def kappa_diff(x_ref, x_test):
        kappa_within = _slope(model, x_ref, x_test - x_ref)
        kappa_group = _slope(model, group_ref_mean, x_test - group_ref_mean)
        return kappa_within - kappa_group

    real = np.mean([kappa_diff(ref_all[i], test_all[i]) for i in range(n_subjects)])
    null_dist = np.array([
        np.mean([kappa_diff(ref_all[i], rng.permutation(test_all[i])) for i in range(n_subjects)])
        for _ in range(n_iterations)
    ])
    return real, null_dist


def null_model_sign_flip(ref_all, test_all, n_iterations, rng):
    model = LinearRegression()
    n_subjects = len(ref_all)
    real = np.mean([_slope(model, ref_all[i], test_all[i] - ref_all[i]) for i in range(n_subjects)])

    null_dist = np.zeros(n_iterations)
    for it in range(n_iterations):
        kappas = []
        for i in range(n_subjects):
            delta = test_all[i] - ref_all[i]
            if rng.random() < 0.5:
                delta = -delta
            kappas.append(_slope(model, ref_all[i], delta))
        null_dist[it] = np.mean(kappas)
    return real, null_dist


def null_model_subject_shuffle(ref_all, test_all, n_iterations, rng):
    model = LinearRegression()
    n_subjects = len(ref_all)
    real = np.mean([_slope(model, ref_all[i], test_all[i] - ref_all[i]) for i in range(n_subjects)])

    null_dist = np.zeros(n_iterations)
    for it in range(n_iterations):
        perm = rng.permutation(n_subjects)
        kappas = [_slope(model, ref_all[i], test_all[perm[i]] - ref_all[i]) for i in range(n_subjects)]
        null_dist[it] = np.mean(kappas)
    return real, null_dist


def null_model_node_shuffle_within_pairs(ref_all, test_all, n_iterations, rng):
    model = LinearRegression()
    n_subjects = len(ref_all)
    real = np.mean([_slope(model, ref_all[i], test_all[i] - ref_all[i]) for i in range(n_subjects)])

    null_dist = np.zeros(n_iterations)
    for it in range(n_iterations):
        kappas = [_slope(model, ref_all[i], rng.permutation(test_all[i]) - ref_all[i]) for i in range(n_subjects)]
        null_dist[it] = np.mean(kappas)
    return real, null_dist


NULL_MODELS = {
    "group-mean-diff": null_model_group_mean_diff,
    "sign-flip-within-pairs": null_model_sign_flip,
    "subject-shuffle": null_model_subject_shuffle,
    "node-shuffle-within-pairs": null_model_node_shuffle_within_pairs,
}


def two_tailed_p_value(real_stat: float, null_dist: np.ndarray) -> float:
    null_mean = np.mean(null_dist)
    n_extreme = np.sum(np.abs(null_dist - null_mean) >= np.abs(real_stat - null_mean))
    return (n_extreme + 1) / (len(null_dist) + 1)


def plot_null_distribution(real_stat, null_dist, title, out_path, overlay_real_kappas=None, overlay_label=None):
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(10, 5))
    if overlay_real_kappas is not None and len(overlay_real_kappas) > 0:
        ax1.hist(overlay_real_kappas, bins=min(12, len(overlay_real_kappas)), alpha=0.8, color="darkorange", edgecolor="black")
        ax1.set_ylabel(f"Frequency ({overlay_label})", color="darkorange")
        ax2 = ax1.twinx()
    else:
        ax2 = ax1
    ax2.hist(null_dist, bins=30, alpha=0.6, color="grey", edgecolor="black")
    ax2.axvline(real_stat, color="red", linestyle="dashed", linewidth=2, label=f"Real = {real_stat:.3f}")
    ax2.set_xlabel("kappa (or kappa difference)")
    ax2.set_ylabel("Frequency (permutations)", color="grey")
    ax2.legend(loc="upper left")
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cfg.add_config_arg(parser)
    parser.add_argument("--data-root")
    parser.add_argument("--null-model", choices=list(NULL_MODELS), default="sign-flip-within-pairs")
    parser.add_argument("--metric", choices=[*HDI_CCML_METRICS, "all"], default="all")
    parser.add_argument("--session-pair", choices=[*HDI_SESSION_PAIRS, "all"], default="all")
    parser.add_argument("--n-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kappa-dir", default=None, help="Directory with kappas_<metric>.csv from compute_hdi.py, to overlay real participant kappas on the plot.")
    parser.add_argument("--overlay-real-kappas", action="store_true")
    parser.add_argument("--output-dir", default="outputs/hdi/permutation_validation")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    matplotlib.use("Agg")
    rng = np.random.default_rng(args.seed)

    if args.test:
        tmp_dir = Path(tempfile.mkdtemp(prefix="hdi_perm_test_"))
        testing.write_synthetic_nodal_metrics(tmp_dir, n_regions=8)
        data_root = str(tmp_dir)
        metrics = ["Degree_centrality"]
        session_pairs = ["acute_vs_control"]
        n_iterations = 30
        output_dir = tmp_dir / "out"
    else:
        config = cfg.load_config(args.config)
        data_root = args.data_root or cfg.get(config, "data_root")
        if not data_root:
            parser.error("--data-root is required (or set data_root in config.yaml).")
        metrics = HDI_CCML_METRICS if args.metric == "all" else [args.metric]
        session_pairs = list(HDI_SESSION_PAIRS) if args.session_pair == "all" else [args.session_pair]
        n_iterations = args.n_iterations
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    null_func = NULL_MODELS[args.null_model]
    rows = []

    for session_pair in session_pairs:
        test_ses, ref_ses = HDI_SESSION_PAIRS[session_pair]
        pvals_this_pair = []
        real_stats_this_pair = {}
        null_dists_this_pair = {}

        for metric in metrics:
            common_ids, ref_all, test_all = load_paired_vectors(data_root, metric, test_ses, ref_ses)
            if len(common_ids) < 2:
                print(f"Skipping {metric}/{session_pair}: fewer than 2 subjects with both sessions.")
                continue

            real_stat, null_dist = null_func(ref_all, test_all, n_iterations, rng)
            p_value = two_tailed_p_value(real_stat, null_dist)
            d = cohens_d(real_stat, null_dist)
            print(f"{metric} | {session_pair} | {args.null_model}: real = {real_stat:.4f}, p = {p_value:.4f}, Cohen's d = {d:.2f}")

            overlay = None
            if args.overlay_real_kappas and args.kappa_dir:
                kappa_path = Path(args.kappa_dir) / f"kappas_{METRIC_FILE_SLUG[metric]}.csv"
                col = f"kappa_{session_pair}"
                if kappa_path.exists():
                    kdf = pd.read_csv(kappa_path)
                    if col in kdf.columns:
                        overlay = kdf[col].dropna().values

            title = f"{metric} - {HDI_SESSION_PAIR_LABELS[session_pair]}\n{args.null_model} permutation"
            plot_path = output_dir / f"{METRIC_FILE_SLUG[metric]}_{session_pair}_{args.null_model}.png"
            plot_null_distribution(real_stat, null_dist, title, plot_path, overlay_real_kappas=overlay, overlay_label="real participants")

            pvals_this_pair.append(p_value)
            real_stats_this_pair[metric] = real_stat
            null_dists_this_pair[metric] = null_dist
            rows.append({"metric": metric, "session_pair": session_pair, "null_model": args.null_model, "real_stat": real_stat, "p_raw": p_value, "cohens_d": d})

        if len(pvals_this_pair) > 1:
            rejected, p_fdr = fdr_correct(pvals_this_pair)
            for row, p, sig in zip(rows[-len(pvals_this_pair):], p_fdr, rejected):
                row["p_FDR"] = p
                row["significant_FDR"] = bool(sig)

    results_df = pd.DataFrame(rows)
    out_path = output_dir / f"{args.null_model}_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    if args.test:
        assert not results_df.empty, "Test fixture produced no permutation validation results"
        print("PASS")


if __name__ == "__main__":
    main()
