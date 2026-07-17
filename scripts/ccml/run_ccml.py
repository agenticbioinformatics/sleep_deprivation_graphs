#!/usr/bin/env python3
"""Covariate-Constrained Manifold Learning (CCML).

Embeds each subject's nodal-metric vector into 2D using an Isomap-derived
distance matrix, constrained so that one embedding axis tracks a covariate
(the subject's within-subject HDI kappa, from compute_hdi.py) via a
multi-start optimization. Then tests whether the acute and chronic groups
occupy different regions of the embedding via a permutation test on
centroid distance.

Renamed for clarity vs. the source notebook: the original used "control" to
mean the *acute* (ses-2) group and "patient" for the *chronic* (ses-3) group
-- both compared against the ses-1 baseline via their respective HDI kappa.
This script calls them acute_subjects / chronic_subjects.

Note: the original notebook only implemented the embedding + permutation
test for 'Degree_centrality' and 'Closeness'; 'Clustering' only had the
data-prep half done. This script completes clustering by mirroring the same
pipeline used for the other two metrics (per explicit user confirmation).
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
import scipy.optimize as opt
from scipy.interpolate import griddata
from sklearn.manifold import Isomap
from sklearn.metrics import pairwise_distances

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.common import config as cfg
from scripts.common import io, testing
from scripts.common.constants import HDI_CCML_METRICS, METRIC_FILE_SLUG
from scripts.ccml import plot_utils


def build_ccml_inputs(data_root, kappa_dir, metric, output_dir):
    kappa_path = Path(kappa_dir) / f"kappas_{METRIC_FILE_SLUG[metric]}.csv"
    kappa_df = pd.read_csv(kappa_path).set_index("subject_id")

    def extract_metric(subject_paths):
        data, ids = [], []
        for subj_path in subject_paths:
            metrics_file = io.find_nodal_metrics_file(subj_path)
            if metrics_file is None:
                continue
            df = pd.read_csv(metrics_file)
            if metric not in df.columns:
                continue
            data.append(df[metric].values)
            ids.append(os.path.basename(subj_path))
        return np.array(data), ids

    acute_data, acute_ids = extract_metric(io.subject_dirs(data_root, "ses-2"))
    chronic_data, chronic_ids = extract_metric(io.subject_dirs(data_root, "ses-3"))
    if len(acute_ids) == 0 or len(chronic_ids) == 0:
        raise RuntimeError(f"No subjects with a '{metric}' nodal-metrics CSV found in ses-2 and/or ses-3 under {data_root}.")

    acute_kappa = kappa_df.loc[acute_ids, "kappa_acute_vs_control"].values
    chronic_kappa = kappa_df.loc[chronic_ids, "kappa_chronic_vs_control"].values

    cov = np.concatenate([acute_kappa, chronic_kappa])
    labels = np.array([0] * len(acute_ids) + [1] * len(chronic_ids))
    subject_names = [f"A{i}" for i in range(len(acute_ids))] + [f"C{i}" for i in range(len(chronic_ids))]
    X = np.vstack([acute_data, chronic_data])

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(X, index=subject_names).to_csv(output_dir / f"{metric}_chronic_and_acute_vs_control.csv")
    pd.DataFrame(cov, index=subject_names, columns=["HDI_kappa"]).to_csv(output_dir / f"HDI_kappa_{metric}_chronic_and_acute_vs_control.csv")
    pd.DataFrame(labels, index=subject_names, columns=["Group"]).to_csv(output_dir / f"group_labels_{metric}_chronic_and_acute_vs_control.csv")
    print(f"CCML input files for {metric} saved in: {output_dir}")

    return X, cov, labels, subject_names


def _f_coord1(x, X2, cov2, alpha2, dist2):
    X2 = np.vstack([X2, x])
    cov2 = cov2.reshape(X2.shape)
    X_tmp2 = np.hstack((alpha2 * cov2, X2))
    return np.linalg.norm(dist2 - pairwise_distances(X_tmp2))


def _f_glob(X2, cov2, alpha2, dist2):
    cov2 = cov2.reshape(X2.shape)
    X_tmp2 = np.vstack((alpha2 * cov2, X2)).T
    return np.linalg.norm(dist2 - pairwise_distances(X_tmp2))


def _f_alpha(alpha, X2, cov2, dist2):
    cov2 = cov2.reshape(X2.shape)
    X_tmp = np.vstack((alpha * cov2, X2))
    return np.linalg.norm(dist2 - pairwise_distances(X_tmp.T))


def fit_ccml_embedding(X, cov, n_neighbors=4, verbose=False):
    D = pairwise_distances(X)
    ll = [np.where(D == np.max(D))[0][0], np.where(D == np.max(D))[1][0]]
    while len(ll) != D.shape[0]:
        D_tmp = D[ll]
        D_tmp[:, ll] = 0
        ll.append(np.where(D_tmp == np.max(D_tmp))[1][0])
    list_extract = np.array(ll)

    iso = Isomap(n_components=2, n_neighbors=n_neighbors)
    iso.fit(X)
    tmp = iso.embedding_[:, 0]
    dist = iso.dist_matrix_
    delta_1 = np.max(tmp) - np.min(tmp)
    delta_2 = np.max(cov) - np.min(cov)
    alpha = delta_1 / delta_2 if delta_2 != 0 else 1.0

    X_tmp = np.zeros([1])
    disp = 1 if verbose else 0
    for i in range(1, X.shape[0]):
        cov_tmp = cov[list_extract[0:i + 1]]
        dist_tmp = dist[list_extract[0:i + 1]][:, list_extract[0:i + 1]]
        best_score, xtmp = 10000, np.array([0.0])
        for j in range(-5, 5):
            xtmptmp, score, *_ = opt.fmin(
                _f_coord1, j, (X_tmp, cov_tmp, alpha, dist_tmp),
                xtol=1e-9, ftol=1e-9, maxiter=1_000_000, maxfun=1_000_000, disp=0, full_output=1,
            )
            if best_score > score:
                xtmp, best_score = xtmptmp, score
        X_tmp = np.vstack([X_tmp, xtmp])
        alpha, _, *_ = opt.fmin(_f_alpha, alpha, (X_tmp, cov_tmp, dist_tmp), xtol=1e-9, ftol=1e-8, disp=disp, full_output=1)
        X_tmp2, _, *_ = opt.fmin(_f_glob, X_tmp, (cov_tmp, alpha, dist_tmp), xtol=1e-12, ftol=1e-11, maxiter=1_000_000, maxfun=1_000_000, disp=disp, full_output=1)
        X_tmp = X_tmp2.reshape(-1, 1)

    cov2 = cov.reshape(X_tmp.shape)
    X_iso = np.hstack((alpha * cov2[list_extract], X_tmp))
    order = np.argsort(list_extract)
    return X_iso[order]


def centroid_distance_permutation_test(X_iso, labels, n_permutations, rng):
    group0, group1 = X_iso[labels == 0], X_iso[labels == 1]
    d_obs = np.linalg.norm(group0.mean(axis=0) - group1.mean(axis=0))
    d_perm = np.zeros(n_permutations)
    for i in range(n_permutations):
        perm_labels = rng.permutation(labels)
        p0, p1 = X_iso[perm_labels == 0], X_iso[perm_labels == 1]
        d_perm[i] = np.linalg.norm(p0.mean(axis=0) - p1.mean(axis=0))
    p_value = np.mean(d_perm >= d_obs)
    return d_obs, d_perm, p_value


def plot_embedding(X_iso, cov, labels, subject_names, metric, output_dir, grid_points=1000, scaling=5):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 8))
    grid_x, grid_y = np.mgrid[-1:1:complex(0, grid_points), -1:1:complex(0, grid_points)] * scaling
    try:
        grid_lin = griddata(X_iso, cov, (grid_x, grid_y), method="linear").reshape(grid_points, grid_points)
        im = ax.imshow(grid_lin.T, extent=(-scaling, scaling, -scaling, scaling), origin="lower", alpha=0.6)
        fig.colorbar(im, ax=ax)
    except Exception as e:
        print(f"Skipping covariate heatmap overlay: {e}")

    plot_utils.scatter_2d(X_iso, labels, subject_names, ax=ax, label_map={0: "Acute", 1: "Chronic"}, title=f"{metric}: TSD and CSR compared to RW")
    out_path = output_dir / f"{METRIC_FILE_SLUG[metric]}_ccml_embedding.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_permutation_histogram(d_obs, d_perm, p_value, metric, output_dir):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(d_perm, bins=50, edgecolor="black", alpha=0.7, color="grey")
    ax.axvline(d_obs, color="red", linestyle="--", linewidth=2, label=f"Observed = {d_obs:.3f}\np = {p_value:.4f}")
    ax.set_title(f"CCML permutation test - centroid distance ({metric})")
    ax.set_xlabel("Centroid distance")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    out_path = output_dir / f"{METRIC_FILE_SLUG[metric]}_ccml_permutation.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cfg.add_config_arg(parser)
    parser.add_argument("--data-root")
    parser.add_argument("--kappa-dir", default="outputs/hdi", help="Directory with kappas_<metric>.csv from compute_hdi.py.")
    parser.add_argument("--metric", choices=[*HDI_CCML_METRICS, "all"], default="all")
    parser.add_argument("--n-neighbors", type=int, default=4, help="Isomap n_neighbors.")
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--grid-points", type=int, default=1000, help="Resolution of the covariate heatmap overlay.")
    parser.add_argument("--scaling", type=float, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-prep-only", action="store_true", help="Only build the CCML input CSVs; skip the embedding + permutation test.")
    parser.add_argument("--output-dir", default="outputs/ccml")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    matplotlib.use("Agg")
    rng = np.random.default_rng(args.seed)

    if args.test:
        tmp_dir = Path(tempfile.mkdtemp(prefix="ccml_test_"))
        testing.write_synthetic_nodal_metrics(tmp_dir, sessions=["ses-2", "ses-3"], subjects=[f"sub-{i:02d}" for i in range(6)], n_regions=8)
        subjects = [f"sub-{i:02d}" for i in range(6)]
        testing.write_synthetic_kappa_csv(tmp_dir / "hdi" / "kappas_degree.csv", subjects=subjects)
        data_root = str(tmp_dir)
        kappa_dir = tmp_dir / "hdi"
        metrics = ["Degree_centrality"]
        n_neighbors = 3
        n_permutations = 20
        grid_points = 30
        output_dir = tmp_dir / "out"
        data_prep_only = False
    else:
        config = cfg.load_config(args.config)
        data_root = args.data_root or cfg.get(config, "data_root")
        if not data_root:
            parser.error("--data-root is required (or set data_root in config.yaml).")
        kappa_dir = Path(args.kappa_dir)
        metrics = HDI_CCML_METRICS if args.metric == "all" else [args.metric]
        n_neighbors = args.n_neighbors
        n_permutations = args.n_permutations
        grid_points = args.grid_points
        output_dir = Path(args.output_dir)
        data_prep_only = args.data_prep_only

    inputs_dir = output_dir / "inputs"
    written = []
    for metric in metrics:
        X, cov, labels, subject_names = build_ccml_inputs(data_root, kappa_dir, metric, inputs_dir)
        written.append(metric)

        if data_prep_only:
            continue

        X_iso = fit_ccml_embedding(X, cov, n_neighbors=n_neighbors)
        plot_path = plot_embedding(X_iso, cov, labels, subject_names, metric, output_dir, grid_points=grid_points, scaling=args.scaling)
        print(f"Saved: {plot_path}")

        d_obs, d_perm, p_value = centroid_distance_permutation_test(X_iso, labels, n_permutations, rng)
        print(f"{metric}: observed centroid distance = {d_obs:.4f}, permutation p-value = {p_value:.4f}")
        perm_plot_path = plot_permutation_histogram(d_obs, d_perm, p_value, metric, output_dir)
        print(f"Saved: {perm_plot_path}")

        result_path = output_dir / f"{METRIC_FILE_SLUG[metric]}_ccml_centroid_test.csv"
        pd.DataFrame([{"metric": metric, "observed_centroid_distance": d_obs, "p_value": p_value, "n_permutations": n_permutations}]).to_csv(result_path, index=False)
        print(f"Saved: {result_path}")

    if args.test:
        assert written, "Test fixture produced no CCML inputs"
        print("PASS")


if __name__ == "__main__":
    main()
