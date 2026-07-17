#!/usr/bin/env python3
"""Identify and plot FDR-significant nodes (regions) from the per-region LMM
+ correction results, in brain-coordinate space.

For each metric x session-comparison: selects regions flagged significant by
correct_nodal_pvalues.py's FDR correction, saves a significant-nodes CSV
(region name, MNI-ish coordinates, LMM coefficient, direction), and plots
them in 3 brain views. Pass --glass-brain to additionally render a
netplotbrain glass-brain figure per comparison (heavier optional dependency).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile
from pathlib import Path

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.common import config as cfg
from scripts.common import io, testing
from scripts.common.constants import NODAL_METRICS, NODAL_SESSION_COMPARISON_LABELS, NODAL_SESSION_COMPARISONS


def load_reference_graph(data_root, session, subject, n_edges) -> nx.Graph | None:
    """A single subject's adjacency matrix, used only to draw background
    connectivity edges under the significant-node markers. Optional -- nodes
    still plot correctly without it."""
    if not data_root:
        return None
    subject_paths = [os.path.join(data_root, session, subject)] if subject else io.subject_dirs(data_root, session)
    for sub_path in subject_paths:
        files = io.find_adjacency_files(sub_path, suffix=f"cost_{n_edges}.txt")
        if files:
            return nx.from_numpy_array(io.load_adjacency_matrix(files[0]))
    return None


def find_significant_nodes(coef_df, fdr_reject_df, comparison, roi_lookup, coord):
    fdr_significant = fdr_reject_df[comparison].astype(bool).to_numpy()
    coef_session = coef_df[comparison].to_numpy()
    node_colors = np.zeros_like(coef_session)
    node_colors[fdr_significant & (coef_session > 0)] = 1
    node_colors[fdr_significant & (coef_session < 0)] = -1

    rows = []
    for idx, (is_sig, coef) in enumerate(zip(fdr_significant, coef_session)):
        if is_sig:
            roi_name = roi_lookup.get(idx, f"ROI_{idx}")
            rows.append({
                "Node": idx,
                "Region": roi_name,
                "X": round(coord[idx, 0], 1),
                "Y": round(coord[idx, 1], 1),
                "Z": round(coord[idx, 2], 1),
                "Coefficient": round(float(coef), 4),
                "Direction": "increase" if coef > 0 else "decrease",
            })
    return node_colors, pd.DataFrame(rows)


def plot_brain_views(G, coord, node_colors, title, out_path):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n_regions = len(node_colors)
    color_map = {-1: "#0080FF", 0: "#A0A0A0", 1: "#FF8000"}
    node_color_list = [color_map[val] for val in node_colors]

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    views = {
        "Sagittal view (Y-Z)": {k: (coord[k, 1], coord[k, 2]) for k in range(n_regions)},
        "Axial view (X-Y)": {k: (coord[k, 1], coord[k, 0]) for k in range(n_regions)},
        "Coronal view (X-Z)": {k: (coord[k, 0], coord[k, 2]) for k in range(n_regions)},
    }
    graph = G if G is not None else nx.empty_graph(n_regions)
    for ax, (view_title, pos) in zip(axs, views.items()):
        nx.draw(graph, pos, node_color=node_color_list, edge_color="0.8", with_labels=False, node_size=100, ax=ax)
        ax.set_title(view_title)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Significant increase", markerfacecolor="#FF8000", markersize=10),
        Line2D([0], [0], marker="o", color="w", label="Significant decrease", markerfacecolor="#0080FF", markersize=10),
        Line2D([0], [0], marker="o", color="w", label="Not significant", markerfacecolor="#A0A0A0", markersize=10),
    ]
    axs[2].legend(handles=legend_elements, loc="upper right", fontsize=10)
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_glass_brain(sig_nodes_dir: Path, output_dir: Path):
    try:
        from netplotbrain import plot as netplotbrain_plot
    except ImportError:
        print("netplotbrain is not installed; skipping --glass-brain plots.")
        return

    direction_color_map = {"increase": "#FF8000", "decrease": "#0080FF"}
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(glob.glob(str(sig_nodes_dir / "*_sig_nodes.csv"))):
        df = pd.read_csv(path)
        if df.empty:
            continue
        basename = os.path.basename(path).replace("_sig_nodes.csv", "")
        df_plot = df.copy()
        df_plot["x"], df_plot["y"], df_plot["z"] = df_plot["X"], df_plot["Y"], df_plot["Z"]
        df_plot["label"] = df_plot["Region"]
        df_plot["color"] = df_plot["Direction"].map(direction_color_map)
        fig, _ = netplotbrain_plot(nodes=df_plot, node_color="color", node_size=6)
        fig.savefig(output_dir / f"{basename}_glass_brain.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cfg.add_config_arg(parser)
    parser.add_argument("--data-root", help="Optional -- only used to draw background connectivity edges.")
    parser.add_argument("--roi-table", help="ROI region-list CSV (overrides config.yaml atlas.aal_region_list).")
    parser.add_argument("--coord-path", help="Atlas coordinates file (overrides config.yaml atlas.aal_coords).")
    parser.add_argument("--pvals-dir", default="outputs/graph_metrics/nodal_metrics_LMM", help="Directory with *_estimates_per_region_LMM.csv and *_fdr_reject_LMM.csv.")
    parser.add_argument("--metrics", nargs="*", default=None)
    parser.add_argument("--reference-session", default="ses-1")
    parser.add_argument("--reference-subject", default=None)
    parser.add_argument("--n-edges", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/graph_metrics/significant_nodes")
    parser.add_argument("--glass-brain", action="store_true", help="Also render netplotbrain glass-brain figures (extra dependency).")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    matplotlib.use("Agg")

    if args.test:
        tmp_dir = Path(tempfile.mkdtemp(prefix="sig_nodes_test_"))
        n_regions = 5
        roi_table_path = testing.write_synthetic_roi_table(tmp_dir / "roi_table.csv", n_regions=n_regions)
        coord = np.random.default_rng(0).random((n_regions, 3)) * 50
        pvals_dir = tmp_dir
        rng = np.random.default_rng(1)
        estimates = pd.DataFrame(rng.normal(size=(n_regions, 3)), columns=NODAL_SESSION_COMPARISONS)
        estimates.to_csv(tmp_dir / "degree_centrality_estimates_per_region_LMM.csv", index=False)
        fdr_reject = pd.DataFrame(rng.random((n_regions, 3)) < 0.5, columns=NODAL_SESSION_COMPARISONS)
        fdr_reject.to_csv(tmp_dir / "degree_centrality_fdr_reject_LMM.csv", index=False)
        metrics = ["Degree_centrality"]
        output_dir = tmp_dir / "out"
        data_root = None
        n_edges = 400
        roi_df = io.load_roi_table(roi_table_path)
    else:
        config = cfg.load_config(args.config)
        roi_table_path = args.roi_table or cfg.get(config, "atlas.aal_region_list")
        coord_path = args.coord_path or cfg.get(config, "atlas.aal_coords")
        if not roi_table_path or not coord_path:
            parser.error("--roi-table and --coord-path are required (or set atlas.aal_region_list / atlas.aal_coords in config.yaml).")
        roi_df = io.load_roi_table(roi_table_path)
        coord = io.load_coordinates(coord_path)
        pvals_dir = Path(args.pvals_dir)
        metrics = args.metrics or NODAL_METRICS
        output_dir = Path(args.output_dir)
        data_root = args.data_root or cfg.get(config, "data_root")
        n_edges = args.n_edges or cfg.get(config, "n_edges_default", 400)

    roi_lookup = dict(zip(roi_df["Node_number"], roi_df["Region"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    G = load_reference_graph(data_root, args.reference_session, args.reference_subject, n_edges)

    written = []
    for metric in metrics:
        coef_path = pvals_dir / f"{metric.lower()}_estimates_per_region_LMM.csv"
        fdr_path = pvals_dir / f"{metric.lower()}_fdr_reject_LMM.csv"
        if not coef_path.exists() or not fdr_path.exists():
            print(f"Skipping {metric}: missing {coef_path} or {fdr_path}.")
            continue
        coef_df = pd.read_csv(coef_path)
        fdr_reject_df = pd.read_csv(fdr_path)

        for comparison in NODAL_SESSION_COMPARISONS:
            node_colors, sig_df = find_significant_nodes(coef_df, fdr_reject_df, comparison, roi_lookup, coord)
            csv_path = output_dir / f"{metric}_{comparison}_sig_nodes.csv"
            sig_df.to_csv(csv_path, index=False)
            written.append(csv_path)
            print(f"{metric} - {comparison}: {len(sig_df)} significant nodes -> {csv_path}")

            title = f"{metric} - {NODAL_SESSION_COMPARISON_LABELS[comparison]}"
            plot_path = output_dir / f"{metric}_{comparison}_brain_views.png"
            plot_brain_views(G, coord, node_colors, title, plot_path)

    if args.glass_brain:
        plot_glass_brain(output_dir, output_dir / "glass_brain")

    if args.test:
        assert written, "Test fixture produced no significant-node CSVs"
        print("PASS")


if __name__ == "__main__":
    main()
