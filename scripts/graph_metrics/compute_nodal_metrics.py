#!/usr/bin/env python3
"""Nodal graph metrics (closeness, betweenness, clustering, degree centrality).

For every subject/session adjacency matrix under data_root, computes nodal
graph metrics with NetworkX and writes a per-subject-session CSV plus a
network layout plot next to the adjacency matrix
(Graphs/wAALours/graph_metrics/), so later pipeline stages can find them.

Note: the source notebook saved the degree-centrality column as "Degree",
while every downstream cell (HDI, CCML) read a column called
"Degree_centrality" -- a latent bug that would KeyError on a clean run. This
script writes "Degree_centrality" consistently.
"""

from __future__ import annotations

import argparse
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
from scripts.common.constants import SESSION_ORDER


def compute_and_save_nodal_metrics(data_root, sessions, coord, n_edges, save_plot=True) -> list[str]:
    import matplotlib.pyplot as plt

    written = []
    for ses in sessions:
        for sub_path in io.subject_dirs(data_root, ses):
            sub = os.path.basename(sub_path)
            adj_files = io.find_adjacency_files(sub_path, suffix=f"_{n_edges}.txt")
            for adj_file in adj_files:
                try:
                    X = io.load_adjacency_matrix(adj_file)
                    G = nx.from_numpy_array(X)
                    pos = {k: (coord[k, 1], coord[k, 2]) for k in range(len(G.nodes))}

                    metrics_dir = Path(adj_file).parent / "graph_metrics"
                    metrics_dir.mkdir(parents=True, exist_ok=True)
                    file_id = f"{ses}_{sub}_{Path(adj_file).stem}"

                    if save_plot:
                        fig, ax = plt.subplots(figsize=(8, 8))
                        nx.draw(G, pos, node_size=50, with_labels=False, ax=ax)
                        fig.savefig(metrics_dir / f"{file_id}.png")
                        plt.close(fig)

                    closeness = nx.closeness_centrality(G)
                    betweenness = nx.betweenness_centrality(G)
                    clustering = nx.clustering(G)
                    degree = nx.degree_centrality(G)

                    metrics_df = pd.DataFrame({
                        "Node": list(closeness.keys()),
                        "Closeness": list(closeness.values()),
                        "Betweenness": list(betweenness.values()),
                        "Clustering": list(clustering.values()),
                        "Degree_centrality": list(degree.values()),
                    })
                    out_path = metrics_dir / f"{file_id}_metrics.csv"
                    metrics_df.to_csv(out_path, index=False)
                    written.append(str(out_path))
                    print(f"Processed: {file_id}")
                except Exception as e:
                    print(f"Error processing {adj_file}: {e}")
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cfg.add_config_arg(parser)
    parser.add_argument("--data-root", help="Root of the derived-data tree (overrides config.yaml).")
    parser.add_argument("--coord-path", help="Atlas coordinates file (whitespace-separated, overrides config.yaml atlas.aal_coords).")
    parser.add_argument("--sessions", nargs="*", default=None, help="Sessions to process (default: all).")
    parser.add_argument("--n-edges", type=int, default=None, help="Adjacency matrix edge-count suffix to match, e.g. 400 for *_400.txt (default: config n_edges_default).")
    parser.add_argument("--no-plot", action="store_true", help="Skip saving the network layout PNG (metrics CSV is always saved).")
    parser.add_argument("--test", action="store_true", help="Run on a tiny synthetic fixture instead of --data-root.")
    args = parser.parse_args()

    if args.test:
        matplotlib.use("Agg")
        tmp_dir = Path(tempfile.mkdtemp(prefix="nodal_metrics_test_"))
        n_regions = 10
        testing.write_synthetic_adjacency_tree(tmp_dir, n_regions=n_regions, n_edges_list=(400,))
        data_root = str(tmp_dir)
        sessions = ["ses-1"]
        n_edges = 400
        coord = np.random.default_rng(0).random((n_regions, 3))
    else:
        config = cfg.load_config(args.config)
        data_root = args.data_root or cfg.get(config, "data_root")
        coord_path = args.coord_path or cfg.get(config, "atlas.aal_coords")
        if not data_root or not coord_path:
            parser.error("--data-root and --coord-path are required (or set data_root / atlas.aal_coords in config.yaml).")
        sessions = args.sessions or SESSION_ORDER
        n_edges = args.n_edges or cfg.get(config, "n_edges_default", 400)
        coord = io.load_coordinates(coord_path)
        matplotlib.use("Agg")

    written = compute_and_save_nodal_metrics(data_root, sessions, coord, n_edges, save_plot=not args.no_plot)

    if args.test:
        assert written, "Test fixture produced no nodal metrics CSVs"
        df = pd.read_csv(written[0])
        assert "Degree_centrality" in df.columns, "Expected 'Degree_centrality' column missing"
        print("PASS")
    elif not written:
        print("Warning: no adjacency matrices found -- nothing was computed.", file=sys.stderr)


if __name__ == "__main__":
    main()
