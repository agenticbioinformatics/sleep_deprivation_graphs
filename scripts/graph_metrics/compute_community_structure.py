#!/usr/bin/env python3
"""Community detection (greedy modularity) per subject/session, plus spatial
and graph-theoretic distances between detected communities.

For each subject/session adjacency matrix: detects communities, plots them
in 3 brain views (sagittal/axial/coronal), computes the Euclidean distance
between community centroids and the average shortest-path distance between
communities, and saves per-subject CSVs plus subjects-by-sessions summary
CSVs (modularity, community count, avg spatial/graph distance).
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
from networkx.algorithms import community
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.common import config as cfg
from scripts.common import io, testing
from scripts.common.constants import SESSION_ORDER


def compute_spatial_distances(communities, coord) -> np.ndarray:
    comm_centroids = [coord[list(comm)].mean(axis=0) for comm in communities]
    return cdist(comm_centroids, comm_centroids, metric="euclidean")


def compute_graph_distances(G, communities) -> np.ndarray:
    n = len(communities)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            paths = []
            for node_i in communities[i]:
                for node_j in communities[j]:
                    try:
                        paths.append(nx.shortest_path_length(G, source=node_i, target=node_j))
                    except nx.NetworkXNoPath:
                        continue
            avg_dist = np.mean(paths) if paths else np.nan
            dist_matrix[i, j] = dist_matrix[j, i] = avg_dist
    return dist_matrix


def run(data_root, sessions, coord, n_edges, output_dir, save_plots=True):
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    community_summary = []
    for ses in sessions:
        for sub_path in io.subject_dirs(data_root, ses):
            sub_id = os.path.basename(sub_path)
            for adj_file in io.find_adjacency_files(sub_path, suffix=f"cost_{n_edges}.txt"):
                try:
                    G = nx.from_numpy_array(io.load_adjacency_matrix(adj_file))
                    communities = community.greedy_modularity_communities(G)
                    modularity_value = community.modularity(G, communities)
                    num_communities = len(communities)

                    node_community_map = {node: idx for idx, comm in enumerate(communities) for node in comm}
                    out_dir = Path(adj_file).parent / "graph_metrics"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    adj_basename = Path(adj_file).stem

                    if save_plots:
                        cmap = cm.get_cmap("tab20", num_communities)
                        node_colors = [cmap(node_community_map[n]) for n in G.nodes()]
                        views = {
                            "sagittal": {k: (coord[k, 0], coord[k, 2]) for k in range(len(G.nodes))},
                            "axial": {k: (coord[k, 0], coord[k, 1]) for k in range(len(G.nodes))},
                            "coronal": {k: (coord[k, 1], coord[k, 2]) for k in range(len(G.nodes))},
                        }
                        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
                        for i, (view_name, pos) in enumerate(views.items()):
                            nx.draw(G, pos, node_color=node_colors, with_labels=False, node_size=100, ax=axs[i])
                            axs[i].set_title(f"{view_name.capitalize()} view")
                        fig.suptitle(f"{sub_id} | {ses} | Community Structure\n{adj_basename}", fontsize=16)
                        fig.tight_layout()
                        fig.savefig(out_dir / f"{adj_basename}_communities.png")
                        plt.close(fig)

                    spatial_distances = compute_spatial_distances(communities, coord)
                    graph_distances = compute_graph_distances(G, communities)
                    pd.DataFrame(spatial_distances).to_csv(out_dir / f"{sub_id}_{ses}_communities_spatial_distances.csv", index=False)
                    pd.DataFrame(graph_distances).to_csv(out_dir / f"{sub_id}_{ses}_communities_graph_distances.csv", index=False)

                    avg_spatial_dist = np.nanmean(spatial_distances[np.triu_indices(num_communities, k=1)]) if num_communities > 1 else np.nan
                    avg_graph_dist = np.nanmean(graph_distances[np.triu_indices(num_communities, k=1)]) if num_communities > 1 else np.nan
                    community_summary.append({
                        "subject_id": sub_id,
                        "session_id": ses,
                        "modularity": modularity_value,
                        "num_communities": num_communities,
                        "avg_spatial_distance": avg_spatial_dist,
                        "avg_graph_distance": avg_graph_dist,
                    })
                    print(f"{sub_id} {ses}: saved community metrics to {out_dir}")
                except Exception as e:
                    print(f"Error processing {adj_file}: {e}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(community_summary)
    written = {}
    if not summary_df.empty:
        for value_col in ["modularity", "num_communities", "avg_spatial_distance", "avg_graph_distance"]:
            pivoted = summary_df.pivot(index="subject_id", columns="session_id", values=value_col).reset_index()
            pivoted.columns.name = None
            out_path = output_dir / f"{value_col}_all_subjects.csv"
            pivoted.to_csv(out_path, index=False)
            written[value_col] = out_path
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cfg.add_config_arg(parser)
    parser.add_argument("--data-root")
    parser.add_argument("--coord-path")
    parser.add_argument("--sessions", nargs="*", default=None)
    parser.add_argument("--n-edges", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/graph_metrics")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        matplotlib.use("Agg")
        tmp_dir = Path(tempfile.mkdtemp(prefix="community_test_"))
        n_regions = 10
        testing.write_synthetic_adjacency_tree(tmp_dir, n_regions=n_regions, n_edges_list=(400,))
        data_root = str(tmp_dir)
        sessions = ["ses-1"]
        n_edges = 400
        coord = np.random.default_rng(0).random((n_regions, 3))
        output_dir = tmp_dir / "out"
    else:
        config = cfg.load_config(args.config)
        data_root = args.data_root or cfg.get(config, "data_root")
        coord_path = args.coord_path or cfg.get(config, "atlas.aal_coords")
        if not data_root or not coord_path:
            parser.error("--data-root and --coord-path are required (or set them in config.yaml).")
        sessions = args.sessions or SESSION_ORDER
        n_edges = args.n_edges or cfg.get(config, "n_edges_default", 400)
        coord = io.load_coordinates(coord_path)
        output_dir = Path(args.output_dir)
        matplotlib.use("Agg")

    written = run(data_root, sessions, coord, n_edges, output_dir, save_plots=not args.no_plot)

    if args.test:
        assert written, "Test fixture produced no community summary CSVs"
        print("PASS")


if __name__ == "__main__":
    main()
