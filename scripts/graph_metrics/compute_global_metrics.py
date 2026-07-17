#!/usr/bin/env python3
"""Global graph metrics: global efficiency, average clustering, average path
length, and modularity, one CSV per metric (subjects x sessions).

Merges 4 near-identical cells from the original notebook (one per metric,
differing only in which NetworkX function was called) into a single script
parameterized by --metric.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import networkx as nx
import pandas as pd
from networkx.algorithms import community

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.common import config as cfg
from scripts.common import io, testing
from scripts.common.constants import GLOBAL_METRIC_FUNCS, SESSION_ORDER


def _average_path_length(G: nx.Graph) -> float:
    """Falls back to the largest connected component if G is disconnected."""
    if nx.is_connected(G):
        return nx.average_shortest_path_length(G)
    largest_cc = max(nx.connected_components(G), key=len)
    return nx.average_shortest_path_length(G.subgraph(largest_cc))


def _modularity(G: nx.Graph) -> float:
    communities = community.greedy_modularity_communities(G)
    return community.modularity(G, communities)


METRIC_FUNCS = {
    "global_efficiency": nx.global_efficiency,
    "average_clustering": nx.average_clustering,
    "average_path_length": _average_path_length,
    "modularity": _modularity,
}


def compute_global_metric(data_root, sessions, metric_name, n_edges) -> pd.DataFrame:
    metric_func = METRIC_FUNCS[metric_name]
    results = []
    for ses in sessions:
        for sub_path in io.subject_dirs(data_root, ses):
            sub_id = os.path.basename(sub_path)
            for adj_file in io.find_adjacency_files(sub_path, suffix=f"cost_{n_edges}.txt"):
                try:
                    G = nx.from_numpy_array(io.load_adjacency_matrix(adj_file))
                    value = metric_func(G)
                    results.append({"subject_id": sub_id, "session_id": ses, metric_name: value})
                    print(f"{sub_id} | {ses} | {value:.4f}")
                except Exception as e:
                    print(f"Error processing {adj_file}: {e}")

    df_out = pd.DataFrame(results)
    if df_out.empty:
        return df_out
    pivoted = df_out.pivot(index="subject_id", columns="session_id", values=metric_name).reset_index()
    pivoted.columns.name = None
    return pivoted


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cfg.add_config_arg(parser)
    parser.add_argument("--data-root", help="Root of the derived-data tree (overrides config.yaml).")
    parser.add_argument("--metric", choices=[*GLOBAL_METRIC_FUNCS, "all"], default="all")
    parser.add_argument("--sessions", nargs="*", default=None)
    parser.add_argument("--n-edges", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/graph_metrics")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        tmp_dir = Path(tempfile.mkdtemp(prefix="global_metrics_test_"))
        testing.write_synthetic_adjacency_tree(tmp_dir, n_regions=10, n_edges_list=(400,))
        data_root = str(tmp_dir)
        sessions = ["ses-1"]
        n_edges = 400
        output_dir = tmp_dir / "out"
        metrics = ["global_efficiency"]
    else:
        config = cfg.load_config(args.config)
        data_root = args.data_root or cfg.get(config, "data_root")
        if not data_root:
            parser.error("--data-root is required (or set data_root in config.yaml).")
        sessions = args.sessions or SESSION_ORDER
        n_edges = args.n_edges or cfg.get(config, "n_edges_default", 400)
        output_dir = Path(args.output_dir)
        metrics = GLOBAL_METRIC_FUNCS if args.metric == "all" else [args.metric]

    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for metric_name in metrics:
        pivoted = compute_global_metric(data_root, sessions, metric_name, n_edges)
        out_path = output_dir / f"{metric_name}_all_subjects.csv"
        pivoted.to_csv(out_path, index=False)
        written.append(out_path)
        print(f"Saved: {out_path}")

    if args.test:
        df = pd.read_csv(written[0])
        assert not df.empty, "Test fixture produced an empty global metrics CSV"
        print("PASS")


if __name__ == "__main__":
    main()
