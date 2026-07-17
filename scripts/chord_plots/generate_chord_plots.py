#!/usr/bin/env python3
"""Per-subject, per-session chord diagrams of functional connectivity.

Uses a sparser adjacency matrix (200 edges by default, vs. 400 for the
graph-metric stages -- the denser version was too visually dense as a chord
diagram) and colors/sizes nodes by their functional network membership.

Fixed vs. the source notebook: the ROI-to-network map had a typo
('Occipial_R' instead of 'Occipital_R') that silently dropped that node into
the 'Other' bucket instead of the visual network (VN) -- see
scripts/common/constants.py.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.common import config as cfg
from scripts.common import io, testing
from scripts.common.constants import ROI_NETWORK_MAP, SESSION_ORDER

DEFAULT_SESSION_FOLDER_SUFFIX = "_AICHA_atlas"


def find_all_participants(data_root, session_folders, n_edges):
    participants = set()
    for folder in session_folders:
        ses_path = os.path.join(data_root, folder)
        for root, _dirs, files in os.walk(ses_path):
            for f in files:
                if f.startswith("Adj_mat") and f.endswith(f"{n_edges}.txt"):
                    for part in root.split(os.sep):
                        if part.startswith("sub-"):
                            participants.add(part)
    return sorted(participants)


def get_matrix(data_root, sub_id, session_folder, n_regions, n_edges):
    path = os.path.join(data_root, session_folder, sub_id, "Graphs", "wAALours")
    if not os.path.isdir(path):
        return None
    files = [f for f in os.listdir(path) if f.startswith("Adj_mat") and f.endswith(f"{n_edges}.txt")]
    if not files:
        return None
    matrix = np.loadtxt(os.path.join(path, files[0]))
    if matrix.shape != (n_regions, n_regions):
        return None
    return matrix.astype(int)


def build_ordered_regions(roi_df: pd.DataFrame):
    roi_df = roi_df.sort_values("Node_number")
    original_region_names = roi_df["Region"].tolist()
    nodes_df = pd.DataFrame({
        "name": original_region_names,
        "network": [ROI_NETWORK_MAP.get(roi, "Other") for roi in original_region_names],
    })
    nodes_df = nodes_df.sort_values(by=["network", "name"]).reset_index(drop=True)
    return original_region_names, nodes_df


def create_chord(matrix, original_region_names, nodes_df, title):
    import holoviews as hv
    from holoviews import opts

    region_names = nodes_df["name"].tolist()
    reordered_indices = [original_region_names.index(name) for name in region_names]
    matrix = matrix[np.ix_(reordered_indices, reordered_indices)]

    edges = []
    connected_nodes = set()
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[1]):
            if matrix[i, j] == 1:
                source, target = region_names[i], region_names[j]
                edges.append((source, target, 1, ROI_NETWORK_MAP.get(source, "Other")))
                connected_nodes.update([source, target])
    for name in region_names:
        if name not in connected_nodes:
            edges.append((name, name, 0.01, ROI_NETWORK_MAP.get(name, "Other")))

    edges_df = pd.DataFrame(edges, columns=["source", "target", "value", "network_src"])
    connection_counts = edges_df[["source", "target"]].stack().value_counts()
    nodes_with_size = nodes_df.copy()
    nodes_with_size["connections"] = nodes_with_size["name"].map(connection_counts).fillna(0)
    nodes_with_size["size"] = nodes_with_size["connections"] * 4

    chord = hv.Chord((edges_df, hv.Dataset(nodes_with_size, kdims="name")))
    return chord.opts(
        opts.Chord(
            labels="name", node_color="network", edge_color="network_src",
            cmap="Set1", edge_cmap="Set1", edge_alpha=0.8, node_size="size",
            width=650, height=650, title=title, colorbar=False, tools=["hover"],
        )
    )


def _build_test_fixture(tmp_dir: Path, n_regions=10):
    subjects = testing.DEFAULT_TEST_SUBJECTS[:2]
    session_folders = [f"{ses}{DEFAULT_SESSION_FOLDER_SUFFIX}" for ses in SESSION_ORDER]
    testing.write_synthetic_adjacency_tree(tmp_dir, sessions=session_folders, subjects=subjects, n_regions=n_regions, n_edges_list=(200,))
    roi_table_path = testing.write_synthetic_roi_table(tmp_dir / "roi_table.csv", n_regions=n_regions)
    return session_folders, roi_table_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cfg.add_config_arg(parser)
    parser.add_argument("--data-root")
    parser.add_argument("--roi-table", help="Overrides config.yaml atlas.aal_region_list.")
    parser.add_argument("--session-folder-suffix", default=DEFAULT_SESSION_FOLDER_SUFFIX, help="Session subfolders are named '<session><suffix>', e.g. 'ses-1_AICHA_atlas'.")
    parser.add_argument("--n-edges", type=int, default=None, help="Overrides config.yaml n_edges_chord.")
    parser.add_argument("--n-regions", type=int, default=None, help="Overrides config.yaml n_regions.")
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--output-dir", default="outputs/chord_plots")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    import holoviews as hv

    hv.extension("bokeh")

    if args.test:
        tmp_dir = Path(tempfile.mkdtemp(prefix="chord_test_"))
        n_regions = 10
        session_folders, roi_table_path = _build_test_fixture(tmp_dir, n_regions=n_regions)
        data_root = str(tmp_dir)
        roi_df = io.load_roi_table(roi_table_path)
        n_edges = 200
        output_dir = tmp_dir / "out"
        subjects = None
    else:
        config = cfg.load_config(args.config)
        data_root = args.data_root or cfg.get(config, "data_root")
        roi_table_path = args.roi_table or cfg.get(config, "atlas.aal_region_list")
        if not data_root or not roi_table_path:
            parser.error("--data-root and --roi-table are required (or set them in config.yaml).")
        roi_df = io.load_roi_table(roi_table_path)
        n_edges = args.n_edges or cfg.get(config, "n_edges_chord", 200)
        n_regions = args.n_regions or cfg.get(config, "n_regions", 89)
        session_folders = [f"{ses}{args.session_folder_suffix}" for ses in SESSION_ORDER]
        output_dir = Path(args.output_dir)
        subjects = args.subjects

    original_region_names, nodes_df = build_ordered_regions(roi_df)
    output_dir.mkdir(parents=True, exist_ok=True)

    participants = subjects or find_all_participants(data_root, session_folders, n_edges)
    print(f"Found {len(participants)} participants")

    written = []
    for sub_id in participants:
        chords = []
        for ses, folder in zip(SESSION_ORDER, session_folders):
            mat = get_matrix(data_root, sub_id, folder, n_regions, n_edges)
            if mat is not None:
                chords.append(create_chord(mat, original_region_names, nodes_df, title=f"{ses} - {sub_id}"))
            else:
                print(f"Missing or invalid matrix for {sub_id} in {folder}")

        if chords:
            layout = hv.Layout(chords).cols(3)
            filename = output_dir / f"{sub_id}_chord_diagram_all_sessions_{n_edges}_sized_colored.html"
            hv.save(layout, filename, backend="bokeh")
            written.append(filename)
            print(f"Saved {filename}")
        else:
            print(f"No valid matrices for {sub_id}, skipping.")

    print("All participant-level multi-session chord diagrams saved.")

    if args.test:
        assert written, "Test fixture produced no chord diagram files"
        print("PASS")


if __name__ == "__main__":
    main()
