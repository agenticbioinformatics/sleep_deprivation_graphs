"""Shared I/O helpers for reading adjacency matrices, coordinates, and ROI tables.

All of these expect the derived-data tree layout produced by the external
rs_graph_processing pipeline: {data_root}/{session}/{subject}/Graphs/wAALours/...
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd


def load_adjacency_matrix(path: str | Path) -> np.ndarray:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    return df.to_numpy()


def load_coordinates(path: str | Path) -> np.ndarray:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    return df.to_numpy()


def load_roi_table(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df.columns = df.columns.str.strip()
    return df


def subject_dirs(data_root: str | Path, session: str) -> list[str]:
    """Sorted list of sub-* directory paths for one session."""
    return sorted(glob.glob(os.path.join(data_root, session, "sub-*")))


def common_subjects(data_root: str | Path, sessions: list[str]) -> list[str]:
    """Subject IDs (basenames) present in every given session."""
    id_sets = []
    for ses in sessions:
        ids = {os.path.basename(p) for p in subject_dirs(data_root, ses)}
        id_sets.append(ids)
    if not id_sets:
        return []
    common = set.intersection(*id_sets)
    return sorted(common)


def find_adjacency_files(subject_path: str | Path, suffix: str = "_400.txt") -> list[str]:
    graph_dir = os.path.join(subject_path, "Graphs", "wAALours")
    return sorted(glob.glob(os.path.join(graph_dir, f"*{suffix}")))


def find_nodal_metrics_file(subject_path: str | Path) -> str | None:
    pattern = os.path.join(subject_path, "Graphs", "wAALours", "graph_metrics", "*_metrics.csv")
    files = sorted(glob.glob(pattern))
    return files[0] if files else None


def load_nodal_metric_vector(subject_path: str | Path, column: str) -> np.ndarray:
    metrics_file = find_nodal_metrics_file(subject_path)
    if metrics_file is None:
        raise FileNotFoundError(f"No nodal metrics CSV found under {subject_path}")
    df = pd.read_csv(metrics_file)
    return df[column].values
