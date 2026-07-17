"""Synthetic-data builders used by every script's --test flag.

None of these touch real data -- they generate a tiny, fast, in-memory-sized
fixture on disk (under a temp dir) so a script's real code path can be
smoke-tested without a real data_root.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.common.constants import ROI_NETWORK_MAP

DEFAULT_TEST_SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-04"]
DEFAULT_TEST_SESSIONS = ["ses-1", "ses-2", "ses-3"]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def write_synthetic_adjacency_tree(
    root: str | Path,
    sessions: list[str] = DEFAULT_TEST_SESSIONS,
    subjects: list[str] = DEFAULT_TEST_SUBJECTS,
    n_regions: int = 10,
    n_edges_list: tuple[int, ...] = (400, 200),
    seed: int = 0,
) -> Path:
    """Writes {root}/{ses}/{sub}/Graphs/wAALours/Adj_mat_synthetic_cost_{n}.txt
    for each n in n_edges_list, as small random symmetric binary matrices.
    """
    root = Path(root)
    rng = _rng(seed)
    for ses in sessions:
        for sub in subjects:
            graph_dir = root / ses / sub / "Graphs" / "wAALours"
            graph_dir.mkdir(parents=True, exist_ok=True)
            for n_edges in n_edges_list:
                mat = (rng.random((n_regions, n_regions)) < 0.3).astype(int)
                mat = np.triu(mat, k=1)
                mat = mat + mat.T
                np.savetxt(graph_dir / f"Adj_mat_synthetic_cost_{n_edges}.txt", mat, fmt="%d")
    return root


def write_synthetic_nodal_metrics(
    root: str | Path,
    sessions: list[str] = DEFAULT_TEST_SESSIONS,
    subjects: list[str] = DEFAULT_TEST_SUBJECTS,
    n_regions: int = 10,
    seed: int = 0,
) -> Path:
    """Writes {root}/{ses}/{sub}/Graphs/wAALours/graph_metrics/{ses}_{sub}_metrics.csv."""
    root = Path(root)
    rng = _rng(seed)
    for ses in sessions:
        for sub in subjects:
            metrics_dir = root / ses / sub / "Graphs" / "wAALours" / "graph_metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame({
                "Node": np.arange(n_regions),
                "Closeness": rng.random(n_regions),
                "Betweenness": rng.random(n_regions),
                "Clustering": rng.random(n_regions),
                "Degree_centrality": rng.random(n_regions),
            })
            df.to_csv(metrics_dir / f"{ses}_{sub}_metrics.csv", index=False)
    return root


def write_synthetic_global_metric_csv(
    path: str | Path,
    subjects: list[str] = DEFAULT_TEST_SUBJECTS,
    sessions: list[str] = DEFAULT_TEST_SESSIONS,
    seed: int = 0,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = _rng(seed)
    df = pd.DataFrame({"subject_id": subjects})
    for ses in sessions:
        df[ses] = rng.random(len(subjects))
    df.to_csv(path, index=False)
    return path


def write_synthetic_rp_file(path: str | Path, n_timepoints: int = 20, seed: int = 0) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = _rng(seed)
    motion = rng.normal(scale=0.01, size=(n_timepoints, 6))
    np.savetxt(path, motion)
    return path


def write_synthetic_timeseries_file(
    path: str | Path, n_timepoints: int = 20, n_regions: int = 10, seed: int = 0
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = _rng(seed)
    ts = rng.normal(loc=100.0, scale=5.0, size=(n_timepoints, n_regions))
    np.savetxt(path, ts)
    return path


def write_synthetic_kappa_csv(
    path: str | Path, subjects: list[str] = DEFAULT_TEST_SUBJECTS, seed: int = 0
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = _rng(seed)
    df = pd.DataFrame({
        "subject_id": subjects,
        "kappa_acute_vs_control": rng.normal(size=len(subjects)),
        "kappa_chronic_vs_control": rng.normal(size=len(subjects)),
        "kappa_chronic_vs_acute": rng.normal(size=len(subjects)),
        "kappa_acute_vs_chronic": rng.normal(size=len(subjects)),
    })
    df.to_csv(path, index=False)
    return path


def write_synthetic_survey_csv(
    path: str | Path,
    subjects: list[str] = DEFAULT_TEST_SUBJECTS,
    kind: str = "state",
    seed: int = 0,
) -> Path:
    """kind='state' -> KSS sleepiness columns; kind='trait' -> AM/ME/PSQI trait columns."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = _rng(seed)
    df = pd.DataFrame({"Subject_ID": subjects})
    if kind == "state":
        for col in ["KSS-B1", "KSS-A1", "KSS-C1"]:
            df[col] = rng.uniform(1, 9, size=len(subjects))
    else:
        for col in ["AM", "ME", "PSQI"]:
            df[col] = rng.normal(size=len(subjects))
    df.to_csv(path, sep=";", index=False)
    return path


def write_synthetic_roi_table(path: str | Path, n_regions: int = 10) -> Path:
    """Uses real ROI names (subset of ROI_NETWORK_MAP) so network-lookup code
    under test exercises real lookups instead of falling back to 'Other'.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(ROI_NETWORK_MAP.keys())[:n_regions]
    df = pd.DataFrame({"Region": names, "Node_number": range(len(names))})
    df.to_csv(path, sep=";", index=False)
    return path
