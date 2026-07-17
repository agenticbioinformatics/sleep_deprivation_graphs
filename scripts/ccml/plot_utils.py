"""Reconstructed replacement for the missing `PlotFunction` module.

The original notebook imported CCML visualization code from a local module
(`PlotFunction.scatter_2D`) via a hardcoded sys.path into a course-tutorial
folder that isn't part of this repo and doesn't exist anywhere in it. This
module reimplements the one function CCML actually calls: a 2D scatter of
the embedding, colored by group and labeled by subject name. It is a
reconstruction based on how the function is called, not the original file.
"""

from __future__ import annotations

import numpy as np


def scatter_2d(X, labels, individual_names, ax=None, label_map=None, title=None):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    labels = np.asarray(labels)
    unique_labels = sorted(set(labels.tolist()))
    cmap = plt.get_cmap("tab10")

    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        name = label_map.get(lbl, str(lbl)) if label_map else str(lbl)
        ax.scatter(X[mask, 0], X[mask, 1], color=cmap(i), label=name, s=60, alpha=0.85, zorder=3)

    for (x, y), name in zip(X, individual_names):
        ax.annotate(name, (x, y), fontsize=8, alpha=0.8, xytext=(3, 3), textcoords="offset points", zorder=4)

    ax.set_xlabel("Component 1 (covariate-weighted)")
    ax.set_ylabel("Component 2")
    if title:
        ax.set_title(title)
    ax.legend()
    return fig, ax
