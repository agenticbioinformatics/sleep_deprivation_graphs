#!/usr/bin/env python3
"""Multiple-comparisons correction (Bonferroni and FDR/LSU) of the per-region
nodal-metric p-values written by compare_nodal_metrics_lmm.py.

Writes, per metric, a Bonferroni-reject and an FDR-reject boolean CSV (one
row per region, one column per session comparison) consumed by
plot_significant_nodes.py.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from multipy.fdr import lsu
from multipy.fwer import bonferroni

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.common import config as cfg
from scripts.common.constants import NODAL_METRICS, NODAL_SESSION_COMPARISONS


def correct_pvalues(pvals_dir: Path, metrics: list[str], alpha: float, fdr_q: float) -> dict:
    """Returns {metric: {"bonferroni": df, "fdr": df}}, each df indexed by
    region with one boolean column per session comparison."""
    results = {}
    for metric in metrics:
        path = pvals_dir / f"{metric.lower()}_pvals_per_region_LMM.csv"
        if not path.exists():
            print(f"Skipping {metric}: {path} not found.")
            continue
        pvals_df = pd.read_csv(path)
        bonf_df = pd.DataFrame(index=pvals_df.index)
        fdr_df = pd.DataFrame(index=pvals_df.index)
        for comparison in NODAL_SESSION_COMPARISONS:
            pvals_session = pvals_df[comparison].fillna(1.0).values
            bonf_df[comparison] = bonferroni(pvals_session, alpha=alpha)
            fdr_df[comparison] = lsu(pvals_session, q=fdr_q)
        results[metric] = {"bonferroni": bonf_df, "fdr": fdr_df}
        print(f"{metric}: Bonferroni rejects = {int(bonf_df.values.sum())}, FDR rejects = {int(fdr_df.values.sum())}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cfg.add_config_arg(parser)
    parser.add_argument("--pvals-dir", default="outputs/graph_metrics/nodal_metrics_LMM")
    parser.add_argument("--metrics", nargs="*", default=None)
    parser.add_argument("--alpha", type=float, default=0.05, help="Bonferroni significance level.")
    parser.add_argument("--fdr-q", type=float, default=0.1, help="FDR (LSU / Benjamini-Hochberg) q-value.")
    parser.add_argument("--output-dir", default="outputs/graph_metrics/nodal_metrics_LMM")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        tmp_dir = Path(tempfile.mkdtemp(prefix="correct_pvals_test_"))
        rng = np.random.default_rng(0)
        pvals_df = pd.DataFrame(rng.uniform(0, 1, size=(5, 3)), columns=NODAL_SESSION_COMPARISONS)
        pvals_dir = tmp_dir
        pvals_df.to_csv(tmp_dir / "degree_centrality_pvals_per_region_LMM.csv", index=False)
        metrics = ["Degree_centrality"]
        output_dir = tmp_dir
    else:
        pvals_dir = Path(args.pvals_dir)
        metrics = args.metrics or NODAL_METRICS
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    results = correct_pvalues(pvals_dir, metrics, args.alpha, args.fdr_q)

    written = []
    for metric, dfs in results.items():
        for kind, df in dfs.items():
            out_path = output_dir / f"{metric.lower()}_{kind}_reject_LMM.csv"
            df.to_csv(out_path, index=False)
            written.append(out_path)
            print(f"Saved: {out_path}")

    if args.test:
        assert written, "Test fixture produced no correction results"
        print("PASS")


if __name__ == "__main__":
    main()
