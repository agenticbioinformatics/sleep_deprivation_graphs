#!/usr/bin/env python3
"""Within-subject Hub Disruption Index (HDI).

For each subject present in all 3 sessions, regresses the change in a nodal
metric (test session minus reference session) on the reference-session value
itself; the regression slope (kappa) summarizes whether high-baseline vs
low-baseline regions reorganize differently after sleep loss -- a
within-subject adaptation of the classic Hub Disruption Index. Computed for
all 4 directional pairs (acute-vs-control, chronic-vs-control,
chronic-vs-acute, acute-vs-chronic).

Reconstructed save step: the original notebook computed these per-subject
kappas (with a scatter+fit-line plot) but never actually wrote them to disk
-- the save cells were commented out and referenced an undefined variable,
even though 6+ downstream cells (permutation validation, CCML, trait
correlations) assumed kappas_<metric>.csv already existed. This script
assembles and saves it for real, so the rest of the pipeline is runnable.
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
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.common import config as cfg
from scripts.common import io, testing
from scripts.common.constants import HDI_CCML_METRICS, METRIC_FILE_SLUG


def _slope(model: LinearRegression, x: np.ndarray, y: np.ndarray) -> float:
    model.fit(x.reshape(-1, 1), y)
    return float(model.coef_[0])


def compute_kappas(data_root: str, metric_column: str, make_plot: bool = True):
    common_ids = io.common_subjects(data_root, ["ses-1", "ses-2", "ses-3"])
    print(f"Subjects present in all 3 sessions: {len(common_ids)}")
    if not common_ids:
        return pd.DataFrame(), None

    model = LinearRegression()
    rows = []
    fig = None
    if make_plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, len(common_ids), figsize=(4 * len(common_ids), 12), sharey=True, squeeze=False)

    for i, subject_id in enumerate(common_ids):
        control = io.load_nodal_metric_vector(os.path.join(data_root, "ses-1", subject_id), metric_column)
        acute = io.load_nodal_metric_vector(os.path.join(data_root, "ses-2", subject_id), metric_column)
        chronic = io.load_nodal_metric_vector(os.path.join(data_root, "ses-3", subject_id), metric_column)

        pairs = [
            ("kappa_acute_vs_control", control, acute - control, control),
            ("kappa_chronic_vs_control", control, chronic - control, control),
            ("kappa_chronic_vs_acute", acute, chronic - acute, acute),
            ("kappa_acute_vs_chronic", chronic, acute - chronic, chronic),
        ]
        kappas = {}
        for row_idx, (name, x, y, x_for_plot) in enumerate(pairs):
            kappas[name] = _slope(model, x, y)
            if make_plot:
                ax = axes[row_idx, i]
                ax.scatter(x_for_plot, y, alpha=0.7)
                ax.plot(x_for_plot, model.predict(x_for_plot.reshape(-1, 1)), color="red")
                title = f"{subject_id}\nκ={kappas[name]:.3f}" if row_idx == 0 else f"κ={kappas[name]:.3f}"
                ax.set_title(title)
                if i == 0:
                    ax.set_ylabel(f"Δ {metric_column} ({name.replace('kappa_', '').replace('_', ' ')})")

        rows.append({"subject_id": subject_id, **kappas})

    if make_plot:
        fig.suptitle(f"Within-subject HDI scatterplots using {metric_column}", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    return pd.DataFrame(rows), fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cfg.add_config_arg(parser)
    parser.add_argument("--data-root")
    parser.add_argument("--metric", choices=[*HDI_CCML_METRICS, "all"], default="all")
    parser.add_argument("--output-dir", default="outputs/hdi")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    matplotlib.use("Agg")

    if args.test:
        tmp_dir = Path(tempfile.mkdtemp(prefix="hdi_test_"))
        testing.write_synthetic_nodal_metrics(tmp_dir, n_regions=8)
        data_root = str(tmp_dir)
        metrics = ["Degree_centrality"]
        output_dir = tmp_dir / "out"
    else:
        config = cfg.load_config(args.config)
        data_root = args.data_root or cfg.get(config, "data_root")
        if not data_root:
            parser.error("--data-root is required (or set data_root in config.yaml).")
        metrics = HDI_CCML_METRICS if args.metric == "all" else [args.metric]
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for metric in metrics:
        df_kappa, fig = compute_kappas(data_root, metric, make_plot=not args.no_plot)
        if df_kappa.empty:
            print(f"No subjects with all 3 sessions available for {metric}; skipping.")
            continue
        out_path = output_dir / f"kappas_{METRIC_FILE_SLUG[metric]}.csv"
        df_kappa.to_csv(out_path, index=False)
        written.append(out_path)
        print(f"Saved: {out_path}")
        if fig is not None:
            plot_path = output_dir / f"kappas_{METRIC_FILE_SLUG[metric]}_scatter.png"
            fig.savefig(plot_path)
            import matplotlib.pyplot as plt

            plt.close(fig)
            print(f"Saved: {plot_path}")

    if args.test:
        assert written, "Test fixture produced no kappa CSVs"
        df = pd.read_csv(written[0])
        for col in ["kappa_acute_vs_control", "kappa_chronic_vs_control", "kappa_chronic_vs_acute", "kappa_acute_vs_chronic"]:
            assert col in df.columns, f"Missing expected column {col}"
        print("PASS")


if __name__ == "__main__":
    main()
