#!/usr/bin/env python3
"""Framewise Displacement (FD) quality control.

Reproduces Power et al. (2012) Figure 2B: FD vs %BOLD-signal-change per ROI,
overlaid across all subjects in a session, on either raw or motion-regressed
timeseries. Consolidates 8 near-identical cells from the original
quality_check_FD_code.ipynb (one per session x timeseries-type combination,
plus 2 single-subject demo cells subsumed by --subjects) into this one
parameterized script.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.common import config as cfg
from scripts.common import testing
from scripts.common.constants import SESSION_ORDER

RADIUS_MM = 50  # Power et al. 2012 brain radius, converts rotations (rad) to mm


def compute_fd_from_rp(rp_file: str, radius: float = RADIUS_MM) -> np.ndarray:
    """Framewise displacement from an SPM rp_*.txt file (6 motion params:
    x, y, z, rot_x, rot_y, rot_z; rotations in radians)."""
    motion = np.loadtxt(rp_file)
    diffs = np.diff(motion, axis=0)
    diffs[:, 3:6] *= radius
    fd = np.sum(np.abs(diffs), axis=1)
    return np.insert(fd, 0, 0)


def load_timeseries_percent_bold(ts_file: str, raw_ts_file: str | None = None) -> np.ndarray:
    """%BOLD signal change per ROI. If raw_ts_file is given, ts_file is treated
    as motion-regressed and raw_ts_file supplies the baseline mean (regressed
    data can have near-zero means that make self-baselining meaningless)."""
    ts = np.loadtxt(ts_file)
    baseline_ts = np.loadtxt(raw_ts_file) if raw_ts_file else ts
    mean_signal = np.mean(baseline_ts, axis=0, keepdims=True)
    if np.any(np.abs(mean_signal) < 1e-3):
        print("Warning: some ROIs have near-zero mean in the baseline signal.", file=sys.stderr)
    return (ts - mean_signal) / mean_signal * 100


def compute_loess_curve(fd: np.ndarray, ts_percent: np.ndarray, frac: float = 0.02) -> np.ndarray:
    diff = np.abs(np.diff(ts_percent, axis=0)) * 10  # delta %BOLD x10, scaled for visibility
    fd = fd[1:]  # no difference computed for the first frame
    fd_expanded = np.repeat(fd, diff.shape[1])
    signal_change = diff.flatten()
    return lowess(signal_change, fd_expanded, frac=frac)


def qc_session(data_root, session, use_regressed, subjects=None, top_n=5, frac=0.02):
    """Runs FD-vs-%BOLD QC for every subject in a session, overlays LOESS
    curves on one figure, and returns (figure, per-subject FD summary df)."""
    import matplotlib.pyplot as plt

    if subjects is None:
        subject_paths = sorted(glob.glob(os.path.join(data_root, session, "sub-*")))
        subjects = [os.path.basename(p) for p in subject_paths]

    fig, ax = plt.subplots(figsize=(10, 6))
    subject_curves = {}
    fd_stats = {}

    for subject in subjects:
        rp_file = os.path.join(data_root, session, subject, "RS", "rp_RS_3x3x3_384dyn.txt")
        ts_raw_file = os.path.join(
            data_root, session, subject, "Timeseries", "data", "wAALours", "wAALours_ts_raw.txt"
        )
        ts_regressed_file = os.path.join(
            data_root, session, subject, "Timeseries", "data", "wAALours", "wAALours_ts.txt"
        )
        ts_file = ts_regressed_file if use_regressed else ts_raw_file
        required = [rp_file, ts_file] + ([ts_raw_file] if use_regressed else [])

        if not all(os.path.exists(p) for p in required):
            print(f"Skipping {subject}: missing file(s).")
            continue
        try:
            ts_percent = load_timeseries_percent_bold(ts_file, ts_raw_file if use_regressed else None)
            fd = compute_fd_from_rp(rp_file)
            loess_curve = compute_loess_curve(fd, ts_percent, frac=frac)
            subject_curves[subject] = loess_curve
            fd_stats[subject] = {"max_fd": float(np.max(fd)), "mean_fd": float(np.mean(fd))}
            ax.plot(loess_curve[:, 0], loess_curve[:, 1], alpha=0.6, linewidth=1)
        except Exception as e:
            print(f"Error processing {subject}: {e}")
            continue

    if not subject_curves:
        raise RuntimeError(
            f"No subjects with complete data found for {session} "
            f"({'regressed' if use_regressed else 'raw'} timeseries) under {data_root}."
        )

    top_n = min(top_n, len(subject_curves))
    sorted_by_bold = sorted(subject_curves.items(), key=lambda x: np.max(x[1][:, 1]), reverse=True)
    print("\nTop subjects with highest Delta%BOLD x10 values (possible outliers):")
    for i, (sub, curve) in enumerate(sorted_by_bold[:top_n]):
        max_y = np.max(curve[:, 1])
        print(f"{i + 1}. {sub}: max Delta%BOLD x10 = {max_y:.2f}")
        x_label, y_label = curve[np.argmax(curve[:, 1])]
        ax.text(x_label, y_label, sub, fontsize=8, color="red", alpha=0.9)

    sorted_by_fd = sorted(fd_stats.items(), key=lambda x: x[1]["max_fd"], reverse=True)
    print("\nTop subjects with highest max Framewise Displacement (FD):")
    for i, (sub, stats) in enumerate(sorted_by_fd[:top_n]):
        print(f"{i + 1}. {sub}: max FD = {stats['max_fd']:.4f}, mean FD = {stats['mean_fd']:.4f}")
        curve = subject_curves[sub]
        x_label, y_label = curve[np.argmax(curve[:, 0])]
        ax.text(x_label, y_label, sub, fontsize=8, color="blue", alpha=0.9)

    ts_kind = "timeseries after regression" if use_regressed else "raw timeseries"
    ax.set_xlabel("Framewise Displacement (FD) [mm]")
    ax.set_ylabel("Delta%BOLD x 10")
    ax.set_title(f"FD vs Delta%BOLD - {session}, {ts_kind}")
    ax.grid(True)
    fig.tight_layout()

    summary_df = pd.DataFrame(
        [
            {
                "subject_id": sub,
                "session": session,
                "timeseries": "regressed" if use_regressed else "raw",
                **stats,
            }
            for sub, stats in fd_stats.items()
        ]
    )
    return fig, summary_df


def _build_test_fixture(tmp_dir: Path) -> Path:
    for ses in SESSION_ORDER:
        for i, sub in enumerate(testing.DEFAULT_TEST_SUBJECTS):
            testing.write_synthetic_rp_file(
                tmp_dir / ses / sub / "RS" / "rp_RS_3x3x3_384dyn.txt", n_timepoints=30, seed=i
            )
            testing.write_synthetic_timeseries_file(
                tmp_dir / ses / sub / "Timeseries" / "data" / "wAALours" / "wAALours_ts_raw.txt",
                n_timepoints=30,
                seed=i,
            )
            testing.write_synthetic_timeseries_file(
                tmp_dir / ses / sub / "Timeseries" / "data" / "wAALours" / "wAALours_ts.txt",
                n_timepoints=30,
                seed=i + 100,
            )
    return tmp_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cfg.add_config_arg(parser)
    parser.add_argument("--data-root", help="Root of the derived-data tree (overrides config.yaml).")
    parser.add_argument("--session", choices=[*SESSION_ORDER, "all"], default="all")
    parser.add_argument("--timeseries-type", choices=["raw", "regressed", "both"], default="both")
    parser.add_argument("--subjects", nargs="*", default=None, help="Restrict to these subject IDs (default: all found).")
    parser.add_argument("--top-n", type=int, default=5, help="Number of outlier subjects to label per plot.")
    parser.add_argument("--output-dir", default="outputs/qc", help="Where to save plots and the FD summary CSV.")
    parser.add_argument("--show", action="store_true", help="Also display plots interactively (default: save only).")
    parser.add_argument("--test", action="store_true", help="Run on a tiny synthetic fixture instead of --data-root.")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frac = 0.02
    if args.test:
        tmp_dir = Path(tempfile.mkdtemp(prefix="fd_qc_test_"))
        data_root = str(_build_test_fixture(tmp_dir))
        sessions = ["ses-1"]
        ts_types = ["raw", "regressed"]
        output_dir = tmp_dir / "qc_out"
        top_n = 2
        frac = 0.5  # a small synthetic fixture needs a much wider LOESS window
    else:
        config = cfg.load_config(args.config)
        data_root = args.data_root or cfg.get(config, "data_root")
        if not data_root:
            parser.error("--data-root is required (or set data_root in config.yaml).")
        sessions = SESSION_ORDER if args.session == "all" else [args.session]
        ts_types = ["raw", "regressed"] if args.timeseries_type == "both" else [args.timeseries_type]
        output_dir = Path(args.output_dir)
        top_n = args.top_n

    output_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    for session in sessions:
        for ts_type in ts_types:
            use_regressed = ts_type == "regressed"
            fig, summary_df = qc_session(
                data_root, session, use_regressed, subjects=args.subjects, top_n=top_n, frac=frac
            )
            plot_path = output_dir / f"{session}_{ts_type}_fd_qc.png"
            fig.savefig(plot_path)
            if args.show:
                plt.show()
            plt.close(fig)
            print(f"Saved plot: {plot_path}")
            all_summaries.append(summary_df)

    combined = pd.concat(all_summaries, ignore_index=True)
    summary_path = output_dir / "fd_summary.csv"
    combined.to_csv(summary_path, index=False)
    print(f"Saved FD summary: {summary_path}")

    if args.test:
        assert summary_path.exists() and not combined.empty, "Test fixture produced no FD summary rows"
        print("PASS")


if __name__ == "__main__":
    main()
