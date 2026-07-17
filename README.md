# Sleep deprivation and graphs

## Overview

This repository accompanies **"Divergent disruption of brain networks
following total and chronic sleep loss: a longitudinal fMRI study"**
(Patrycja Scislewska, Arturo Cabrera Vazquez, Iwona Szatkowska, Halszka
Kontrymowicz-Ogińska, Sophie Achard, Aleksandra Domagalik; *bioRxiv*,
2025.10.10.681651, [doi.org/10.1101/2025.10.10.681651](https://doi.org/10.1101/2025.10.10.681651)).
In this within-subject resting-state fMRI study, 28 healthy adults were
scanned under three conditions — rested wakefulness (RW), after one night of
acute total sleep deprivation (TSD), and after five nights of chronic sleep
restriction (CSR) — to examine how sleep loss alters intrinsic functional
brain organization. The pipeline here computes nodal/global graph-theoretical
metrics, a novel within-subject Hub Disruption Index (HDI), Covariate-
Constrained Manifold Learning (CCML) embeddings, connectivity chord plots,
and relates all of the above to subjective sleepiness, sleep quality, and
circadian trait measures.

## Installation

CPU-only; no GPU-specific setup is needed anywhere in this pipeline.

```bash
pip install -r requirements.txt
```

## Usage

**Before running anything**, set `data_root` in `config.yaml` (or pass
`--data-root` to every script) — see the note at the top of the **Data**
section below on what that directory needs to contain. Every script also
accepts `--config config.yaml` for shared defaults and a `--test` flag that
runs the same code path against a small synthetic fixture instead of real
data, so you can confirm a script works before pointing it at real data.

Run scripts as modules from the repo root, e.g.:

```bash
python -m scripts.qc.compute_framewise_displacement --config config.yaml
```

Pipeline stages, in the order you'd normally run them:

1. **QC** — `scripts/qc/compute_framewise_displacement.py`: Framewise
   Displacement vs. %BOLD signal change per subject/session (Power et al.
   2012 Fig. 2B), on raw or motion-regressed timeseries. `--test` runs on a
   tiny synthetic timeseries/motion-parameter fixture.
2. **Nodal graph metrics** — `scripts/graph_metrics/compute_nodal_metrics.py`:
   closeness, betweenness, clustering, and degree centrality per subject/
   session adjacency matrix. `--test` runs on synthetic adjacency matrices.
3. **Global graph metrics** — `scripts/graph_metrics/compute_global_metrics.py`:
   global efficiency, average clustering, average path length, modularity
   (`--metric` to pick one or all). `--test` included.
4. **Community structure** — `scripts/graph_metrics/compute_community_structure.py`:
   greedy-modularity communities plus spatial/graph distances between them.
   `--test` included.
5. **Global metric comparisons** —
   `scripts/graph_metrics/compare_global_metrics_lmm.py` (mixed-model
   session contrasts with FDR correction) and
   `scripts/graph_metrics/compare_global_metrics_permutation.py`
   (`--method unpaired|paired` permutation robustness checks). Both read the
   CSVs from steps 3-4 and include `--test`.
6. **Nodal metric comparisons** —
   `scripts/graph_metrics/compare_nodal_metrics_lmm.py` (per-region mixed
   models), then `scripts/graph_metrics/correct_nodal_pvalues.py`
   (Bonferroni + FDR/LSU correction), then
   `scripts/graph_metrics/plot_significant_nodes.py` (brain-space plots of
   FDR-significant regions; `--glass-brain` adds a netplotbrain figure).
   Each includes `--test`.
7. **Hub Disruption Index** — `scripts/hdi/compute_hdi.py` (per-subject
   within-subject kappa regression, `--metric degree_centrality|closeness|
   clustering|all`) then `scripts/hdi/validate_hdi_permutation.py`
   (`--null-model` chooses among 4 permutation null models). Both include
   `--test`.
8. **CCML** — `scripts/ccml/run_ccml.py`: builds the CCML input CSVs, fits
   the covariate-constrained 2D embedding, and runs a centroid-distance
   permutation test between the acute and chronic groups (`--metric`,
   `--data-prep-only` to stop after the input CSVs). `--test` included.
9. **Chord plots** — `scripts/chord_plots/generate_chord_plots.py`:
   per-subject, per-session connectivity chord diagrams colored by
   functional network. `--test` included.
10. **Behavior** — `scripts/behavior/state_sleepiness_vs_metrics.py`
    (sleepiness vs. graph metrics per session),
    `scripts/behavior/traits_vs_baseline_metrics.py` (traits vs. baseline
    graph metrics), `scripts/behavior/traits_vs_hdi_kappa.py` (traits vs.
    HDI kappas). Each needs `--survey-csv`; all include `--test`.

## Data

Two kinds of data live in this repo: reference atlases and the
already-extracted ROI timeseries. Everything the analysis scripts read
beyond that — per-subject adjacency matrices and SPM motion-parameter files
— is **derived data produced outside this repo** by
[rs_graph_processing](https://github.com/veronicamunoz/rs_graph_processing),
which fully reproduces the preprocessing described in the preprint; point
`data_root` at wherever you've generated that tree.

- **`AAL_atlas_89_regions/`** — a modified 89-region AAL atlas: NIFTI
  (`AAL_89_regions.nii`), region coordinates (`AAL_89_regions_coords.txt`,
  `..._coords_MNI.txt`), and a `;`-delimited region list
  (`AAL_89_regions_list.csv`, columns `Region;Node_number`). See
  `AAL_atlas_89_regions/Readme.md` for the atlas citation and modification
  details.
- **`AICHA_AAL_391_regions/`** — a modified 391-region AICHA atlas, same
  file layout (coordinates + `;`-delimited region list), no NIFTI.
- **`Timeseries/`** — extracted ROI timeseries, one file per
  condition/atlas/subject: `{RW|TSD|CSR}_{AAL|AICHA}_sub-NN_timeseries`,
  each a whitespace-separated matrix with no header, 384 timepoints (rows)
  by 89 (AAL) or 391 (AICHA) regions (columns).
- **Derived data (not in this repo)** — the scripts under `scripts/`
  expect `{data_root}/{ses-1,ses-2,ses-3}/{subject}/Graphs/wAALours/*.txt`
  adjacency matrices and `{data_root}/{session}/{subject}/RS/rp_*.txt` SPM
  motion parameters, produced by rs_graph_processing.
