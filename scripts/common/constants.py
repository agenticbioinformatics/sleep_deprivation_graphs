"""Shared constants for the sleep-deprivation graph analysis pipeline."""

SESSION_ORDER = ["ses-1", "ses-2", "ses-3"]

SESSION_LABELS = {
    "ses-1": "baseline (rested wakefulness, RW)",
    "ses-2": "acute (total sleep deprivation, TSD)",
    "ses-3": "chronic (sleep restriction, CSR)",
}

# Short condition codes used in filenames throughout the repo (e.g. Timeseries/).
SESSION_TO_CONDITION = {"ses-1": "RW", "ses-2": "TSD", "ses-3": "CSR"}

NODAL_METRICS = ["Closeness", "Betweenness", "Clustering", "Degree_centrality"]

# Nodal metrics used by the HDI and CCML stages (a subset of NODAL_METRICS --
# betweenness is not used there in the source notebook).
HDI_CCML_METRICS = ["Degree_centrality", "Closeness", "Clustering"]
METRIC_FILE_SLUG = {"Degree_centrality": "degree", "Closeness": "closeness", "Clustering": "clustering"}

# (test_session, ref_session) pairs used by the HDI stages, keyed by a short name.
HDI_SESSION_PAIRS = {
    "acute_vs_control": ("ses-2", "ses-1"),
    "chronic_vs_control": ("ses-3", "ses-1"),
    "chronic_vs_acute": ("ses-3", "ses-2"),
}
HDI_SESSION_PAIR_LABELS = {
    "acute_vs_control": "TSD vs RW",
    "chronic_vs_control": "CSR vs RW",
    "chronic_vs_acute": "CSR vs TSD",
}

# Pairwise session comparisons used throughout the per-region nodal-metric
# stages (compare_nodal_metrics_lmm.py, correct_nodal_pvalues.py, plot_significant_nodes.py).
NODAL_SESSION_COMPARISONS = ["ses1_vs_ses2", "ses1_vs_ses3", "ses2_vs_ses3"]
NODAL_SESSION_COMPARISON_LABELS = {
    "ses1_vs_ses2": "Acute vs Control",
    "ses1_vs_ses3": "Chronic vs Control",
    "ses2_vs_ses3": "Chronic vs Acute",
}

GLOBAL_METRIC_FUNCS = [
    "global_efficiency",
    "average_clustering",
    "average_path_length",
    "modularity",
]

# GLOBAL_METRIC_FUNCS plus avg_graph_distance (written by
# compute_community_structure.py, not compute_global_metrics.py) -- the full
# set used by the LMM/permutation comparisons and the behavior stages.
ALL_GLOBAL_METRICS = [*GLOBAL_METRIC_FUNCS, "avg_graph_distance"]

# ROI (AAL, 89-region) -> functional network, used for chord-plot coloring.
# Fixed typo from the source notebook: 'Occipial_R' -> 'Occipital_R' (the
# original typo silently dropped this node into the 'Other' bucket instead
# of 'VN').
ROI_NETWORK_MAP = {
    "PreGy_L": "SMN", "PreGy_R": "SMN", "SMA_L": "SMN", "SMA_R": "SMN",
    "RolandOperc_L": "SMN", "RolandOperc_R": "SMN", "Postcentral_L": "SMN", "Postcentral_R": "SMN",
    "ParacentralLob_L": "SMN", "ParacentralLob_R": "SMN",

    "FrontSup_L": "FPN", "FrontSup_R": "FPN", "FrontMid_L": "FPN", "FrontMid_R": "FPN",
    "FrontInfOperc_L": "FPN", "FrontInfOperc_R": "FPN", "FrontInfTri_L": "FPN", "FrontInfTri_R": "FPN",
    "FrontInfOrb_L": "FPN", "FrontInfOrb_R": "FPN", "FrontSupMed_L": "FPN", "FrontSupMed_R": "FPN",

    "CingAnt_L": "DMN", "CingAnt_R": "DMN", "CingMid_L": "DMN", "CingMid_R": "DMN",
    "CingPost_L": "DMN", "CingPost_R": "DMN", "Precuneus_L": "DMN", "Precuneus_R": "DMN",
    "Angular_L": "DMN", "Angular_R": "DMN", "FrontMedOrb_L": "DMN", "FrontMedOrb_R": "DMN",
    "TempMid_L": "DMN", "TempMid_R": "DMN",

    "Insula_L": "Salience", "Insula_R": "Salience", "Olfactory_L": "Salience", "Olfactory_R": "Salience",

    "Calcarine_L": "VN", "Calcarine_R": "VN", "Cuneus_L": "VN", "Cuneus_R": "VN",
    "Lingual_L": "VN", "Lingual_R": "VN", "Occipital_L": "VN", "Occipital_R": "VN",
    "Fusiform_L": "VN", "Fusiform_R": "VN",

    "Amygdala_L": "Limbic", "Amygdala_R": "Limbic", "Hippocampus_L": "Limbic", "Hippocampus_R": "Limbic",
    "ParaHippoc_L": "Limbic", "ParaHippoc_R": "Limbic", "TempPole_L": "Limbic", "TempPole_R": "Limbic",

    "Heschl_L": "Auditory", "Heschl_R": "Auditory", "TempSup_L": "Auditory", "TempSup_R": "Auditory",
    "TempInf_L": "Language", "TempInf_R": "Language",

    "Caudate_L": "Subcortical", "Caudate_R": "Subcortical", "Putamen_L": "Subcortical",
    "Putamen_R": "Subcortical", "Pallidum_L": "Subcortical", "Pallidum_R": "Subcortical",
    "Thalamus_L": "Subcortical", "Thalamus_R": "Subcortical",

    "ParietalSup_L": "DAN", "ParietalSup_R": "DAN", "ParietalInf_L": "DAN", "ParietalInf_R": "DAN",
    "SupraMarginal_L": "DAN", "SupraMarginal_R": "DAN",

    "Cereb_I_II_L": "Cerebellum", "Cereb_I_II_R": "Cerebellum",
    "Cereb_III_VI_L": "Cerebellum", "Cereb_III_VI_R": "Cerebellum",
    "Cereb_VII_X_L": "Cerebellum", "Cereb_VII_X_R": "Cerebellum",
    "Vermis": "Cerebellum",

    "FrontSupOrb_L": "Other", "FrontSupOrb_R": "Other",
    "FrontMidOrb_L": "Other", "FrontMidOrb_R": "Other",
}
