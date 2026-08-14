## AUC,ECDF,Good's C, and Obs/Chao1 
### Quantify and visualize how completely each source has been sampled
# 1. Lineage-based coverage – how many distinct genetic lineages (clusters) have already been represented in the dataset
# For each source, isolates were grouped by lineage (e.g., HC50 cluster).
# Good’s coverage C=1−f1/N was calculated, where: 
# N = number of isolates in that source
# f1 = number of lineages observed only once (singletons)
# This metric reflects the fraction of the population’s lineage diversity already represented in the sample.
# High C (~1.0) → few singletons → sampling nearly complete.
# Low C → many unique lineages → under-sampling.
# 2. Distance-based coverage – how densely the sampled isolates cover the underlying genetic space (i.e., how many isolates have close genomic neighbors).
# Using the “closest distance” column (pairwise allelic differences), we measured the proportion of isolates whose nearest neighbor lies within a threshold (e.g., ≤ 50 allelic differences).
# High coverage → most isolates have close relatives → dense sampling in genetic space.
# Low coverage → many isolates are genetically isolated → sparse sampling of the population.
# 3. Rarefied, bootstrapped comparison
# Because different sources have very different sample sizes (some with only a few hundred isolates, others with >10,000), we used rarefaction to ensure fair, size-matched comparison.
# To correct for uneven sampling effort, both coverage metrics were recomputed at a standardized subsample size n∗  (e.g., 200 isolates per source). For each source:
# 1,000 bootstrap subsamples of size n were drawn without replacement.
# Mean coverage and 95 % confidence intervals were estimated across replicates.
# This ensures that differences between sources reflect sampling completeness, not simply sample size.
# 4. Mean AUC (within-source compactness) — derived from ECDFs of pairwise genetic distances between isolates within each source.
# AUC ≈ 1 means isolates are tightly clustered (genetically homogeneous).
# AUC ≈ 0.5 means intermediate compactness.
# Lower AUC indicates broader genetic diversity within the source.
# Error bars: 95% bootstrap confidence intervals for both metrics, computed using the same rarefied sample size (n*=200) per source.


from diversity_funcs import *
import os

current_path = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(current_path)
DATA_PATH = os.path.join(SRC_ROOT, "data")


def data_file(filename):
    path = os.path.join(DATA_PATH, filename)
    gz_path = path + ".gz"
    return gz_path if os.path.exists(gz_path) else path

### Comparing innate diversity (AUC) with sampling completeness
# Nodes with at least 150 available isolates after HC5-level de-redundancy were included
df_ecdf_HC5 = pd.read_csv(data_file("ECDF_HC5.csv"))

# Compute rarefied metrics
res_ecdf_HC5, res_auc_HC5 = rarefied_ecdf(
    df_ecdf_HC5,
    lineage_col="HC50",
    distance_col="Closest distance",
    thresholds=(20, 50, 100),
    B=500,
    n_star=150,
    random_seed=42,
    max_d=100,
    auc_dmax=50
)
print(res_auc_HC5)

## Plot Mean AUC vs Good's C for each source, colored by source category, with error bars and annotations
# Diversity (AUC) vs lineage coverage, fit a line, define quadrant, medians are same as the defult values
# names for all 60 nodes
node_labels_override = {'animal_feed': 'Animal Feed',
 'aquatic_animal': 'Aquatic Animals',
 'companion_animal': 'Companion Animals',
 'equine': 'Equine',
 'fruit_and_vegetable': 'Fruits/Vegetables',
 'grains': 'Grains',
 'herbs': 'Herbs',
 'nuts': 'Nuts',
 'production_env_produce': 'FPE-Produce',
 'soil_northeast': 'Soil-NE',
 'wild_animal': 'Wild Animals',
 'dairy': 'Dairy',
 'forest': 'Forest',
 'multi_ingredient_other': 'Multi-ingredient Food (other)',
 'multi_ingredient_poultry': 'MIF-Poultry',
 'rodents': 'Rodents',
 'rte': 'RTE',
 'poultry': 'Poultry',
 'bovine': 'Bovine',
 'swine': 'Swine',
 'human': 'Humans',
 'produce_outbreak': 'Produce Outbreaks',
 'bird': 'Birds',
 'plant_based_low_water_activity_food': 'PLWF',
 'bsaaos': 'BSAAOs',
 'goat_sheep': 'Goat/Sheep',
 'multi_ingredient_pork_beef_dairy': 'MTF-PBD',
 'seeds': 'Seeds',
 'seasoning': 'Seasoning',
 'root_undergroudn': 'Root/Underground',
 'confectionery': 'Confectionery',
 'camelids': 'Camelids',
 'beans': 'Beans',
 'water': 'Water (combined)',
 'soil': 'Soil',
 'env_water': 'Env. Water',
 'aquatic_animals': 'Aquatic Animals',
 'env_unclear': 'Food Production Env.',
 'vegetable_snack_others': 'Plant-based Dry Food',
}

label_pos_override={
    "wild_animal": (-1, 0.1),
    "fruit_and_vegetable": (-1, 0.1),
    "equine": (-1, 0.1),
    "swine": (1, 0.8),
    "poultry": (1, 0),
    "bovine": (0, 0),
    "vegetable_snack_others": (0, 0.1),
    "goat_sheep": (-1, 0.1),
    "nuts": (-1, 0.1),
    "aquatic_animals": (-1, 0.1),
    "multi_ingredient_other": (-1, 0.1),
    "nuts": (1, 0.1),
    "env_water": (-1,0),
    "multi_ingredient_other": (1,0),
    "companion_animal": (0,0),
    "vegetable_snack_others":(-1,0),
}

plot_diversity_vs_coverage(
    df=res_auc_HC5,
    figsize=(8, 6.5),
    y_metric="AUC",
    x_metric="GoodsC",
    regression='linear',
    line_color="gray",
    band_alpha=0.3,
    node_label=node_labels_override,
    node_label_fontsize=13,
    axis_label_fontsize=15,
    tick_fontsize=15,
    frame_weight=1,
    min_node_size=150, # number of isolates
    random_seed=186,
    quadrant=False,
    color_by_ObsChao1=True,
    cmap_name='viridis', # plasma, inferno, magma, cividis, turbo, etc
    xlim=(0.2, 0.97),
    ylim=(0.1, 0.75),
    label_pad_pts=1,
    label_pos_override=label_pos_override,
    errorbar_color="gray",
    errorbar_alpha=0.5,
    errorbar_lw=0.8,
    errorbar_capsize=0,
    node_alpha=0.9,
    node_edge_color="black",
    node_size_range=(50, 1000)
)

## AUC curves for each source, with confidence intervals
title_label_override={
    "env_unclear": "FPE",
    "vegetable_snack_others": "plant-based dry food",
    "fruit_and_vegetable": "fruits/vegetables",
    "multi_ingredient_other": "MIF (other)",
}
plot_ecdf_multiples(res_ecdf_HC5, res_auc_HC5, ncols=4,
                    line_color="black", ci_color="#9AD872", ci_alpha=0.8,
                    show_subtitle=False, show_subplot_title=False, show_auc_ci=False,
                    auc_dmax=50, area_alpha=0.25, area_color="gray", label_override=title_label_override)
