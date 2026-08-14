# S1HS Submission Code

This repository contains the analysis code and supporting data for the manuscript entitled "System-scale genomic surveillance resolves the eco-epidemiological landscape of Salmonella" (aka S1HS). The code is organized around four analysis areas: diversity/coverage analysis, network construction and analyses, node centrality analysis/null-model testing, and KNN pan-serotype source attribution.

## Repository Layout

- `data/`: input data used by the scripts.
- `diversity_coverage/`: diversity/coverage analysis, including rarefaction, ECDF, AUC, Good's coverage, and sampling-completeness visualizations.
- `network/`: network construction and analyses, including source-similarity network construction, edge/source-direction analyses, and animal-feed association tests.
- `centrality/`: node centrality analysis/null-model testing using Dirichlet-multinomial null models for network centrality metrics.
- `KNN/`: KNN pan-serotype source attribution model, cross-validation, prediction scripts, and helper utilities.

## Data Files

The scripts expect input files in `data/` by default. The key committed files are:

- `metadata_training_testing.csv`: main isolate metadata table with HC assignments, curated source labels, KNN predictions, and trust/distance fields.
- `ECDF_HC5.csv`: precomputed closest-distance table used by diversity/coverage analysis.
- `HC50_distance_matrix.csv`: HC50 allelic-distance matrix used for phylogenetic/diversity calculations.
- `source_attribution_all_nodes.csv`: source-attribution summary table used for node-level analyses.
- `KNN_training_cgMLST.parquet`, `KNN_training_metadata.txt`, `KNN_external_cgMLST.csv`, `KNN_external_metadata.txt`: KNN pan-serotype source attribution model inputs and external prediction inputs.
- `metadata_Fig6.csv`: figure-specific metadata.


## Environment

Use Python 3.11 or newer. The analyses rely on the scientific Python stack:

```bash
pip install numpy pandas scipy statsmodels scikit-learn networkx matplotlib seaborn joblib dill pyarrow jupyterthemes
```

Some network plotting functions may require optional graph-layout packages available in the original analysis environment.

## Common Commands

Run the diversity/coverage analysis workflow:

```bash
python diversity_coverage/diversity_analysis.py
```

Run network construction and analyses:

```bash
python network/network_analysis.py
```

Run the animal-feed association analysis:

```bash
python network/animal_feed_analysis.py --skip-human-burden
```

Run animal-feed breadth plus human-burden sensitivity analyses and save result tables:

```bash
python network/animal_feed_analysis.py \
  --human-subsets all F G FG \
  --output-dir results/animal_feed
```

Run node centrality analysis/null-model testing:

```bash
python centrality/centrality_analysis.py
```

Run KNN pan-serotype source attribution cross-validation:

```bash
python KNN/KNN_CV.py
```

Run KNN pan-serotype source attribution prediction:

```bash
python KNN/KNN_pred_BPS.py
python KNN/KNN_pred_nBPS.py
```

## KNN pan-serotype source attribution

`KNN/source_attribution.py` contains reusable helper functions for classifying human isolates into food-animal (`F`), wild-animal (`W`), and generalist (`G`) categories. Defaults are:

- nearest-neighbor distance cutoff: `38`
- trust cutoff: `0.54`
- first nearest-neighbor index for both food-animal and wild-animal checks

If `animal_type` is already present in the metadata, the helper reuses it by default. Otherwise, it applies the distance/trust threshold logic to the stored KNN distance fields.


