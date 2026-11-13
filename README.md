# S1HS Submission Code

This repository contains the analysis code and supporting data for the manuscript entitled "Mapping the eco-epidemiological landscape of Salmonella across the United Sates food system" (aka Salmonella One Health surveillance, S1HS). The code is organized around five analysis areas: metadata curation (node definition), diversity/coverage analysis, network construction and analyses, node centrality and bootstrap role-stability analysis, and KNN pan-serotype source attribution.

## Repository Layout

- `data/`: input data used by the scripts.
- `metadata_curation/`: rule-based cleaning and source-category curation for isolate metadata and definition of network nodes.
- `diversity_coverage/`: diversity/coverage analysis, including rarefaction, ECDF, AUC, Good's coverage, and sampling-completeness visualizations.
- `network/`: network construction and analyses, including source-similarity network construction and parameter grid search, edge/source-direction analyses, and animal-feed association tests.
- `centrality/`: node centrality and generalism analyses; `centrality/role_stability/` contains the Figure 4b bootstrap node-role analysis.
- `KNN/`: KNN pan-serotype source attribution model, cross-validation, prediction scripts, and helper utilities.


## Data Files

The scripts expect input files in `data/` by default. The key committed files are:

- `metadata_training_testing.csv.gz`: main isolate metadata table with HC assignments, curated source labels, KNN predictions, and trust/distance fields.
- `ECDF_HC5.csv.gz`: precomputed closest-distance table used by diversity/coverage analysis; this is an upstream input for `diversity_coverage/diversity_analysis.py`.
- `HC50_distance_matrix.csv.gz`: HC50 allelic-distance matrix used for phylogenetic/diversity calculations.
- `source_attribution_all_nodes.csv.gz`: source-attribution summary table used for node-level analyses.
- `KNN_training_cgMLST.parquet`, `KNN_testing_cgMLST.parquet`, `KNN_external_BPS_cgMLST.csv.gz`, `KNN_external_BPS_metadata.txt.gz`: KNN pan-serotype source attribution model inputs and external prediction inputs.
- `metadata_Fig6.csv.gz`: figure-specific metadata.


## Environment

Use Python 3.11 or newer. The analyses rely on the scientific Python stack:

```bash
pip install numpy pandas scipy statsmodels scikit-learn networkx matplotlib seaborn joblib dill pyarrow jupyterthemes
```

Some network plotting functions may require optional graph-layout packages available in the original analysis environment.

Tabular text files in `data/` are stored as `.gz` files to keep the repository smaller. Pandas reads these files directly, so decompression is not required before running the scripts. `KNN_training_cgMLST.parquet` and `HC50_distance_matrix.csv.gz` are configured for Git LFS because they remain large after compression.

## Common Commands

Run these commands from the repository root.

Curate sample-source labels in a tab-separated NCBI Pathogen Detection metadata file (replace `metadata.tsv` with your input filename):

```bash
python metadata_curation/curate_source.py data/metadata.tsv
```

The script retains U.S. records and assigns broad `curated_source` labels by combining IFSAC categories with host, epidemiological, and free-text isolation-source metadata. By default, this example writes `data/metadata.USA.curated_sources.tsv`.

Gzip-compressed input is also supported. To choose the output path explicitly:

```bash
python metadata_curation/curate_source.py data/metadata.tsv.gz \
  --output results/metadata.USA.curated_sources.tsv.gz
```

The script creates the output directory if needed and compresses output files ending in `.gz`.

Run the diversity/coverage analysis workflow:

```bash
python diversity_coverage/diversity_analysis.py
```

The diversity/coverage script starts from the precomputed `data/ECDF_HC5.csv.gz` table and validates that `Source`, `HC50`, and `Closest distance` are present before computing rarefied ECDF/AUC and coverage metrics.

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

Run node centrality and generalism analyses:

```bash
python centrality/centrality_analysis.py
```

Run the Figure 4b bootstrap node-role analysis in a separate environment (Python 3.11):

```bash
python3.11 -m venv ~/.venvs/s1hs-role-stability
source ~/.venvs/s1hs-role-stability/bin/activate
python -m pip install -r centrality/role_stability/requirements-executed.txt jupyterlab nbconvert
python -m ipykernel install --user --name s1hs-role-stability --display-name "S1HS role stability"
cd centrality/role_stability
python -m jupyterlab centrality_role_stability.ipynb
```

Select the **S1HS role stability** kernel and run all cells. Alternatively, from `centrality/role_stability/`, execute the notebook without opening JupyterLab:

```bash
python -m jupyter nbconvert --to notebook --execute centrality_role_stability.ipynb \
  --ExecutePreprocessor.kernel_name=s1hs-role-stability \
  --ExecutePreprocessor.timeout=-1 \
  --output centrality_role_stability.executed.ipynb
```

The workflow uses the bundled `inputs/` snapshot: 125,582 isolates across 59 source nodes. It evaluates all 38 originally classified nodes using **1,000 outer bootstrap datasets**, each reconstructed with **500 independent within-source inner resamples**. Role thresholds are recalculated in every replicate. Four role frequencies plus non-retention are reported using all 1,000 datasets as the denominator.

Results and resumable checkpoints are written to `centrality/role_stability/outputs/<fingerprint>/`. Main outputs include `tables/role_frequencies_percent.csv`, `tables/role_recovery.csv`, `tables/centrality_summary.csv`, and `figures/Fig4b_role_stability.png` (also SVG). Four workers are used by default; adjust `WORKERS` for available resources. The notebook verifies the original baseline before running and stops on numerical failures. See [the detailed workflow instructions](centrality/role_stability/README.md) for inputs, environment caveats, resuming, and plot regeneration.

Return to the repository root (`cd ../..`) before running the following commands.

Run KNN pan-serotype source attribution cross-validation:

```bash
python KNN/KNN_CV.py
```

Run KNN pan-serotype source attribution prediction using external BPS (Bovine, Poultry, Swine) dataset:

```bash
python KNN/KNN_pred_external_BPS.py
```

`KNN/KNN_pred_testing.py` is a seeded demo script that intentionally subsamples records for non-BPS source prediction.

## KNN pan-serotype source attribution

`KNN/source_attribution.py` contains reusable helper functions for classifying human isolates into food-animal (`F`), wild-animal (`W`), and generalist (`G`) categories. Defaults are:

- nearest-neighbor distance cutoff: `38`
- trust cutoff: `0.54`
- first nearest-neighbor index for both food-animal and wild-animal checks

If `animal_type` is already present in the metadata, the helper reuses it by default. Otherwise, it applies the distance/trust threshold logic to the stored KNN distance fields.
