# S1HS Submission Code

This repository contains the analysis code and supporting data for the manuscript entitled "Mapping the eco-epidemiological landscape of Salmonella across the United Sates food system" (aka Salmonella One Health surveillance, S1HS). The code is organized around five analysis areas: metadata curation (node definition), diversity/coverage analysis, network construction and analyses, node centrality and bootstrap role-stability analysis, and KNN pan-serotype reservoir attribution.

## Repository Layout

- `data/`: input data used by the scripts.
- `metadata_curation/`: rule-based cleaning and source-category curation for isolate metadata and definition of network nodes.
- `diversity_coverage/`: diversity/coverage analysis, including rarefaction, ECDF, AUC, Good's coverage, and sampling-completeness visualizations.
- `network/`: network construction and analyses, including source-similarity network construction and parameter grid search, edge/source-direction analyses, and animal-feed association tests.
- `centrality/`: node centrality and generalism analyses; `centrality/role_stability/` contains the node role stability workflow.
- `KNN/`: KNN pan-serotype reservoir attribution model, cross-validation, prediction scripts, and helper utilities.


## Data Files

The scripts expect input files in `data/` by default. The key committed files are:

- `metadata_training_testing.csv.gz`: main isolate metadata table with HC assignments, curated source labels, KNN predictions, and trust/distance fields.
- `ECDF_HC5.csv.gz`: precomputed closest-distance table used by diversity/coverage analysis; this is an upstream input for `diversity_coverage/diversity_analysis.py`.
- `HC50_distance_matrix.csv.gz`: HC50 allelic-distance matrix used for phylogenetic/diversity calculations.
- `source_attribution_all_nodes.csv.gz`: source-attribution summary table used for node-level analyses.
- `KNN_training_cgMLST.parquet`, `KNN_testing_cgMLST.parquet`, `KNN_external_BPS_cgMLST.csv.gz`, `KNN_external_BPS_metadata.txt.gz`: KNN pan-serotype reservoir attribution model inputs and external prediction inputs.
- `metadata_Fig6.csv.gz`: figure-specific metadata.


## System requirements

### Operating system and tested versions

The demos were tested on **macOS 26.5.1 (build 25F80), Apple Silicon (arm64)**. The network demo used **Python 3.13.2**, and the KNN reservoir-attribution demo used **Python 3.11.8**. The node role stability workflow was also validated with Python 3.13.2, using archived checkpoints and three freshly recomputed replicates. Linux and Windows have not been tested for this release; installation commands below use a macOS/Linux shell.

### Software dependencies

The following versions were present in the successful demo runs. A dash means the package is not required by that demo. These are tested configurations, not minimum-version guarantees.

| Dependency | Network demo | KNN demo |
| --- | --- | --- |
| NumPy | 2.4.4 | 1.26.4 |
| pandas | 2.3.1 | 2.2.3 |
| SciPy | 1.15.2 | 1.14.1 |
| scikit-learn | 1.6.1 | 1.5.1 |
| NetworkX | 3.6.1 | — |
| Matplotlib | 3.10.9 | 3.9.2 |
| seaborn | 0.13.2 | 0.13.2 |
| tqdm | 4.67.1 | — |
| python-louvain | 0.16 | — |
| igraph | 0.11.8 | — |
| leidenalg | 0.10.2 | — |
| scikit-bio | 0.7.0 | — |
| joblib | 1.4.2 | 1.4.2 |
| dill | — | 0.3.9 |
| jupyterthemes | — | 0.20.0 |
| h5py | 3.13.0 | 3.12.1 |
| psutil | — | 5.9.8 |
| PyTorch | — | 2.2.2 |

Install demo dependencies with `demos/requirements.txt`. scikit-learn is constrained to `>=1.5,<1.7` because the KNN implementation uses its internal neighbor classes. Other demo dependencies are not pinned, so a new installation may differ from the tested environments. In the network test environment, pandas imported as 2.3.1 although its installed distribution metadata reported 2.2.3; the table reports the imported version.

Additional workflow dependencies include **statsmodels** (0.14.2 in the KNN environment; 0.14.5 in the network environment) for association analyses and **pyarrow** (14.0.2 and 24.0.0, respectively) for the full Parquet datasets and demo-data regeneration. Their presence does not imply that every full-data workflow was rerun during demo validation.

The node role stability workflow's pinned versions, including Pillow and Jupyter kernel packages, are in `centrality/role_stability/requirements-executed.txt`; its validation environment also used **IPython 9.13.0**. JupyterLab is needed only to open that notebook interactively and was not part of the command-line validation. Full-network `sfdp` plots additionally require Graphviz and PyGraphviz (1.14 was installed in the network environment); those layouts were not exercised by the demos, and no tested Graphviz version is claimed.

### Hardware

**No non-standard hardware or GPU is required.** Both demos run on a standard desktop CPU; PyTorch is imported by shared KNN utilities, but the demo does not use GPU acceleration. Four CPU cores and 8 GB RAM are a practical starting configuration for the small bundled demos, not a measured minimum. Full-data analyses may require more memory. The node role stability workflow defaults to four CPU workers, adjustable through `WORKERS` in the notebook.

## Installation guide

### Instructions

Install Python 3.11 and Git first. The following commands use a macOS/Linux shell. Clone the repository and create an isolated environment:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/denglab/S1HS.git
cd S1HS
python3.11 -m venv ~/.venvs/s1hs-demo
source ~/.venvs/s1hs-demo/bin/activate
python -m pip install --upgrade pip
python -m pip install -r demos/requirements.txt
```

This installs the dependencies for both bundled demos. The clone command skips large Git LFS files, which the demos do not need. If you already have the repository, start with the environment-creation command from its root directory.

For the full-data analyses, install Git LFS, then download the large datasets and install additional analysis dependencies:

```bash
git lfs install
git lfs pull
python -m pip install statsmodels joblib pyarrow
```

The node role stability workflow has a separate pinned environment; follow its setup under Common Commands. Some full-network plots additionally require Graphviz and its Python bindings for the `sfdp` layout; the network demo uses a spring layout and does not require Graphviz.

Tabular text files in `data/` are compressed as `.gz`; pandas reads them directly without decompression. Both KNN training/testing Parquet files and `HC50_distance_matrix.csv.gz` use Git LFS.

### Typical installation time

Allow approximately **5–20 minutes** to create the environment and install demo dependencies on a typical desktop with broadband internet and prebuilt Python package wheels available. This is a planning estimate, not a measured clean-install benchmark. It excludes installing Python/Git, downloading the full Git LFS datasets, and running analyses. Slow downloads or packages that require compilation can take longer.

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

Run the node role stability workflow:

```bash
python3.11 -m venv ~/.venvs/s1hs-role-stability
source ~/.venvs/s1hs-role-stability/bin/activate
python -m pip install -r centrality/role_stability/requirements-executed.txt jupyterlab
python -m ipykernel install --user --name s1hs-role-stability --display-name "S1HS role stability"
cd centrality/role_stability
python -m jupyterlab centrality_role_stability.ipynb
```

Select the **S1HS role stability** kernel and run all cells. Using the bundled inputs, the analysis assesses role stability for 38 nodes across 1,000 bootstrap datasets, each with 500 inner resamples to reconstruct the consensus network. Roles are reassigned in each dataset, and role and non-retention frequencies are summarized across all datasets.

Tables, the node role stability heatmap, and resumable checkpoints are saved under `centrality/role_stability/outputs/<fingerprint>/`. Matching checkpoints are reused when rerunning the notebook.

Return to the repository root (`cd ../..`) before running the following commands.

Run KNN pan-serotype reservoir attribution cross-validation:

```bash
python KNN/KNN_CV.py
```

Run KNN pan-serotype reservoir attribution prediction using external BPS (Bovine, Poultry, Swine) dataset:

```bash
python KNN/KNN_pred_external_BPS.py
```

`KNN/KNN_pred_testing.py` is a seeded demo script that intentionally subsamples records for non-BPS source prediction.

## Demo

Two small demos use the study's existing network-construction and weighted KNN functions. Their input datasets are bundled in `data/demo/`, so running them does not require downloading the full Git LFS datasets. The subsets demonstrate execution and output formats; they do not reproduce the full manuscript results.

### Setup

Follow the Installation guide above, then activate the demo environment from the repository root:

```bash
source ~/.venvs/s1hs-demo/bin/activate
```

### Network construction

**Demo data:** `data/demo/network_metadata.csv.gz` contains 1,896 isolates from 10 source nodes, with BioSample identifiers, source labels, and HC50 assignments. Up to 200 isolates per source were sampled without replacement from the study metadata.

**Run:**

```bash
python demos/network_demo.py
```

The demo constructs a cosine-similarity network using 100 stratified bootstrap iterations, a similarity cutoff of 0.35, Leiden communities, and consensus edge/community thresholds of 0.5.

**Expected output:** `results/demo_network/` contains:

- `nodes.csv`: centrality metrics and community assignments for 10 nodes.
- `edges.csv`: consensus edges and their bootstrap support (9 edges in the tested environment).
- `edge_support.csv`: the 10 × 10 edge-support matrix.
- `network.png`: a network visualization using a seeded spring layout.
- `run_summary.json`: dataset size, network size, and runtime.

**Expected desktop runtime:** usually under one minute after installation; approximately **6 seconds** in the local macOS test, including imports and figure export. Results and timing can vary with software versions and hardware.

### KNN reservoir attribution

**Demo data:** `data/demo/knn_training.csv.gz` contains 900 training isolates (300 each from Bovine, Poultry, and Swine); `data/demo/knn_queries.csv.gz` contains 30 non-BPS query isolates. Both retain all 3,002 cgMLST loci. Training and query BioSample identifiers do not overlap.

**Run:**

```bash
python demos/knn_demo.py
```

The demo fits the study's weighted KNN classifier with 11 neighbors and predicts source labels for the 30 queries using the reduced training set.

**Expected output:** `results/demo_knn/` contains:

- `predictions.csv`: 30 rows with query identifiers, recorded source labels, predicted Bovine/Poultry/Swine labels, rounded class probabilities, trust scores, and nearest-neighbor details.
- `run_summary.json`: training/query counts, number of loci, and runtime.
- A `.pkl` BallTree cache written by the existing classifier.

Recorded query sources are not restricted to the three training classes; this demo is not an accuracy benchmark. Predictions from this reduced training set should not be substituted for the full-data analysis.

**Expected desktop runtime:** usually under one minute after installation; approximately **14 seconds** in the local macOS test, including imports and file loading. No GPU is required.

Both commands accept `--output-dir PATH` to choose another output location. Reusing an output directory overwrites demo output files.

The sampling seed (`1127825`), source filenames, and demo-file checksums are recorded in `data/demo/manifest.json`. To regenerate the bundled subsets from the full input datasets, install `pyarrow` and run `python demos/prepare_data.py`; this optional step requires the full Git LFS data.

## KNN pan-serotype reservoir attribution

`KNN/source_attribution.py` contains reusable helper functions for classifying human isolates into food-animal (`F`), wild-animal (`W`), and generalist (`G`) categories. Defaults are:

- nearest-neighbor distance cutoff: `38`
- trust cutoff: `0.54`
- first nearest-neighbor index for both food-animal and wild-animal checks

If `animal_type` is already present in the metadata, the helper reuses it by default. Otherwise, it applies the distance/trust threshold logic to the stored KNN distance fields.
