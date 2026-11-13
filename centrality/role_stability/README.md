# Bootstrap role stability

This analysis asks whether each source retains its original descriptive network role when isolates are sampled again within sources. It reconstructs a full consensus network for each outer bootstrap dataset. It does not test departures from random HC50 allocation.

## Run

Follow the environment setup in the [repository README](../../README.md), then open `centrality_role_stability.ipynb` using the `S1HS role stability` kernel, with `centrality/role_stability/` as the working directory. Keep the adjacent `inputs/` directory with it. Run all cells for 1,000 outer replicates, each using 500 independent-source inner bootstrap iterations. A 20-replicate pilot is included in that total. Four independent process workers are used by default; changing worker count does not change seeds.

All functions are in the notebook. It imports no project-local helper modules and requires no original notebooks, external data paths, or downloads. The portable inputs contain the complete ordered, pre-filter source/HC50 vectors and original baseline references. `prepared_isolates.csv.gz` and `hc50_mapping.csv` preserve isolate provenance and feature identities; the vector archive is the efficient computational representation. `manifest.json` records original source paths as provenance only, not runtime dependencies.

Third-party packages are NumPy, pandas, SciPy, scikit-learn, NetworkX, python-igraph, leidenalg, joblib, Matplotlib, Pillow, IPython, and the Jupyter kernel. Exact executed versions are recorded in each output manifest and `requirements-executed.txt`.

The local environment reports pandas 2.3.1 from the imported module and 2.2.3 from distribution metadata. Both are recorded in `runtime_identity.json`; the requirements file uses imported versions. The notebook checks both identities before resuming. Reproduction was verified in the existing `sklearn-env` kernel, not in a newly installed environment.

## Resume and plot

Outputs are fingerprinted under `outputs/`. Completed matching replicates are reused; incompatible inputs, scientific definitions, configuration, or package versions create a different output directory. Numerical failures are recorded and stop completion. Do not delete checkpoints unless deliberately rerunning from scratch.

After a completed run, execute configuration, scientific definitions, input/baseline verification, and the final table/figure section to regenerate plots without redoing the 1,000 outer replicates. The baseline verification itself takes two consensus reconstructions. Edit `PLOT` to adjust font sizes, node labels, column labels, `NR`, role colors, panel letter, margins, aspect ratio, rotation, resolution, and cell annotations. Plot configuration is separate from the scientific fingerprint.

The publication panel is sized to the original panel-b rectangle (1,544 x 3,515 pixels within the 4,565 x 3,515 full figure). A larger inspection panel is also exported. The assembly operation verifies the legacy ratio rather than distorting a custom-aspect panel. The original figure is a visual assembly input only; it does not supply any numerical analysis results.

## Interpretation

Every original node has five outcomes: global connector, broker/bridge, core influencer, peripheral/local, and not retained. Frequencies use all 1,000 outer replicates and sum to 100%. All 38 original nodes, including human, are shown. Label colors remain their original descriptive assignments.

Role stability is conditional on isolate exchangeability and the chosen reconstruction procedure. It does not remove surveillance bias or outbreak dependence, establish transmission direction, or measure the probability that a biological role is true. The independent inner-source implementation is distinguished from the original baseline implementation; baseline labels are not silently redefined.

Complete table definitions are in `table-notes.md`. The notebook performs baseline, deterministic-rerun, checkpoint-resume, input-integrity, and figure-layout checks. Generated outputs and checkpoints are not bundled; a first run computes all replicates.

## Repository integration validation

On 2026-11-13, the imported notebook was validated in a temporary directory using the existing `sklearn-env` environment. All cell sources and input-file hashes matched the archived workflow. Both baseline reconstructions ran successfully, all 1,000 archived checkpoints were loaded, and replicates 0, 499, and 999 were freshly recomputed and matched their archived results. Table and PNG/SVG exports completed; role frequencies, role recovery, and centrality summaries matched the archived tables. This check reused archived checkpoints rather than recomputing all 1,000 replicates and did not test installation into a fresh environment.
