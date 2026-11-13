# Table definitions

## Input tables

`prepared_isolates.csv.gz`: one row per retained pre-filter isolate record. `biosample_acc` is the original BioSample accession; `Name` is the original isolate name; `HC50` is its cluster identifier; `curated_source_region` is the curated source/region category; `HC50_code` is the zero-based feature code in the vector archive; `input_order` is the zero-based row order after the original preprocessing. These rows are not a claim of epidemiological independence.

`hc50_mapping.csv`: one row per feature; `HC50_code` is its zero-based computational code and `HC50` its original identifier.

## Frequencies

`role_frequencies_percent.csv`: one row per original `node` (source identifier). `Global Connector`, `Broker/Bridge`, `Core Influencer`, and `Peripheral/Local` are percentages assigned those roles. `Not retained` is the percentage removed by consensus-community filtering. All five columns use all 1,000 outer replicates and sum to 100% per row. `NR` is the figure abbreviation for `Not retained`.

`role_frequency_mcse_percentage_points.csv`: the same rows and columns give Monte Carlo standard errors in percentage points for the estimated frequencies. These quantify finite-replicate error, not biological uncertainty or probabilities that roles are true.

`role_recovery.csv`: one row per original `node`. `baseline_role` is its original descriptive label; `recovery_percent` is the percentage of all 1,000 replicates recovering that label; `retention_percent` is the percentage surviving filtering, irrespective of role. Non-retention is included in both denominators.

## Replicate tables

`replicate_outcomes.csv.gz`: one row per original `node` and zero-based `replicate`. `role` is one of the five outcomes. `eigenvector`, `betweenness`, `degree`, `strength`, `closeness`, and `clustering` are the raw calculated network metrics. Degree is normalized by graph size minus one; strength by graph maximum; path metrics use inverse-support lengths. Columns ending `_percentile` are average-tie percentile ranks (0-100) among all retained graph nodes. Metrics and ranks are missing for non-retained nodes, not zero.

`all_retained_metrics.csv.gz`: the same columns and definitions, but rows include every retained node in each reconstructed graph, not only the original 38-node reporting cohort. `role` therefore excludes `Not retained` in this table.

`replicate_thresholds.csv`: one row per zero-based `replicate`; `graph_nodes` and `graph_edges` count the filtered network's vertices and edges; `eigenvector` and `betweenness` are that replicate's 75th-percentile role cutoffs. Values equal to cutoffs are classified as high.

## Centrality summaries

`centrality_summary.csv`: one row per original `node` and `metric`. `retained_replicates` is the number contributing to that summary. `raw_lower`, `raw_median`, and `raw_upper` are the 2.5th, 50th, and 97.5th percentiles of the raw metric. `percentile_lower`, `percentile_median`, and `percentile_upper` are the corresponding summaries of within-graph percentile ranks. Summaries omit non-retained observations and do not assign them zero. The ranges describe bootstrap variation, not simultaneous confidence intervals.

## Editorial mapping

The parent `equation_mapping.csv` has one row per `original` equation number. `revised` is the accepted document's new number, missing if deleted; `action` distinguishes retained, deleted, and renumbered equations. It is not an analysis-results table.
