import os
from centrality_funcs import *
import numpy as np
import pandas as pd

current_path = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(current_path)
DATA_PATH = os.path.join(SRC_ROOT, "data")

def data_file(filename):
    path = os.path.join(DATA_PATH, filename)
    gz_path = path + ".gz"
    return gz_path if os.path.exists(gz_path) else path


df_all_training_testing = pd.read_csv(data_file("metadata_training_testing.csv"))

# Consensus network by bootstrapping (HC50/0.35/0.5/0.5) -- cosine
Graphs_boot_cosine, G_mean_cosine, G_consensus_cosine, sim_matrix_cosine, feature_matrix_cosine, partitions_cosine, partition_consensus_cosine, modularity_scores_cosine, modularity_score_mean_cosine, modularity_score_consensus_cosine, classes, coassign_matrix_partitions_df_cosine, coassign_matrix_bootstraps_df_cosine = build_similarity_network(
    df_all_training_testing, feature_col='HC50', class_col='curated_source_region',
    similarity='cosine',
    sim_threshold=0.35,
    community_detection_method='leiden',
    consensus_partition_threshold=0.5, # controls how many communities are detected using the consensus partition of all bootstrapping iterations. The lower, the fewer communities
    consensus_graph_threshold=0.5, # Use threshold to create a consensus graph from the coassign_matrix_bootstraps_df; add edges only if the co-assignment score is above this threshold
    bootstrap_iters=500,
    bootstrapping_strategy='stratified',
    G_type='consensus',
    show=True,
)

## Remove singleton communities 
result_summary_no_singltons, G_consensus_cosine_no_singletons, partition_consensus_cosine_no_singletons, singleton_nodes_consensus = remove_singleton_communities(G_consensus_cosine, partition_consensus_cosine, weight='weight')

## Nodes remaining in consensus network after removing singleton communities
nodes_in_consensus_network = ['animal_feed', 'aquatic_animal', 'companion_animal', 'equine', 'fruit_and_vegetable', 'grains', 'herbs', 'nuts', 'production_env_produce', 'soil_northeast', 'dairy', 'food_production_env_pacific', 'food_production_env_mountain', 'food_production_env_southern_plains', 'food_production_env_corn_belt', 'food_production_env_northeast', 'food_production_env_appalachian', 'food_production_env_na', 'food_production_env_southeast', 'food_production_env_lake_states', 'food_production_env_northern_plains', 'water_northeast', 'water_southeast', 'water_pacific', 'water_appalachian', 'sewage_hi', 'water_northern_plains', 'water_southern_plains', 'water_na', 'forest', 'multi_ingredient_other', 'multi_ingredient_poultry', 'rodents', 'rte', 'poultry', 'bovine', 'swine', 'human',]
# update df_all_training_testing using node_list
df_all_training_testing_no_singletons = df_all_training_testing[df_all_training_testing['curated_source_region'].isin(nodes_in_consensus_network)]

# Dirichlet-Multinomial test for node metrics 

## Identify testable and untestable nodes for H0 and H1
# H0: A node with the same richness and isolate count as the observed node, but whose isolates are drawn from the population-wide HC50 distribution (excluding that node), has the same expected centrality metrics
# H1: The observed node’s centrality metrics differ significantly from those expected under this population-based composition model, implying a distinctive ecological or transmission role

# Run the screen for all nodes
results_step2_screen = screen_nodes_for_dm_testability(
    df=df_all_training_testing_no_singletons,
    node_col="curated_source_region",
    level_col="HC50",
    id_col="biosample_acc",
    taus=TAU_GRID_GENERIC,
    B_calib=1000,
    B_pseudo=2000,
    alpha_support=0.10,
    counts_mode="positive",
    min_avail_per_level=1,
    median_tol=0.10,
    require_min_yield=0.80,
    rng_seed=123,
)

# Nodes that are "like human": structurally cannot instantiate H0
nodes_structural_not_testable = results_step2_screen.loc[
    results_step2_screen["structural_not_testable"],
    ["node", "obs_n", "obs_K", "K_bg", "K_star", "tau_star",
     "draws_generated", "supply_fail_rate", "reason_structural"]
]

print("\n[Nodes structurally not testable for H0/H1 (human-like)]:")
print(nodes_structural_not_testable.to_string(index=False))

# If you also want all nodes that fail Step 2 QC (including structural), use:
nodes_not_testable_overall = results_step2_screen.loc[
    results_step2_screen["not_testable_overall"],
    ["node", "obs_n", "obs_K", "K_bg", "K_star", "tau_star",
     "yield_rate", "delta_med_l1o", "ready_step3",
     "structural_not_testable", "qc_not_testable",
     "reason_structural", "reason_qc"]
]

print("\n[All nodes not testable for H0/H1 (structural OR QC)]:")
print(nodes_not_testable_overall.to_string(index=False))

print("\n[Nodes testable for H0/H1]:")
print(results_step2_screen)
testable_nodes = results_step2_screen.loc[
    ~results_step2_screen["not_testable_overall"],
    "node"
].tolist()
print(testable_nodes)


## DM test for all testable nodes
## run Step 2 and stability analysis for all testable nodes
all_step2_results = run_step2_and_stability_for_all(
    testable_nodes=testable_nodes,
    df=df_all_training_testing_no_singletons,
)

## Step 3: Hypothesis testing using synthetic nodes
target_nodes_for_step3 = [n for n in testable_nodes if n != "water_northern_plains"] # water_northern_plains only has 4 isolates
print(target_nodes_for_step3)

step3_results = run_step3_for_target_nodes(
    target_nodes=target_nodes_for_step3,
    df=df_all_training_testing_no_singletons,
    G_obs=G_consensus_cosine_no_singletons,
    qc_suffix="_res",
    info_suffix="_info",
    n_null=2000,
    edge_threshold=0.50,
    #edge_threshold=0.35,
    rng_seed=123,
    #rng_seed=186,
    verbose=True,
)

## Summarize Step 3 results in a tidy DataFrame
import pandas as pd
import numpy as np

metrics = ["degree", "strength", "betweenness", "closeness", "eigenvector", "clustering"]
rows = []

for node, res in step3_results.items():
    pvals = res.get("p_values", {})
    zscores = res.get("z_scores", {})

    for m in metrics:
        row = {
            "node": node,
            "metric": m,
            "z": zscores.get(f"{m}_z", np.nan),
            "p_one_sided": pvals.get(f"{m}_p", np.nan),             # legacy one-sided
            "p_upper": pvals.get(f"{m}_p_upper", np.nan),           # P(null >= obs)
            "p_lower": pvals.get(f"{m}_p_lower", np.nan),           # P(null <= obs)
            "p_two_sided": pvals.get(f"{m}_p_two_sided", np.nan),   # symmetric 2-sided
        }
        rows.append(row)

step3_stats_df = pd.DataFrame(rows)

print(step3_stats_df)


# Node role classification by centrality space
node_metrics_no_singletons=node_metrics_weighted(G_consensus_cosine_no_singletons, weighted=True, degree_normalize=True)
print(node_metrics_no_singletons)


# NGI: Cluster and node level generalism/specialism metrics
# 1) Geometry-only per-cluster input from raw df
cluster_input = get_cluster_score_input(
    df_all_training_testing_no_singletons,
    node_col="curated_source_region",
    cluster_col="HC50",
)
print(cluster_input)

# 2) Add combined geometry + enrichment specialism indices per HC50
cluster_scores_en = add_cluster_generalism_scores(
    df_all_training_testing_no_singletons,
    cluster_input,
    node_col="curated_source_region",
    cluster_col="HC50",
    lambda_geom=0.5,   # geometry vs enrichment weight
    min_entropy=0.0,
)
print(cluster_scores_en)

# 3) Compute node-level specialism/generalism indices
node_scores_en = compute_node_specialism_generalism(
    df=df_all_training_testing_no_singletons,
    cluster_scores=cluster_scores_en,
    node_col="curated_source_region",
    cluster_col="HC50",
    use_total=True,
    min_node_size=300,
    min_cluster_size=10,
    top_HC_pct=[1, 5, 10, 25, 50],
)
print(node_scores_en)
