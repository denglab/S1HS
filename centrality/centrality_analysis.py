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
