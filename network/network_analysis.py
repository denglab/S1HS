import os
from network_funcs import *

current_path = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(current_path)
DATA_PATH = os.path.join(SRC_ROOT, "data")

df_all_training_testing = pd.read_csv(os.path.join(DATA_PATH, "metadata_training_testing.csv"))

## Single cosine similarity network
G_original_single_cosine, G_mean_single_cosine, G_consensus_single_cosine, sim_matrix_single_cosine, feature_matrix_single_cosine, partitions_single_cosine, partition_consensus_single_cosine, modularity_scores_single_cosine, modularity_score_mean_single_cosine, modularity_score_consensus_single_cosine, classes, coassign_matrix_partitions_df_single_cosine, coassign_matrix_bootstraps_df_single_cosine= build_similarity_network(
    df_all_training_testing, feature_col='HC50', class_col='curated_source_region',
    similarity='cosine',
    sim_threshold=0.35,
    community_detection_method='leiden', # 'louvain', 'leiden', 'infomap'
    consensus_partition_threshold=None, # controls how many communities are detected using the consensus partition of all bootstrapping iterations. The lower, the fewer communities
    consensus_graph_threshold=None,
    bootstrap_iters=0,
    bootstrapping_strategy='stratified',
    show=True,
)

# find totoal connected nodes and numberf of edges in G_original_single_cosine
num_nodes = G_original_single_cosine[0].number_of_nodes()
num_edges = G_original_single_cosine[0].number_of_edges()
print(f"G_original_single_cosine: {num_nodes} nodes, {num_edges} edges")

node_metrics_single_cosine = node_metrics_weighted(G_original_single_cosine[0], weighted=True, degree_normalize=True)
print(f"Node metrics: {node_metrics_single_cosine}")

### Remove disconnected nodes
result_summary_disconnected_nodes, G_single_cosine_no_disconnected_nodes, disconnected_nodes_single = remove_disconnected_nodes(G_original_single_cosine[0],communities=partitions_single_cosine[0], weight="weight")
print(f"Disconnected nodes: {disconnected_nodes_single}")

plot_community_network(G_single_cosine_no_disconnected_nodes, partitions_single_cosine[0], df_all_training_testing, layout='sfdp', layout_params={'K': 1.0, 'repulsiveforce':10,'overlap': False},  # singleton are defined as size=1 communities, not unconnected nodes
                         modularity_score=None, cliques=None, size_attribute='HC_count')

#### Remove singleton coommunities 
result_summary_single, G_single_cosine_no_singletons, partitions_single_cosine_no_singletons, singleton_nodes_single = remove_singleton_communities(G_original_single_cosine[0], partitions_single_cosine[0], weight='weight')
print(f"Singleton nodes: {singleton_nodes_single}")

plot_community_network(G_single_cosine_no_singletons, partitions_single_cosine_no_singletons, df_all_training_testing, layout='sfdp', layout_params={'K': 1.0, 'repulsiveforce':10,'overlap': False},  # singleton are defined as size=1 communities, not unconnected nodes
                         modularity_score=None, cliques=None, size_attribute='HC_count')


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

# for each row in featre_matrix_cosine, count the number of non-zero values and compute the percentage of zero values in each row, ouput results in a dataframe with columns 'non_zero_count' and 'zero_percentage'
feature_matrix_cosine_summary = feature_matrix_cosine.apply(lambda row: pd.Series({
    'non_zero_count': (row != 0).sum(),
    'zero_percentage': (row == 0).mean() * 100
}), axis=1)
print(feature_matrix_cosine_summary)

## Remove disconnected nodes (not singleton communities)
result_summary_disconnected_nodes, G_consensus_cosine_no_disconnected_nodes, disconnected_nodes_consensus = remove_disconnected_nodes(G_consensus_cosine, partition_consensus_cosine, weight='weight')
print(f"Disconnected nodes: {disconnected_nodes_consensus}")

plot_community_network(G_consensus_cosine_no_disconnected_nodes, partition_consensus_cosine,df_all_training_testing, layout='sfdp', layout_params={'K': 1, 'repulsiveforce':2,'overlap': False},  # singleton are defined as size=1 communities, not unconnected nodes
                         modularity_score=None, cliques=None, size_attribute='HC_count')

### Remove singleton communities 
result_summary_no_singltons, G_consensus_cosine_no_singletons, partition_consensus_cosine_no_singletons, singleton_nodes_consensus = remove_singleton_communities(G_consensus_cosine, partition_consensus_cosine, weight='weight')

### Renumber communities using anchor nodes, so community designations are consistent
partition_consensus_cosine_no_singletons_renumbered, old2new = renumber_communities(partition_consensus_cosine_no_singletons)
print(partition_consensus_cosine_no_singletons_renumbered)

#### Final network plot with sfdp layout
selected_edges=[("fruit_and_vegetable","water_appalachian"), ("herbs","water_appalachian"), ("herbs","production_env_produce"),
                ("production_env_produce","water_northeast"),("production_env_produce","soil_northeast"), 
                ("soil_northeast","water_northeast"), ("production_env_produce","water_appalachian"),
                ("animal_feed", "companion_animal"), ("animal_feed", "equine"), ("animal_feed", "poultry"),("animal_feed","swine"),
                ("dairy","bovine"),("poultry","multi_ingredient_poultry"),
                ("companion_animal","human"), ("equine","human"), 
                ("human","rte"),
                ("human","produce_outbreak"),
                ("aquatic_animal","water_northeast"),
                ("grains", "water_northeast"),
                ]
plot_community_network_selected_edges(G_consensus_cosine_no_singletons, partition_consensus_cosine_no_singletons_renumbered, df_all_training_testing, figsize=(13, 9), layout='sfdp', layout_params={'K': 2, 'repulsiveforce':2.5,'overlap': False}, seed=GLOBAL_SEED, selected_edges=selected_edges, selected_edge_color="#F54927",
                         modularity_score=None, cliques=None, node_lables=node_labels,size_attribute='HC_count', node_edge_width=0, color_scheme={'Community 1': '#dd6670', 'Community 2': '#edae93','Community 3': '#93b071','Community 4': '#ede2cc','Community 5': '#598f91','Community 6': '#88a2c4','Community 7': '#14698D'})


### Evaluate modularity/partition significance (Inverse-normal z score)
z, q_obs, null_q = evaluate_partition_significance_modularity(
    G_consensus_cosine_no_singletons,
    partition_consensus_cosine_no_singletons,
    n_iter=1000,
    score_type="z",       # inverse-normal z (bounded by Monte Carlo resolution)
    weight="weight",
    resolution=1.0,
    seed=42,
    community_detection="leiden",  # 'louvain' or 'leiden'
)

### cosine similarity and coassignment
from collections import Counter
order = coassign_matrix_bootstraps_df_cosine.index
coassign = coassign_matrix_bootstraps_df_cosine.loc[order, order]
sim = sim_matrix_single_cosine.loc[order, order]
pairs = []
n = len(order)
for i in range(n):
    for j in range(i + 1, n):
        if sim.iat[i, j] > 0.35 and coassign.iat[i, j] < 0.5:
            pairs.append((order[i], order[j], sim.iat[i, j], coassign.iat[i, j]))
# print all pairs
for a, b, s, c in pairs:
    print(a, b, "cosine=", s, "coassign=", c)

# rank nodes by frequency in those pairs
counts = Counter()
for a, b, *_ in pairs:
    counts[a] += 1
    counts[b] += 1
# print ranked nodes
for node, cnt in counts.most_common():
    print(node, cnt)

#### Node metrics using graph with singleton communities removed
node_metrics_no_singletons=node_metrics_weighted(G_consensus_cosine_no_singletons)
print(node_metrics_no_singletons)

#### Node metrics using graph with disconnected nodes removed
node_metrics_no_disconnected_nodes = node_metrics_weighted(G_consensus_cosine_no_disconnected_nodes)
print(node_metrics_no_disconnected_nodes)

## Node classification by clustering of centrality metrics (heatmap)
node_metrics_with_roles = add_node_role(node_metrics_no_singletons)
node_metrics_with_roles = node_metrics_with_roles.reset_index().rename(columns={"index": "node"})
print(node_metrics_with_roles)

import seaborn as sns
import matplotlib.pyplot as plt
# pick the numeric metric columns you want to cluster
cols = ["eigenvector", "betweenness", "degree", "strength", "closeness", "clustering"]
data = node_metrics_no_singletons[cols]
g = sns.clustermap(
    data,
    method="complete",
    metric="euclidean",
    cmap="rocket_r",
    figsize=(6, 12),
    linewidths=0.5,
    linecolor="white"
)
plt.show()


### Allelic distance matrix
df_distance_matrix = pd.read_csv(os.path.join(DATA_PATH, "HC50_distance_matrix.csv"), index_col=0)
df_distance_matrix.index = df_distance_matrix.index.astype(float)
df_distance_matrix.columns = df_distance_matrix.columns.astype(float)

tree = build_nj_tree(df_distance_matrix)
print(tree)

alpha_df = compute_full_alpha_diversity(feature_matrix_cosine, tree)
print(alpha_df)

# add to aplha_df a column 'Obs/Chao1' which is the ratio of 'Observed_HCs' to 'Chao1'
alpha_df['Obs/Chao1'] = alpha_df['Observed_HCs'] / alpha_df['Chao1']

# rank Obseved_HC/Isolate_N and add as a column 'Richness_per_Isolate'
alpha_df['Richness_per_Isolate'] = alpha_df['Observed_HCs'] / alpha_df['Isolate_N']
alpha_df[alpha_df['Isolate_N'] > 100].sort_values('Richness_per_Isolate', ascending=False)
print(alpha_df)

### Correlation between node centrality metrics and node characteristics
# add the HC50 count to the node metrics DataFrame
node_metrics_df = node_metrics_weighted(G_consensus_cosine_no_singletons, weighted=True, degree_normalize=True)
HC50_count_df= count_HC_clusters(df_all_training_testing, HC_col='HC50', source_col='curated_source_region')
HC50_count_df=HC50_count_df.set_index('curated_source_region')
node_metrics_df = HC50_count_df.join(node_metrics_df)
node_metrics_df = alpha_df.join(node_metrics_df)
node_metrics_df

### Only moderate correlations between node centrality metrics and distinct HC50 counts (0.33 - 0.46) as well as between alpha diversity metrics and distinct HC50 counts(<= 0.46)
calculate_correlations(node_metrics_df, ['Isolate_N','Faith_PD','Shannon','Simpson','Chao1','Observed_HCs', 'Evenness', 'strength', 'betweenness', 'closeness', 'degree'], correlations=['spearman'])

## Community/Subgraph analyses
community_metrics_df = community_importance(G_consensus_cosine_no_singletons, partition_consensus_cosine_no_singletons_renumbered, density_thr=0.726, external_thr=0.218, weight='weight')
print(community_metrics_df)

### Community meta-graph construction
meta_graph, comm_size = build_metagraph(G_consensus_cosine_no_singletons, partition_consensus_cosine_no_singletons_renumbered, weight='weight')

# Edge characterizaton and classification
## All node pair characterization by cosine similarity and node richness
df_all_node_pairs = build_pairwise_cosine_df_tidy(
    sim_matrix_single_cosine=sim_matrix_single_cosine,
    df_all_training_testing=df_all_training_testing,
    node_col="curated_source_region",
    hc_col="HC50",
    pair_sep="-",
    drop_self_pairs=True,
    upper_triangle_only=True,
    node_name_override=all_node_labels
)
print(df_all_node_pairs)


# Edge direction inference
## Shared source test: vector projection method

### Thresholds:
# max_residual: This is the maximum allowed residual cosine between node_a and node_b after removing the proposed source component. Smaller: stricker.
# base_min_delta: This is the minimum required drop in cosine similarity between node_a and node_b after removing the proposed source component, compared to the original cosine similarity. Bigger: require larger effect. 
# min_fraction: This is the minimum required fraction of the original cosine similarity removed. Bigger: stricker
# min_gap: This is the minimum required gap in cosine similarity reduction between the proposed source and the next best alternative source. Bigger: stricker
# source_richness_top_n: Only consider candidate sources that are ranked in the top n among all nodes in terms of HC50 richness. This helps to avoid spurious inference of common sources that are not well represented in the dataset. 

triangle_result_af_h_p = evaluate_triangle_sources(
    df=df_all_training_testing,
    triangle=("animal_feed", "human", "poultry"),
    run_bootstrap=True,
    n_boot=500,
    random_state=186,
    bootstrap_frac_range=(1.0, 1.0),
    bootstrap_min_total_per_node=1,
    bootstrap_stability_threshold=0.50,
    max_residual=0.35,
    base_min_delta=0.15,
    min_fraction=0.50,
    min_gap=0.05,
    require_size_filter=False,
    source_richness_top_n=10,
    print_summary=True,
)
candidate_df_af_h_p = triangle_result_af_h_p["candidate_results_df"]
summary_df_af_h_p = triangle_result_af_h_p["triangle_summary_df"]
bootstrap_candidate_summary_df_af_h_p = triangle_result_af_h_p["bootstrap_candidate_summary_df"]
bootstrap_triangle_summary_df_af_h_p = triangle_result_af_h_p["bootstrap_triangle_summary_df"]
bootstrap_long_df = triangle_result_af_h_p["bootstrap_long_df"]
print(candidate_df_af_h_p)
print(summary_df_af_h_p)

### In this framework, “source” means the node whose HC50 composition best accounts for the overlap between the other two nodes after projection removal. It should be interpreted as a statistical mediator or explanatory profile, not necessarily as the epidemiologic direction of transmission


## Screening all 3-node cliques in the network for supported common sources
triangles = find_3_node_cliques_fast(G_consensus_cosine_no_singletons)
print(triangles)

### All triangles with at least one major zoonotic source 
triangles_major_zoonotic = find_triangles_with_nodes(triangles, ["animal_feed",
    "poultry",
    "bovine",
    "equine",
    "swine",
    "wild_animal",])
print(triangles_major_zoonotic)

### Overlapping triangles that establsiehd bidirectional edge between poultry and animal feed
# subset specified rows and columns from sim_matrix_single_cosine
subset_nodes = ["animal_feed", "poultry", "companion_animal", "human"]
subset_sim_matrix = sim_matrix_single_cosine.loc[subset_nodes, subset_nodes]
print(subset_sim_matrix)
