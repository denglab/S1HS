import pandas as pd
import ast
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
import networkx as nx
from tqdm import tqdm
import community.community_louvain as community_louvain

np.random.seed(42)

def compute_similarity(data_matrix, similarity):
    if similarity == 'cosine':
        sim_array = cosine_similarity(data_matrix)
    elif similarity == 'pearson':
        sim_array = np.corrcoef(data_matrix)
    elif similarity == 'jaccard':
        binary_matrix = (data_matrix > 0).astype(int)
        dist_matrix = pdist(binary_matrix, metric='jaccard')
        sim_array = 1 - squareform(dist_matrix)
    elif similarity == 'braycurtis':
        dist_matrix = pdist(data_matrix, metric='braycurtis')
        sim_array = 1 - squareform(dist_matrix)
    else:
        raise ValueError(f"Unsupported similarity: {similarity}")
    return sim_array

def detect_communities_leiden(G):
    import leidenalg
    import igraph as ig
    if G.number_of_edges() == 0:
        return {node: i for i, node in enumerate(G.nodes())}, 0.0
    mapping = dict(zip(G.nodes(), range(len(G.nodes()))))
    inv_mapping = {v: k for k, v in mapping.items()}
    edges = [(mapping[u], mapping[v]) for u, v in G.edges()]
    weights = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
    # Preserve isolated NetworkX nodes in the igraph representation.
    ig_graph = ig.Graph(n=len(mapping), edges=edges, directed=False)
    ig_graph.es['weight'] = weights
    partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition, weights='weight')
    score = partition.modularity
    part_dict = {inv_mapping[i]: comm for i, comm in enumerate(partition.membership)}
    return part_dict, score

def detect_communities_louvain(G):
        if G.number_of_edges() == 0:
            return {node: i for i, node in enumerate(G.nodes())}, 0.0
        partition = community_louvain.best_partition(G, resolution=1.0)
        score = community_louvain.modularity(partition, G)
        return partition, score

def compute_coassignment_matrix_by_bootstraps(Graphs):
    """
    Compute a co-assignment matrix where entry (i, j) represents the
    proportion of bootstrap graphs in which node i and node j are connected
    by an edge.

    Parameters:
    -----------
    Graphs : list of networkx.Graph
        A list of undirected graphs (bootstrapped replicates), all with the same set of nodes.

    Returns:
    --------
    co_matrix : np.ndarray
        A 2D numpy array (n_nodes x n_nodes) where each entry (i, j) is the
        fraction of graphs where nodes i and j are connected.
    node_order : list
        The list of node labels corresponding to the rows/columns of the matrix.
    co_df : pandas.DataFrame
        A DataFrame representation of the co-assignment matrix, indexed and
        labeled by node names.
    """
    if not Graphs:
        raise ValueError("Graphs list is empty.")

    # Get consistent node ordering from the first graph
    node_order = list(Graphs[0].nodes)
    n_nodes = len(node_order)
    node_to_idx = {node: i for i, node in enumerate(node_order)}

    # Initialize the co-assignment matrix
    coassign_matrix_bootstraps = np.zeros((n_nodes, n_nodes), dtype=float)

    for G in Graphs:
        for u, v in G.edges():
            if u in node_to_idx and v in node_to_idx:
                i, j = node_to_idx[u], node_to_idx[v]
                coassign_matrix_bootstraps[i, j] += 1
                coassign_matrix_bootstraps[j, i] += 1  # Symmetric

    # Normalize by number of graphs to get proportions
    coassign_matrix_bootstraps /= len(Graphs)

    # Create DataFrame
    coassign_matrix_bootstraps_df = pd.DataFrame(coassign_matrix_bootstraps, index=node_order, columns=node_order)

    return coassign_matrix_bootstraps_df


def compute_coassignment_matrix_by_partitions(partitions, labels):
    '''
    Takes:  
            partitions: A list of community assignments (dictionaries mapping each node to a community ID). Each dictionary represents one run of a clustering or community detection algorithm.
            labels: A list of all node labels, used to maintain consistent ordering in the matrix.
    Outputs:  
            A symmetric matrix (NumPy array) where entry (i, j) is the fraction of times that node i and node j were assigned to the same community across all partitions.
    '''
    from collections import defaultdict
    from itertools import combinations
    label_idx = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    coassign_matrix_partitions = np.zeros((n, n))
    for part in partitions:
        comm_to_nodes = defaultdict(list)
        for node, comm in part.items():
            comm_to_nodes[comm].append(node)
        for group in comm_to_nodes.values():
            for i, j in combinations(group, 2):
                idx_i = label_idx[i]
                idx_j = label_idx[j]
                coassign_matrix_partitions[idx_i, idx_j] += 1
                coassign_matrix_partitions[idx_j, idx_i] += 1
    coassign_matrix_partitions = coassign_matrix_partitions / len(partitions)
    np.fill_diagonal(coassign_matrix_partitions, 1.0)
    coassign_matrix_partitions_df = pd.DataFrame(coassign_matrix_partitions, index=labels, columns=labels)
    return coassign_matrix_partitions, coassign_matrix_partitions_df

def consensus_partition_from_coassign(coassign_matrix_partitions, labels, threshold=0.5):
    """
    Derive a consensus partition from a co-assignment matrix using hierarchical clustering.

    Parameters:
        coassign_matrix (np.ndarray): Square symmetric matrix where [i,j] indicates 
            the fraction of times labels i and j were co-assigned.
        labels (list): List of node or sample identifiers, same order as matrix rows/columns.
        threshold (float): Dissimilarity threshold for cutting the dendrogram (0 to 1). 
        Higher threshold → more clusters

    Returns:
        dict: Mapping of label -> consensus cluster ID
    """
    from scipy.spatial.distance import squareform
    dist_matrix = 1 - coassign_matrix_partitions
    Z = linkage(squareform(dist_matrix), method='average')
    clusters = fcluster(Z, t=1 - threshold, criterion='distance')
    return {label: cluster_id for label, cluster_id in zip(labels, clusters)}

def create_consensus_graph(coassign_matrix_bootstraps_df, threshold=0.6):
    """
    Create a consensus graph from a co-assignment matrix.

    Args:
        coassign_matrix_df (pd.DataFrame): A square DataFrame where (i, j) is the fraction of partitions
                                           in which nodes i and j were co-assigned to the same community.
        threshold (float): Minimum co-assignment frequency to consider an edge (default is 0.6).

    Returns:
        G (networkx.Graph): An undirected graph where edges exist between nodes that were co-assigned
                            in more than `threshold` fraction of partitions.
    """
    G_consensus = nx.Graph()
    
    # Add all nodes
    for node in coassign_matrix_bootstraps_df.index:
        G_consensus.add_node(node)

    # Add edges where co-assignment score exceeds threshold
    for i, node_i in enumerate(coassign_matrix_bootstraps_df.index):
        for j in range(i + 1, len(coassign_matrix_bootstraps_df.columns)):
            node_j = coassign_matrix_bootstraps_df.columns[j]
            weight = coassign_matrix_bootstraps_df.iloc[i, j]
            if weight >= threshold:
                G_consensus.add_edge(node_i, node_j, weight=weight)

    return G_consensus

def build_similarity_network(df, feature_col, class_col,
                             similarity='cosine', sim_threshold=0.30,
                             binary=False, remove_singletons=False,
                             normalize=False,
                             layout='spring', k=0.7,  # layout and k for spring layout
                             bootstrap_iters=100,
                             bootstrapping_strategy='stratified', # 'balanced_min', or 'n_samples_X' where X is the number of samples to bootstrap from each class
                             community_detection_method='louvain',
                             consensus_partition_threshold=0.6, # When using the threshold to compute consensus patition, higher threshold means more clusters
                             consensus_graph_threshold=0.5, # Use threshold to create a consensus graph from the co-assignment matrix; add edges only if the co-assignment score is above this threshold
                             G_type='consensus',  # which graph to plot if doing bootstrapping, 'consensus' or 'mean'
                             show=False,): # show network graph and co-assignment heatmap or not
                             #G_type='G_consensus'):  # which graph to plot if doing bootstrapping, 'G_consensus' or 'G_mean'
    '''
    This fuction returns 13 variables:
        1)  Graphs: When doing bootstrapping, a list of graphs from all bootstrapping iterations. When iter=0, it collects one graph.
        2)  G_mean: the graph based on the mean similarity matrix across all bootstrapping iterations, when iter > 0; returns None if no bootstrapping
        3)  G_consensus: When doing bootstrapping, it is the graph based on the co-assignment matrix by bootstrapping using the function create_consensus_graph() with coassign_matrix_bootstraps_df as the input. when iter > 0; returns None if no bootstrapping
        4)  sim_matrix: a similarity matrix by compute_similarity(). If doing bootstrapping, it is the mean similarity matrix across all bootstrapping iterations
        5)  feature_matrix: a pivot table of the df, showing counts of HC features
        6)  partitions: a list of partitions from all bootstrapping iterations or a single partition if not doing bootstrapping.
        7)  partition_consensus: the consensus of all partitions computed by consensus_partition_from_coassign()
        8)  moduarity_scores: a list of the modularity scores of all bootstrapping iterations, when iter=0, collects one modularity score
        9)  modularity_score_mean: the mean of modularity scores of all bootstrapping iterations if doing bootstrapping; returns None if no bootstrapping
        10) modularity_score_consensus: the modularity score of G_consensus and partition_consensus; returns None if no bootstrapping
        11) classes: all node names
        12) coassign_matrix_partitions_df: a coassign_matrix by partitions, computed by compute_coassignment_matrix_by_partitions()
        13) coassign_matrix_bootstraps_df: a coassign_matrix by bootstraps, computed by compute_coassignment_matrix_by_bootstraps()
    '''

    print(feature_col)
    print("similarity: ", similarity)
    print("similarity_threshold: ", sim_threshold)
    print("consensus_partition_threshold: ",consensus_partition_threshold)
    print("consensus_graph_threshold: ", consensus_graph_threshold)
    print("bootstrap: ", str(bootstrap_iters))
    print("bootstrapping_strategy: ", bootstrapping_strategy)
    print("community_detection_method: ", community_detection_method)

    classes = df[class_col].unique().tolist()
    sim_matrices = []
    partitions = []
    modularity_scores = []
    Graphs = []

    if community_detection_method == 'louvain':
        community_func = detect_communities_louvain
    elif community_detection_method == 'leiden':
        community_func = detect_communities_leiden
    elif community_detection_method == 'infomap':
        community_func = detect_communities_infomap
    else:
        raise ValueError(f"Unsupported community_detection_method: {community_detection_method}")
   

    if bootstrap_iters > 0:
        for _ in tqdm(range(bootstrap_iters), desc="Bootstrapping"):
            boot_samples = []
            for cls in classes:
                class_data = df[df[class_col] == cls]
                if bootstrapping_strategy == 'stratified':
                    n_samples = len(class_data)
                elif bootstrapping_strategy == 'balanced_min':
                    n_samples = df.groupby(class_col).size().min()
                else:
                    try:
                        n_samples = int(bootstrapping_strategy.split('_')[-1])
                    except:
                        raise ValueError(f"Invalid bootstrapping_strategy: {bootstrapping_strategy}")
                boot_cls = class_data.sample(n=n_samples, replace=True)
                boot_samples.append(boot_cls)

            boot_df = pd.concat(boot_samples, ignore_index=True)
            feature_matrix_boot = pd.pivot_table(
                boot_df, index=class_col, columns=feature_col,
                aggfunc='size', fill_value=0).reindex(classes).fillna(0)

            if remove_singletons:
                feature_counts = (feature_matrix_boot > 0).sum(axis=0)
                feature_matrix_boot = feature_matrix_boot.loc[:, feature_counts > 1]

            data_matrix_boot = feature_matrix_boot.values
            if binary:
                data_matrix_boot = (data_matrix_boot > 0).astype(int)
            if normalize:
                row_sums = data_matrix_boot.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1
                data_matrix_boot = data_matrix_boot / row_sums

            sim_array = compute_similarity(data_matrix_boot, similarity)
            sim_matrices.append(sim_array)
            sim_matrix = pd.DataFrame(sim_array, index=classes, columns=classes)

            G = nx.Graph()
            G.add_nodes_from(classes)
            for i in sim_matrix.columns:
                for j in sim_matrix.columns:
                    if i != j and sim_matrix.loc[i, j] > sim_threshold:
                        G.add_edge(i, j, weight=sim_matrix.loc[i, j])
            Graphs.append(G)

            partition, modularity = community_func(G)
            partitions.append(partition)
            modularity_scores.append(modularity if modularity is not None else 0)
            
        # Compute co-assignment matrix by bootstraps
        coassign_matrix_bootstraps_df = compute_coassignment_matrix_by_bootstraps(Graphs)
           
        # Compute co-assignment matrix by partitions 
        coassign_matrix_partitions = compute_coassignment_matrix_by_partitions(partitions, classes)[0]
        coassign_matrix_partitions_df = compute_coassignment_matrix_by_partitions(partitions, classes)[1]

        sim_array = np.mean(sim_matrices, axis=0) # compute the mean similarity matrix across all bootstrapping iterations
        sim_matrix = pd.DataFrame(sim_array, index=classes, columns=classes)
        
        # Create a mean graph based on the mean similarity matrix
        G_mean = nx.Graph()
        G_mean.add_nodes_from(classes)
        for i in sim_matrix.columns:
            for j in sim_matrix.columns:
                if i != j and sim_matrix.loc[i, j] > sim_threshold:
                    G_mean.add_edge(i, j, weight=sim_matrix.loc[i, j])
        
        # Create a consensus graph from the co-assignment matrix by bootstraps
        G_consensus = create_consensus_graph(coassign_matrix_bootstraps_df, threshold=consensus_graph_threshold)
        

        feature_matrix = pd.pivot_table(df, index=class_col, columns=feature_col, aggfunc='size', fill_value=0)
        if remove_singletons:
            feature_counts = (feature_matrix > 0).sum(axis=0)
            feature_matrix = feature_matrix.loc[:, feature_counts > 1]
            
        # create a consensus partition from the coassign matrix
        partition_consensus = consensus_partition_from_coassign(coassign_matrix_partitions, classes, threshold=consensus_partition_threshold)
        modularity_score_mean = np.mean(modularity_scores)
        # partition = partitions[-1] # If bootstrapping, partition is the one from the last iteration
 
        # Compute consensus partition modularity    
        from collections import defaultdict
        # Convert dict -> list of communities
        def partition_dict_to_nodelist(partition_dict):
            nodelist = defaultdict(list)
            for node, community_id in partition_dict.items():
                nodelist[community_id].append(node)
            return list(nodelist.values())

        nodelist = partition_dict_to_nodelist(partition_consensus)
        # Network construction is unchanged; an edgeless consensus graph simply has modularity 0.
        if G_consensus.number_of_edges() == 0:
            modularity_score_consensus = 0.0
        else:
            modularity_score_consensus = nx.algorithms.community.quality.modularity(
                G_consensus, nodelist, weight='weight'
            )
     
    else:
        G_mean = None
        G_consensus = None
        partition_consensus = None
        modularity_score_mean = None
        modularity_score_consensus = None
        coassign_matrix_partitions_df = None
        coassign_matrix_bootstraps_df = None
        
        
        feature_matrix = pd.pivot_table(
            df, index=class_col, columns=feature_col,
            aggfunc='size', fill_value=0).reindex(classes).fillna(0)

        if remove_singletons:
            feature_counts = (feature_matrix > 0).sum(axis=0)
            feature_matrix = feature_matrix.loc[:, feature_counts > 1]

        data_matrix = feature_matrix.values
        if binary:
            data_matrix = (data_matrix > 0).astype(int)
        if normalize:
            row_sums = data_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            data_matrix = data_matrix / row_sums

        sim_array = compute_similarity(data_matrix, similarity)
        sim_matrix = pd.DataFrame(sim_array, index=classes, columns=classes)

        G = nx.Graph()
        G.add_nodes_from(classes)
        for i in sim_matrix.columns:
            for j in sim_matrix.columns:
                if i != j and sim_matrix.loc[i, j] > sim_threshold:
                    G.add_edge(i, j, weight=sim_matrix.loc[i, j])
        
        Graphs.append(G)

        partition, modularity_score = community_func(G)
        partitions = [partition]
        modularity_scores = [modularity_score if modularity_score is not None else 0]
    return Graphs, G_mean, G_consensus, sim_matrix, feature_matrix, partitions, partition_consensus, modularity_scores, modularity_score_mean, modularity_score_consensus, classes,  coassign_matrix_partitions_df, coassign_matrix_bootstraps_df

def evaluate_parameters(func, param_grid, report_params=None):
    from itertools import product
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))

    results = []
    for combo in combinations:
        params = dict(zip(keys, combo))

        if report_params is None:
            reported_params = params
        else:
            reported_params = {k: params[k] for k in report_params if k in params}
        
        try:
            output = func(**params)
            # Unpack assuming the return signature matches your network builder
            Graphs, G_mean, G_consensus, sim_matrix, feature_matrix, partitions, partition_consensus, modularity_scores, modularity_score_mean, modularity_score_consensus, classes,  coassign_matrix_partitions_df, coassign_matrix_bootstraps_df= output
            result = {
                'parameters': reported_params,
                'modularity_score_mean': float(round(modularity_score_mean, 3)),
                'modularity_score_consensus': float(round(modularity_score_consensus, 3)),
                'connectivity_summary': graph_connectivity_summary(G_consensus)
            }
        except Exception as e:
            # Keep a consistent result schema so failed parameter combinations do not
            # make the final DataFrame construction fail.
            result = {
                'parameters': reported_params,
                'modularity_score_mean': None,
                'modularity_score_consensus': None,
                'connectivity_summary': None,
                'error': str(e)
            }

        results.append(result)

    return results

def graph_connectivity_summary(G):
    """
    Returns a tuple with:
    - number of unconnected nodes (isolated),
    - number of connected nodes (non-isolated),
    - fraction of connected nodes.
    
    Parameters:
        G (networkx.Graph): The input network graph.
    
    Returns:
        tuple: (num_unconnected, num_connected, fraction_connected)
    """
    total_nodes = G.number_of_nodes()
    unconnected_nodes = list(nx.isolates(G))
    num_unconnected = len(unconnected_nodes)
    num_connected = total_nodes - num_unconnected
    fraction_connected = num_connected / total_nodes if total_nodes > 0 else 0.0
    
    return num_unconnected, num_connected, fraction_connected


# Load the same input data used in the original script.
df_all_training_testing = pd.read_csv(
    '../data/metadata_training_testing.csv.gz',
    low_memory=False
)

param_grid = {
    'df': [df_all_training_testing],
    'feature_col': ['HC20','HC50','HC100'], # n=3
    'class_col': ['curated_source_region'],
    'similarity': ['cosine', 'jaccard', 'braycurtis', 'pearson'], # n=4
    'sim_threshold': [0.25,0.30,0.35,0.40,0.45,0.50], # n=6
    'consensus_partition_threshold':[0.1, 0.5, 0.9], # n=3
    'consensus_graph_threshold': [0.5, 0.6, 0.7, 0.8, 0.9], # n=4
    'bootstrap_iters': [100],
    'bootstrapping_strategy': ['stratified', 'balanced_min'], # n=2 num=samples to bootstrap from each class
    'community_detection_method': ['louvain','leiden'], #n=2 infomap does not provide modularity score
}

results = evaluate_parameters(build_similarity_network, param_grid, 
                              report_params=['feature_col','similarity','sim_threshold','consensus_partition_threshold','consensus_graph_threshold','bootstrapping_strategy','community_detection_method'])
for res in results:
    print(res)

# Convert to DataFrame
df_gridsearch = pd.DataFrame(results)

# Extract parameters into their own columns
param_df = df_gridsearch['parameters'].apply(pd.Series)

# Expand connectivity summary into separate columns
connectivity_df = df_gridsearch['connectivity_summary'].apply(
    lambda x: pd.Series(
        x if isinstance(x, (tuple, list)) and len(x) == 3 else [None, None, None],
        index=['isolated_nodes', 'connected_nodes', '% connected_nodes']
    )
)

# Combine all into a single DataFrame
final_df = pd.concat([param_df, df_gridsearch[['modularity_score_mean', 'modularity_score_consensus']], connectivity_df], axis=1)

# Optional: Sort by modularity score
final_df_sorted = final_df.sort_values(by='modularity_score_consensus', ascending=False)

final_df_sorted.to_csv('gridsearch_results.csv', index=False)
