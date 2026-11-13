from __future__ import annotations
from typing import Dict, List, Tuple, Optional

# ============================================
# Network functions
# ============================================
# --- imports ---
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import pearsonr
import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster
import community.community_louvain as community_louvain
#from infomap import Infomap  # Infomap has no modularity score
import random as _py_random

# --- SINGLE global seed for everything except per-iteration bootstraps ---
GLOBAL_SEED = 186
np.random.seed(GLOBAL_SEED)
_py_random.seed(GLOBAL_SEED)

# ---------------- core utilities ----------------
def count_HC_clusters(df, HC_col='HC2', source_col='curated_source_region'):
    """
    Count distinct HC clusters in each curated source.
    
    Parameters:
    - df: pandas DataFrame containing the data
    - HC_col: str, column name for HC clusters (default is 'HC2')
    - source_col: str, column name for sources (default is 'curated_source_region')
    
    Returns:
    - pandas DataFrame with counts of distinct HC clusters per source
    """
    count_col = f'Distinct {HC_col} Clusters'
    return df.groupby(source_col)[HC_col].nunique().reset_index(name=count_col).sort_values(count_col, ascending=False)

def compute_similarity(data_matrix, similarity):
    X = np.asarray(data_matrix)
    if similarity == 'cosine':
        sim_array = cosine_similarity(X)
    elif similarity == 'pearson':
        sim_array = np.corrcoef(X)
    elif similarity in ('jaccard_binary', 'jaccard'):
        binary_matrix = (X > 0).astype(int)
        dist_matrix = pdist(binary_matrix, metric='jaccard')
        sim_array = 1.0 - squareform(dist_matrix)
        np.fill_diagonal(sim_array, 1.0)
    elif similarity == 'jaccard_count':
        if np.any(X < 0):
            raise ValueError("jaccard_count requires nonnegative counts.")
        row_sums = X.sum(axis=1, dtype=float)
        l1 = cdist(X, X, metric='cityblock').astype(float)
        num = 0.5 * (row_sums[:, None] + row_sums[None, :] - l1)
        den = 0.5 * (row_sums[:, None] + row_sums[None, :] + l1)
        sim_array = np.divide(num, den, out=np.zeros_like(num), where=(den != 0))
        both_zero = (den == 0)
        sim_array[both_zero] = 1.0
        np.fill_diagonal(sim_array, 1.0)
    elif similarity == 'braycurtis':
        dist_matrix = pdist(X, metric='braycurtis')
        sim_array = 1.0 - squareform(dist_matrix)
        np.fill_diagonal(sim_array, 1.0)
    else:
        raise ValueError(f"Unsupported similarity: {similarity}")
    return sim_array



# ---------------- graph plotting ----------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import networkx as nx
import seaborn as sns


def _compute_layout_pos(G, layout, seed=None, layout_params=None):
    """
    Compute node positions for a given layout.

    layout : str | callable
      - "spring"         (nx.spring_layout)
      - "kamada_kawai"   (nx.kamada_kawai_layout)
      - "circular"       (nx.circular_layout)
      - "shell"          (nx.shell_layout)
      - "spectral"       (nx.spectral_layout)
      - "random"         (nx.random_layout)
      - "planar"         (nx.planar_layout)
      - "sfdp"           (Graphviz sfdp via graphviz_layout)   <-- no custom wrapper
      - "graphviz"       (Graphviz via graphviz_layout; choose prog in layout_params)

    layout_params : dict
      spring: k, iterations, scale, center, dim, weight
      ... (etc)
      sfdp / graphviz:
        - prog: e.g. "sfdp", "neato", "dot", "fdp", "twopi"
        - args: extra graphviz command args string (optional)
        - root: root node for some progs (optional)
        - graph_attrs: dict of graph attributes passed to graphviz (recommended)
          OR pass common attrs at top-level (K, repulsiveforce, overlap, ...)
    """
    layout_params = dict(layout_params or {})

    # Allow custom layout function
    if callable(layout):
        return layout(G, **layout_params)

    layout = (layout or "").lower()

    if layout == "spring":
        defaults = {"k": 2.0, "iterations": 300, "scale": 3.0}
        defaults.update(layout_params)
        return nx.spring_layout(G, seed=seed, **defaults)

    if layout == "kamada_kawai":
        defaults = {"scale": 1.0}
        defaults.update(layout_params)
        return nx.kamada_kawai_layout(G, **defaults)

    if layout == "circular":
        defaults = {"scale": 1.0}
        defaults.update(layout_params)
        return nx.circular_layout(G, **defaults)

    if layout == "shell":
        defaults = {"scale": 1.0}
        defaults.update(layout_params)
        return nx.shell_layout(G, **defaults)

    if layout == "spectral":
        defaults = {"scale": 1.0}
        defaults.update(layout_params)
        return nx.spectral_layout(G, **defaults)

    if layout == "random":
        defaults = {"center": None, "dim": 2}
        defaults.update(layout_params)
        return nx.random_layout(G, seed=seed, **defaults)

    if layout == "planar":
        defaults = {"scale": 1.0}
        defaults.update(layout_params)
        return nx.planar_layout(G, **defaults)

    # ---- Graphviz-backed layouts (includes SFDP) ----
    if layout in {"sfdp", "graphviz"}:
        # Choose engine
        prog = layout_params.pop("prog", "sfdp" if layout == "sfdp" else "neato")
        args = layout_params.pop("args", "")
        root = layout_params.pop("root", None)

        # Collect graph attributes (recommended way to tune sfdp)
        graph_attrs = dict(layout_params.pop("graph_attrs", {}) or {})

        # Convenience: accept common SFDP graph attributes at top-level too
        # (Graphviz generally wants these as *strings*)
        common_graphviz_attrs = (
            "K",
            "repulsiveforce",
            "overlap",
            "maxiter",
            "epsilon",
            "start",
            "beautify",
            "normalize",
            "pack",
            "quadtree",
            "levels",
            "mode",
        )
        for k in common_graphviz_attrs:
            if k in layout_params:
                graph_attrs[k] = layout_params.pop(k)

        if layout_params:
            # Anything left is likely a typo/misplaced param
            unknown = ", ".join(sorted(layout_params.keys()))
            raise ValueError(
                f"Unused layout_params for layout='{layout}': {unknown}. "
                f"For graphviz/sfdp, use graph_attrs={{...}}, plus optional prog/args/root."
            )

        # Apply graph attributes on a temporary copy so we don't mutate the original graph
        H = G.copy()
        if graph_attrs:
            H.graph.update({k: str(v) for k, v in graph_attrs.items()})

        # Try pygraphviz first, then pydot
        try:
            from networkx.drawing.nx_agraph import graphviz_layout
            return graphviz_layout(H, prog=prog, root=root, args=args)
        except Exception:
            try:
                from networkx.drawing.nx_pydot import graphviz_layout
                return graphviz_layout(H, prog=prog, root=root)  # pydot version may ignore args
            except Exception as e:
                raise ImportError(
                    "Graphviz layouts require Graphviz + (pygraphviz or pydot) installed."
                ) from e

    raise ValueError(f"Unsupported layout: {layout}")


def plot_community_network(
    G,
    communities,
    df,
    *,
    layout="sfdp", # or "spring", "kamada_kawai", "circular", "shell", "spectral", "random", "planar"
    layout_params= None, # pass dict of layout-specific params
    modularity_score=None,
    cliques=None,
    size_attribute="HC_count",
    seed=GLOBAL_SEED,
    color_scheme=None,
    figsize=(12, 8),
    dpi=600,
):
    """
    Plot a community-labeled network with node size proportional to HC richness.

    Layout control:
      - Choose layout via `layout=...`
      - Pass layout-specific knobs via `layout_params={...}`

    Examples:
      plot_community_network(G, comms, df, layout="spring",
                             layout_params={"k": 6, "iterations": 800, "scale": 2.0})

      plot_community_network(G, comms, df, layout="sfdp",
                             layout_params={'K': 1.0, 'repulsiveforce':6,'overlap': False})

      plot_community_network(G, comms, df, layout="random",
                             layout_params={"dim": 2})
    """

    # ---- Layout ----
    use_seed = seed if seed is not None else GLOBAL_SEED
    pos = _compute_layout_pos(G, layout=layout, seed=use_seed, layout_params=layout_params)

    # ---- Communities & colors ----
    community_counts = {}
    for node, comm_id in communities.items():
        community_counts[comm_id] = community_counts.get(comm_id, 0) + 1

    significant_comms = {cid for cid, cnt in community_counts.items() if cnt >= 2}
    color_map = {}
    if significant_comms:
        colors = sns.color_palette("hls", len(significant_comms))
        color_map = dict(zip(significant_comms, colors))

    # optional override
    if color_scheme is not None and color_map:
        for comm_id in list(color_map.keys()):
            key_id = comm_id
            key_label = f"Community {comm_id}"
            if key_id in color_scheme:
                color_map[comm_id] = color_scheme[key_id]
            elif key_label in color_scheme:
                color_map[comm_id] = color_scheme[key_label]

    node_colors = [color_map.get(communities[node], "lightgrey") for node in G.nodes()]

    # ---- Node attributes ----
    hc_counts = count_HC_clusters(df, HC_col="HC50", source_col="curated_source_region")
    hc_counts_dict = dict(zip(hc_counts.iloc[:, 0], hc_counts.iloc[:, 1]))

    metrics_df = node_metrics_weighted(G)
    metrics_dict = {col: metrics_df[col].to_dict() for col in metrics_df.columns}

    for node in G.nodes():
        G.nodes[node]["community"] = communities[node]
        G.nodes[node]["HC_count"] = hc_counts_dict.get(node, 0)
        for metric in ["betweenness", "closeness", "degree", "eigenvector", "clustering"]:
            G.nodes[node][metric] = metrics_dict.get(metric, {}).get(node, 0)

    # ---- Node sizes ----
    size_values = []
    for node in G.nodes():
        try:
            size_values.append(G.nodes[node][size_attribute])
        except KeyError:
            size_values.append(1)
            print(f"Warning: Node {node} is missing {size_attribute} attribute, using default value 1")

    min_size = min(size_values) if size_values else 1
    max_size = max(size_values) if size_values else 1
    log_sizes = [np.log10(v) if v > 0 else 0 for v in size_values]
    min_log = min(log_sizes) if log_sizes else 0
    max_log = max(log_sizes) if log_sizes else 1
    range_log = max_log - min_log if max_log > min_log else 1

    min_node_size = 10
    max_node_size = 1500
    node_sizes = [
        min_node_size + (max_node_size - min_node_size) * ((lv - min_log) / range_log)
        for lv in log_sizes
    ]

    # ---- Edge widths by weight ----
    weights = []
    for u, v in G.edges():
        try:
            weights.append(G[u][v]["weight"])
        except KeyError:
            weights.append(1)
            print(f"Warning: Edge ({u}, {v}) is missing weight attribute, using default weight 1")

    if weights:
        min_w = min(weights)
        max_w = max(weights)
        range_w = max_w - min_w if max_w > min_w else 1
        edge_widths = [1 + 3 * ((w - min_w) / range_w) for w in weights]
    else:
        edge_widths = [1] * len(G.edges())
        print("Warning: No edge weights found, using default width for all edges")

    # ---- Clique highlighting (optional) ----
    edge_colors, edge_color_map, node_color_map = [], {}, {}
    if cliques:
        for clique_nodes, color in cliques.items():
            cn = list(clique_nodes)
            for i in range(len(cn)):
                for j in range(i + 1, len(cn)):
                    u, v = cn[i], cn[j]
                    if G.has_edge(u, v):
                        edge_color_map[frozenset((u, v))] = color
            for node in cn:
                node_color_map[node] = color

    for u, v in G.edges():
        edge_colors.append(edge_color_map.get(frozenset((u, v)), "lightgrey"))

    # ---- Draw ----
    plt.figure(figsize=figsize, dpi=dpi)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.9)

    nodes = nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8
    )
    nodes.set_edgecolor("grey")

    if node_color_map:
        for color in set(node_color_map.values()):
            nodes_in_clique = [n for n in node_color_map if node_color_map[n] == color]
            clique_sizes = [node_sizes[list(G.nodes()).index(n)] for n in nodes_in_clique]
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=nodes_in_clique,
                node_color="none",
                edgecolors=color,
                linewidths=2.5,
                node_size=[s + 50 for s in clique_sizes],
                alpha=0.8,
            )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=10,
        bbox=dict(facecolor="none", edgecolor="none", alpha=0.7, boxstyle="round,pad=0.2"),
    )

    # Legend
    legend_elements = []
    for comm_id, color in color_map.items():
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=f"Community {comm_id} ({community_counts[comm_id]} nodes)",
                markerfacecolor=color,
                markersize=10,
            )
        )
    legend_elements.append(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"Node size = {size_attribute}\n(Min: {min_size}, Max: {max_size})",
            markerfacecolor="grey",
            markersize=10,
        )
    )
    if legend_elements:
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left")

    title_parts = []
    if modularity_score is not None:
        title_parts.append(f"Modularity: {modularity_score:.3f}")
    title_parts.append(f"Node size: {size_attribute}")
    plt.title("\n".join(title_parts), pad=20)

    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ---------------- community detection algorithms ----------------

def detect_communities_leiden(G, seed=GLOBAL_SEED):
    import leidenalg, igraph as ig
    mapping = dict(zip(G.nodes(), range(len(G.nodes()))))
    inv_mapping = {v: k for k, v in mapping.items()}
    edges = [(mapping[u], mapping[v]) for u, v in G.edges()]
    weights = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
    ig_graph = ig.Graph(edges=edges, directed=False)
    ig_graph.es['weight'] = weights
    partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition, weights='weight', seed=seed)
    score = partition.modularity
    part_dict = {inv_mapping[i]: comm for i, comm in enumerate(partition.membership)}
    return part_dict, score


def detect_communities_louvain(G, seed=GLOBAL_SEED):
    partition = community_louvain.best_partition(G, resolution=1.0, weight='weight', random_state=seed)
    score = community_louvain.modularity(partition, G, weight='weight')
    return partition, score


def detect_communities_infomap(G, seed=GLOBAL_SEED):
    im = Infomap("--two-level --silent", seed=seed)
    node_id_map = {node: i for i, node in enumerate(G.nodes())}
    reverse_node_id_map = {i: node for node, i in node_id_map.items()}
    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 1.0)
        im.addLink(node_id_map[u], node_id_map[v], weight)
    im.run()
    partition = {reverse_node_id_map[node.node_id]: node.module_id for node in im.nodes}
    return partition, None


# ---------------- co-assignment matrices ----------------

def compute_coassignment_matrix_by_bootstraps(Graphs):
    """
    Edge-based co-assignment (unrounded): fraction of bootstrap graphs where an edge exists.
    """
    if not Graphs:
        raise ValueError("Graphs list is empty.")
    node_order = list(Graphs[0].nodes)
    n_nodes = len(node_order)
    node_to_idx = {node: i for i, node in enumerate(node_order)}
    coassign_matrix_bootstraps = np.zeros((n_nodes, n_nodes), dtype=float)
    for G in Graphs:
        for u, v in G.edges():
            if u in node_to_idx and v in node_to_idx:
                i, j = node_to_idx[u], node_to_idx[v]
                coassign_matrix_bootstraps[i, j] += 1
                coassign_matrix_bootstraps[j, i] += 1
    coassign_matrix_bootstraps /= len(Graphs)
    coassign_matrix_bootstraps_df = pd.DataFrame(coassign_matrix_bootstraps, index=node_order, columns=node_order)
    return coassign_matrix_bootstraps_df


def compute_coassignment_matrix_by_partitions(partitions, labels):
    """
    Partition-based co-assignment (unrounded): fraction of runs where two nodes share a community.
    """
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
    Hierarchical clustering on the partition co-assignment matrix (unrounded).
    """
    dist_matrix = 1 - coassign_matrix_partitions
    Z = linkage(squareform(dist_matrix), method='average')
    clusters = fcluster(Z, t=1 - threshold, criterion='distance')
    return {label: cluster_id for label, cluster_id in zip(labels, clusters)}


def create_consensus_graph_unrounded(coassign_matrix_bootstraps_df, threshold=0.5):
    """
    Build consensus graph using UNROUNDED edge co-assignment for both inclusion and weight.
    """
    M = coassign_matrix_bootstraps_df
    G_consensus = nx.Graph()
    for node in M.index:
        G_consensus.add_node(node)
    for i, u in enumerate(M.index):
        for j in range(i + 1, len(M.columns)):
            v = M.columns[j]
            w = float(M.iloc[i, j])
            if w >= threshold:                  # threshold on raw value
                G_consensus.add_edge(u, v, weight=w)  # weight is raw value
    return G_consensus


# ---------------- metrics ----------------

def plot_heatmap_seaborn(matrix, labels, title="Partition co-assignment Matrix Heatmap"):
    df_matrix = pd.DataFrame(matrix, index=labels, columns=labels)
    plt.figure(figsize=(12, 8), dpi=600)
    sns.heatmap(df_matrix, cmap='viridis', square=True, cbar_kws={"label": "Co-assignment Frequency"})
    plt.title(title); plt.xticks(rotation=90); plt.yticks(rotation=0)
    plt.tight_layout(); plt.show()


def node_metrics(G):
    metrics = {
        "degree": nx.degree_centrality(G),
        "betweenness": nx.betweenness_centrality(G),
        "closeness": nx.closeness_centrality(G),
        "eigenvector": nx.eigenvector_centrality(G, max_iter=500),
        "clustering": nx.clustering(G)
    }
    df = pd.DataFrame(metrics)
    return df



def node_metrics_weighted(G, weighted=True, degree_normalize=True):
    """
    Centrality metrics for a weighted similarity network.

    If weighted=True:
      - degree = normalized count of neighbors
      - strength = normalized sum of 'weight' edges
      - betweenness/closeness use a 'distance' attribute = 1 / weight
      - eigenvector and clustering use 'weight' directly.

    If weighted=False:
      - all metrics are computed on the unweighted graph.
    """
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return pd.DataFrame()

    n = G.number_of_nodes()

    # Degree (unweighted) and strength (sum of similarity weights)
    degree = dict(G.degree())
    strength = dict(G.degree(weight='weight'))

    if degree_normalize:
        degree = {k: v / (n - 1) for k, v in degree.items()}
        max_strength = max(strength.values()) if strength else 1.0
        strength = {k: v / max_strength for k, v in strength.items()}

    if weighted:
        # Make a copy with a proper distance attribute = 1/weight
        H = G.copy()
        for u, v, d in H.edges(data=True):
            w = d.get('weight', 0.0)
            d['distance'] = 1.0 / w if w > 0 else np.inf

        # Weighted clustering and eigenvector use 'weight'
        clustering = nx.clustering(G, weight='weight')
        try:
            eigenvector = nx.eigenvector_centrality(G, weight='weight', max_iter=500)
        except nx.PowerIterationFailedConvergence:
            eigenvector = {node: np.nan for node in G.nodes()}

        # Path-based metrics use 'distance' as the edge length
        betweenness = nx.betweenness_centrality(H, weight='distance', normalized=True)
        closeness = nx.closeness_centrality(H, distance='distance')

    else:
        # Completely unweighted metrics
        clustering = nx.clustering(G)
        try:
            eigenvector = nx.eigenvector_centrality(G, max_iter=500)
        except nx.PowerIterationFailedConvergence:
            eigenvector = {node: np.nan for node in G.nodes()}
        betweenness = nx.betweenness_centrality(G, normalized=True)
        closeness = nx.closeness_centrality(G)

    df = pd.DataFrame({
        'degree': pd.Series(degree),
        'strength': pd.Series(strength),
        'clustering': pd.Series(clustering),
        'betweenness': pd.Series(betweenness),
        'closeness': pd.Series(closeness),
        'eigenvector': pd.Series(eigenvector),
    })

    return df


def build_similarity_network(df, feature_col, class_col,
                             similarity='cosine', sim_threshold=0.35,
                             binary=False, remove_singletons=False,
                             normalize=False,
                             layout='sfdp', 
                             layout_params={'K': 1.0, 'repulsiveforce':6,'overlap': False},
                            # layout='spring',
                            # layout_params={'k':6.0, 'iterations':800, 'scale':3},
                             bootstrap_iters=500,
                             bootstrapping_strategy='stratified',
                             community_detection_method='leiden',
                             consensus_partition_threshold=0.5,
                             consensus_graph_threshold=0.5,
                             G_type='consensus',
                             selected_nodes=None,
                             show=False,
                             seed=GLOBAL_SEED):
    print(feature_col)
    print("similarity: ", similarity)
    print("similarity_threshold: ", sim_threshold)
    print("consensus_partition_threshold: ", consensus_partition_threshold)
    print("consensus_graph_threshold: ", consensus_graph_threshold)
    print("bootstrap: ", str(bootstrap_iters))
    print("bootstrapping_strategy: ", bootstrapping_strategy)
    print("community_detection_method: ", community_detection_method)
    print("selected_nodes: ", selected_nodes)

    # Use single global seed for deterministic components
    if seed is not None:
        np.random.seed(seed)

    def _as_list(x):
        if isinstance(x, (list, tuple, set)):
            return list(x)
        return [x]

    # Select nodes
    if selected_nodes is None:
        classes = df[class_col].unique().tolist()
    else:
        selected_nodes = _as_list(selected_nodes)
        classes = [cls for cls in df[class_col].unique() if cls in selected_nodes]
        if len(classes) < len(selected_nodes):
            missing = set(selected_nodes) - set(classes)
            print(f"Warning: selected_nodes contains {len(missing)} nodes not found in data: {sorted(missing)}")
        if not classes:
            raise ValueError("No valid nodes found in selected_nodes that exist in the data.")

    # Filter df to the classes actually used
    if selected_nodes is not None:
        df_filtered = df[df[class_col].isin(classes)].copy()
        print(f"Filtering to {len(classes)} nodes from selected_nodes")
    else:
        df_filtered = df.copy()

    sim_matrices, partitions, modularity_scores, Graphs = [], [], [], []

    # Community method
    if community_detection_method == 'louvain':
        community_func = detect_communities_louvain
    elif community_detection_method == 'leiden':
        community_func = detect_communities_leiden
    elif community_detection_method == 'infomap':
        community_func = detect_communities_infomap
    else:
        raise ValueError(f"Unsupported community_detection_method: {community_detection_method}")

    # RandomState seeded from GLOBAL_SEED to generate per-iteration seeds
    #rs = np.random.RandomState(seed) if seed is not None else None
    rng = np.random.Generator(np.random.MT19937(seed)) if seed is not None else None

    # ---- bootstraps ----
    if bootstrap_iters > 0:
        for _ in tqdm(range(bootstrap_iters), desc="Bootstrapping"):
            # per-iteration seed for re-sampling only
            #iter_seed = None if rs is None else int(rs.randint(0, 2**32 - 1))
            iter_seed = None if rng is None else int(rng.integers(0, 2**32, dtype=np.uint32))

            boot_samples = []
            for cls in classes:
                class_data = df_filtered[df_filtered[class_col] == cls]
                if bootstrapping_strategy == 'stratified':
                    n_samples = len(class_data)
                elif bootstrapping_strategy == 'balanced_min':
                    n_samples = df_filtered.groupby(class_col).size().min()
                else:
                    try:
                        n_samples = int(bootstrapping_strategy.split('_')[-1])
                    except Exception:
                        raise ValueError(f"Invalid bootstrapping_strategy: {bootstrapping_strategy}")
                # >>> the ONLY place we use per-iteration randomness <<<
                boot_cls = class_data.sample(n=n_samples, replace=True, random_state=iter_seed)
                boot_samples.append(boot_cls)

            boot_df = pd.concat(boot_samples, ignore_index=True)
            feature_matrix_boot = pd.pivot_table(
                boot_df, index=class_col, columns=feature_col,
                aggfunc='size', fill_value=0
            ).reindex(classes).fillna(0)

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

            # Community detection uses the fixed GLOBAL_SEED for reproducibility
            partition, modularity = community_func(G, seed=seed)
            partitions.append(partition)
            modularity_scores.append(modularity if modularity is not None else 0)

        # Co-assignment matrices (UNROUNDED)
        coassign_matrix_bootstraps_df = compute_coassignment_matrix_by_bootstraps(Graphs)  # edge-based (raw)
        coassign_matrix_partitions, coassign_matrix_partitions_df = compute_coassignment_matrix_by_partitions(
            partitions, classes
        )

        if show:
            plot_heatmap_seaborn(coassign_matrix_partitions, classes,
                                 title="Co-assignment Matrix Heatmap")

        # Mean similarity (for G_mean)
        sim_array = np.mean(sim_matrices, axis=0)
        sim_matrix = pd.DataFrame(sim_array, index=classes, columns=classes)

        # Mean graph
        G_mean = nx.Graph()
        G_mean.add_nodes_from(classes)
        for i in sim_matrix.columns:
            for j in sim_matrix.columns:
                if i != j and sim_matrix.loc[i, j] > sim_threshold:
                    G_mean.add_edge(i, j, weight=sim_matrix.loc[i, j])

        # Consensus graph: UNROUNDED values for inclusion and weight
        G_consensus = create_consensus_graph_unrounded(
            coassign_matrix_bootstraps_df,
            threshold=consensus_graph_threshold
        )

        # Feature matrix on full data (for any downstream use)
        feature_matrix = pd.pivot_table(
            df_filtered, index=class_col, columns=feature_col,
            aggfunc='size', fill_value=0
        )
        if remove_singletons:
            feature_counts = (feature_matrix > 0).sum(axis=0)
            feature_matrix = feature_matrix.loc[:, feature_counts > 1]

        # Consensus partition from partition-based matrix (also UNROUNDED)
        partition_consensus = consensus_partition_from_coassign(
            coassign_matrix_partitions, classes, threshold=consensus_partition_threshold
        )
        modularity_score_mean = np.mean(modularity_scores)

        from collections import defaultdict
        def partition_dict_to_nodelist(partition_dict):
            nodelist = defaultdict(list)
            for node, community_id in partition_dict.items():
                nodelist[community_id].append(node)
            return list(nodelist.values())

        nodelist = partition_dict_to_nodelist(partition_consensus)
        modularity_score_consensus = nx.algorithms.community.quality.modularity(
            G_consensus, nodelist, weight='weight'
        )

    else:
        # No bootstraps: single graph path
        G_mean = None
        G_consensus = None
        partition_consensus = None
        modularity_score_mean = None
        modularity_score_consensus = None
        coassign_matrix_partitions_df = None
        coassign_matrix_bootstraps_df = None

        feature_matrix = pd.pivot_table(
            df_filtered, index=class_col, columns=feature_col,
            aggfunc='size', fill_value=0
        ).reindex(classes).fillna(0)

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

        # Fixed global seed for the single partition
        partition, modularity_score = community_func(G, seed=seed)
        partitions = [partition]
        modularity_scores = [modularity_score if modularity_score is not None else 0]

    # Decide which partition to use for plotting / return
    communities = partition_consensus if bootstrap_iters > 0 else partition

    if show:
        if bootstrap_iters > 0:
            if G_type == 'consensus':
                G_to_plot = G_consensus
                mod_score_for_plot = modularity_score_consensus
            elif G_type == 'mean':
                G_to_plot = G_mean
                mod_score_for_plot = modularity_score_mean
            else:
                raise ValueError(f"Unsupported G_type: {G_type}")

            plot_community_network(
                G_to_plot,
                communities,
                df=df_filtered,                      # <-- pass df here
                layout=layout,
                layout_params=layout_params,
                modularity_score=mod_score_for_plot,
                seed=GLOBAL_SEED,
            )
        else:
            plot_community_network(
                Graphs[0],
                communities,
                df=df_filtered,                      # <-- and here
                layout=layout,
                layout_params=layout_params,
                modularity_score=modularity_scores[0],
                seed=GLOBAL_SEED,
            )

    return (Graphs, G_mean, G_consensus, sim_matrix, feature_matrix, partitions,
            partition_consensus, modularity_scores, modularity_score_mean,
            modularity_score_consensus, classes, coassign_matrix_partitions_df,
            coassign_matrix_bootstraps_df)


from collections import defaultdict
def print_sorted_communities(partition):
    grouped = defaultdict(list)
    for node, community in partition.items():
        grouped[community].append(node)
    sorted_communities = sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True)
    for community, nodes in sorted_communities:
        node_str = ", ".join(map(str, nodes))
        print(f"Community {community} (size {len(nodes)}): {node_str}")


def remove_singleton_communities(G, communities, weight='weight'):
    """
    Remove singleton communities from a graph, recompute modularity and community stats.
    Returns (result, G_filtered, communities_filtered, singleton_nodes).
    """
    from collections import Counter, defaultdict
    import networkx as nx
    from networkx.algorithms.community.quality import modularity

    communities_in_G = {n: c for n, c in communities.items() if n in G}
    if not communities_in_G:
        empty_result = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'num_nodes_before': G.number_of_nodes(),
            'num_edges_before': G.number_of_edges(),
            'num_communities': 0,
            'num_singletons': 0,
            'num_large_communities': 0,
            'modularity_all': 0.0,
            'modularity_no_singletons': 0.0,
            'communities': [],
            'singleton_nodes': []
        }
        return empty_result, G.copy(), {}, []

    comm_sizes = Counter(communities_in_G.values())
    singleton_nodes = [n for n, c in communities_in_G.items() if comm_sizes[c] == 1]

    G_filtered = G.copy()
    G_filtered.remove_nodes_from(singleton_nodes)
    communities_filtered = {n: c for n, c in communities_in_G.items() if n not in singleton_nodes}

    unique_comms = sorted(set(communities_filtered.values()), key=str)
    relabel_map = {old: new for new, old in enumerate(unique_comms)}
    communities_filtered = {n: relabel_map[c] for n, c in communities_filtered.items()}

    def dict_to_partition(comm_dict):
        comm_map = defaultdict(set)
        for n, c in comm_dict.items():
            comm_map[c].add(n)
        return list(comm_map.values())

    partition_all = dict_to_partition(communities_in_G)
    partition_no_singletons = dict_to_partition(communities_filtered)

    modularity_all = modularity(G, partition_all, weight=weight) if partition_all else 0.0
    if (G_filtered.number_of_nodes() > 0) and partition_no_singletons:
        modularity_no_singletons = modularity(G_filtered, partition_no_singletons, weight=weight)
    else:
        modularity_no_singletons = 0.0

    records = []
    if communities_filtered:
        comm_nodes = defaultdict(list)
        for n, c in communities_filtered.items():
            comm_nodes[c].append(n)

        for comm, nodes in comm_nodes.items():
            subG = G_filtered.subgraph(nodes)
            n_nodes = len(nodes)

            if n_nodes > 1:
                internal_weight = sum(d.get(weight, 1) for _, _, d in subG.edges(data=True))
                max_possible_edges = n_nodes * (n_nodes - 1) / 2
                internal_density = internal_weight / max_possible_edges if max_possible_edges > 0 else 0.0
            else:
                internal_weight = 0.0
                internal_density = 0.0

            external_weight = sum(
                d.get(weight, 1) for _, _, d in nx.edge_boundary(G_filtered, nodes, data=True)
            )
            total_incident = 2 * internal_weight + external_weight
            conductance = external_weight / total_incident if total_incident > 0 else 0.0

            records.append({
                'community': comm,
                'size': n_nodes,
                'internal_weighted_density': internal_density,
                'external_ratio(conductance)': conductance,
                'low_density_flag': internal_density < 0.3
            })

    num_communities = len(comm_sizes)
    num_singletons = sum(1 for s in comm_sizes.values() if s == 1)
    num_large_communities = sum(1 for s in comm_sizes.values() if s > 1)

    result = {
        'num_nodes': G_filtered.number_of_nodes(),
        'num_edges': G_filtered.number_of_edges(),
        'num_nodes_before': G.number_of_nodes(),
        'num_edges_before': G.number_of_edges(),
        'num_communities': num_communities,
        'num_singletons': num_singletons,
        'num_large_communities': num_large_communities,
        'modularity_all': modularity_all,
        'modularity_no_singletons': modularity_no_singletons,
        'communities': records,
        'singleton_nodes': singleton_nodes,
    }

    return result, G_filtered, communities_filtered, singleton_nodes


# NGI: Cluster and node level generalism/specialism metrics
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


def get_cluster_score_input(
    df,
    *,
    node_col="curated_source_region",
    cluster_col="HC50",
):
    """
    From raw isolate-level data, compute the minimal per-cluster geometry needed
    for generalism/specialism scoring.

    Parameters
    ----------
    df : DataFrame
        Isolate-level table with at least node_col and cluster_col.
    node_col : str
        Column name for nodes (e.g. source region).
    cluster_col : str
        Column name for clusters (e.g. HC50 labels).

    Returns
    -------
    cluster_input : DataFrame
        Columns:
        - cluster_col           (cluster ID, e.g. 'HC50')
        - n_total               (total isolates in this cluster)
        - n_categories_present  (# nodes where the cluster is observed)
        - norm_entropy          (normalized entropy of node distribution, [0, 1])
        - max_share             (max node share for the cluster, [0, 1])
    """
    # Cross-tab: clusters (rows) x nodes (columns)
    ct = pd.crosstab(df[cluster_col], df[node_col], dropna=False)

    if ct.shape[0] == 0 or ct.shape[1] == 0:
        return pd.DataFrame(
            columns=[cluster_col, "n_total", "n_categories_present", "norm_entropy", "max_share"]
        )

    # Counts per cluster
    total_counts = ct.sum(axis=1)
    n_categories_present = (ct > 0).sum(axis=1)

    # Distribution over nodes: p(node | cluster)
    P = ct.div(total_counts.replace(0, np.nan), axis=0).fillna(0.0).values

    # Normalized entropy across nodes
    K = ct.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(P * np.log(P + 1e-12)).sum(axis=1)
        if K > 1:
            ent = ent / np.log(K)
        else:
            ent = np.zeros_like(ent)
    ent = np.clip(ent, 0.0, 1.0)

    # Max share across nodes
    max_share = P.max(axis=1)

    cluster_input = pd.DataFrame(
        {
            cluster_col: ct.index,
            "n_total": total_counts.astype(int),
            "n_categories_present": n_categories_present.astype(int),
            "norm_entropy": ent.astype(float),
            "max_share": max_share.astype(float),
        }
    ).reset_index(drop=True)

    return cluster_input


def _compute_cluster_enrichment_index(
    df,
    *,
    node_col="curated_source_region",
    cluster_col="HC50",
    eps=1e-12,
):
    """
    Internal helper: compute a continuous enrichment index per cluster.

    For each (cluster, node), we form a 2x2 table and compute:
    - Fisher's exact test p-value (one-sided, enrichment)
    - Odds ratio OR

    Then define per-pair strength:
        s(h, n) = max(0, log2(OR)) * -log10(p + eps)

    For each cluster h, we take:
        x_h = max_n s(h, n)

    Then map {x_h} to [0,1] via rank-normalization:
        enrichment_index(h) = (rank(x_h) - 1) / (H - 1)

    Parameters
    ----------
    df : DataFrame
        Isolate-level table.
    node_col : str
        Node column.
    cluster_col : str
        Cluster column.
    eps : float
        Small constant to avoid log(0).

    Returns
    -------
    enrich_df : DataFrame
        Columns:
        - cluster_col           (cluster ID)
        - enrichment_index      (in [0, 1])
        - max_enrichment_strength (raw max s(h, n) before normalization)
    """
    ct = pd.crosstab(df[cluster_col], df[node_col], dropna=False)
    if ct.shape[0] == 0 or ct.shape[1] == 0:
        return pd.DataFrame(columns=[cluster_col, "enrichment_index", "max_enrichment_strength"])

    clusters = ct.index.to_numpy()
    row_sum = ct.sum(axis=1).to_numpy()
    col_sum = ct.sum(axis=0).to_numpy()
    N = int(ct.values.sum())

    n_clusters, n_nodes = ct.shape
    scores = np.zeros((n_clusters, n_nodes), dtype=float)

    # Compute s(h, n) for each cluster/node pair
    for i in range(n_clusters):
        for j in range(n_nodes):
            a = int(ct.iat[i, j])
            b = int(row_sum[i] - a)
            c = int(col_sum[j] - a)
            d = int(N - (a + b + c))

            # Handle degenerate tables
            if (a + b) == 0 or (c + d) == 0 or (a + c) == 0 or (b + d) == 0:
                OR = 1.0
                p = 1.0
            else:
                OR, p = fisher_exact([[a, b], [c, d]], alternative="greater")

            if OR <= 1.0:
                log2_or_pos = 0.0
            else:
                log2_or_pos = float(np.log2(OR))

            weight = float(-np.log10(p + eps))  # larger for smaller p
            if weight < 0:
                weight = 0.0

            scores[i, j] = log2_or_pos * weight

    max_strength = scores.max(axis=1)

    # Rank-normalize max_strength into [0, 1] without choosing a cutoff
    series_strength = pd.Series(max_strength)
    if len(series_strength) > 1:
        ranks = series_strength.rank(method="min")  # 1..H
        enrichment_index = (ranks - 1) / (len(series_strength) - 1)
    else:
        # Only one cluster: set index to 0
        enrichment_index = pd.Series(np.zeros_like(max_strength, dtype=float))

    enrich_df = pd.DataFrame(
        {
            cluster_col: clusters,
            "enrichment_index": enrichment_index.to_numpy(dtype=float),
            "max_enrichment_strength": max_strength.astype(float),
        }
    )

    return enrich_df


def add_cluster_generalism_scores(
    df,
    cluster_input_df,
    *,
    node_col="curated_source_region",
    cluster_col="HC50",
    lambda_geom=0.5,
    min_entropy=0.0,
):
    """
    Add continuous generalism/specialism scores to a minimal cluster input DataFrame,
    combining:
    - geometry-based specialism (entropy + max_share)
    - enrichment-based strength (Fisher + OR across nodes, no hard cutoffs)

    Parameters
    ----------
    df : DataFrame
        Original isolate-level DataFrame.
    cluster_input_df : DataFrame
        Output of get_cluster_score_input(df, ...). Must contain:
        - cluster_col
        - n_categories_present
        - norm_entropy
        - max_share
    node_col : str
        Column name for nodes in df.
    cluster_col : str
        Column name for clusters in df and in cluster_input_df.
    lambda_geom : float
        Weight for geometry-based specialism in [0, 1].
        1.0 = pure geometry, 0.0 = pure enrichment index.
    min_entropy : float
        Lower bound to clip norm_entropy before computing specialism_index_geom.

    Returns
    -------
    cluster_scores : DataFrame
        cluster_input_df with additional columns:
        - specialism_index_geom      (geometry-only)
        - generalism_index_geom
        - enrichment_index           (0-1, rank-based)
        - max_enrichment_strength    (raw max s(h, n))
        - specialism_index_total     (combined geom + enrichment)
        - generalism_index_total
    """
    dfc = cluster_input_df.copy()

    # Infer K = total number of nodes as the maximum number of nodes
    # any cluster is seen in. This assumes all nodes have at least one cluster.
    if "n_categories_present" not in dfc.columns:
        raise ValueError("cluster_input_df must contain 'n_categories_present' column.")
    K = int(dfc["n_categories_present"].max())
    if K <= 1:
        raise ValueError("Need at least 2 nodes to define generalism/specialism.")

    # Ensure required columns exist
    if "norm_entropy" not in dfc.columns or "max_share" not in dfc.columns:
        raise ValueError("cluster_input_df must contain 'norm_entropy' and 'max_share' columns.")

    # Geometry-based specialism index
    # SI_geom(h) = (1 - H_norm(h)) * D(h)
    # where D(h) = (max_share - 1/K) / (1 - 1/K), clipped to [0, 1]
    H = dfc["norm_entropy"].fillna(0.0)
    H = H.clip(lower=min_entropy, upper=1.0)  # optional entropy floor

    max_share = dfc["max_share"].fillna(0.0)
    uniform_share = 1.0 / K
    denom = 1.0 - uniform_share

    D = (max_share - uniform_share) / denom
    D = D.clip(lower=0.0, upper=1.0)

    specialism_geom = (1.0 - H) * D
    specialism_geom = specialism_geom.clip(lower=0.0, upper=1.0)
    generalism_geom = 1.0 - specialism_geom

    dfc["specialism_index_geom"] = specialism_geom
    dfc["generalism_index_geom"] = generalism_geom

    # Enrichment-based index (continuous, no hard cutoffs)
    enrich_df = _compute_cluster_enrichment_index(
        df, node_col=node_col, cluster_col=cluster_col
    )

    # Merge enrichment_index into dfc
    dfc = dfc.merge(enrich_df, on=cluster_col, how="left")
    dfc["enrichment_index"] = dfc["enrichment_index"].fillna(0.0)
    dfc["max_enrichment_strength"] = dfc["max_enrichment_strength"].fillna(0.0)

    # Combine geometry and enrichment into total specialism
    lambda_geom = float(lambda_geom)
    if not (0.0 <= lambda_geom <= 1.0):
        raise ValueError("lambda_geom must be in [0, 1].")

    specialism_total = (
        lambda_geom * dfc["specialism_index_geom"].astype(float)
        + (1.0 - lambda_geom) * dfc["enrichment_index"].astype(float)
    )
    specialism_total = specialism_total.clip(lower=0.0, upper=1.0)
    generalism_total = 1.0 - specialism_total

    dfc["specialism_index_total"] = specialism_total
    dfc["generalism_index_total"] = generalism_total

    return dfc


def compute_node_specialism_generalism(
    df,
    cluster_scores,
    *,
    node_col="curated_source_region",
    cluster_col="HC50",
    use_total=True,
    min_node_size=100,
    min_cluster_size=100,
    top_HC_pct=None,
):
    """
    Compute node-level specialism/generalism indices by averaging cluster scores,
    weighted by the within-node cluster composition.

    Also computes how concentrated each node is in its largest HC50 clusters:
    for each percentage p in top_HC_pct, we compute the fraction of isolates in
    the node (within scored clusters) that come from the top p% of HC50 clusters
    in that node (ranked by within-node cluster size).

    Parameters
    ----------
    df : DataFrame
        Original isolate-level DataFrame.
    cluster_scores : DataFrame
        Output of add_cluster_generalism_scores().
        Must contain:
        - a column matching cluster_col (e.g. 'HC50')
        - 'specialism_index_geom' and 'generalism_index_geom'
        - 'specialism_index_total' and 'generalism_index_total'
    node_col : str
        Column name for nodes in df.
    cluster_col : str
        Column name for clusters in df and in cluster_scores.
    use_total : bool
        If True, use specialism_index_total/generalism_index_total
        (geometry + enrichment). If False, use geometry-only indices.
    min_node_size : int or None
        Minimum number of isolates (all clusters) required for a node
        to receive a score. If None, no node-size filtering.
    min_cluster_size : int or None
        Minimum number of isolates (all nodes) required for a cluster
        to be used in node score calculations. If None, no cluster-size filtering.
    top_HC_pct : list of numbers, optional
        Percentages (e.g. [5, 10, 25, 50]) defining what “top p% of clusters”
        means for the concentration metrics. Each p is interpreted as p/100.

    Returns
    -------
    node_scores : DataFrame
        Columns:
        - node
        - node_specialism_index
        - node_generalism_index
        - n_isolates  (per node, all clusters pre-filtering)
        - pct_node_from_topXpct_clusters  (fraction in [0, 1] for each X in top_HC_pct)
    """
    import numpy as np
    import pandas as pd

    # Default top_HC_pct if not provided
    if top_HC_pct is None:
        top_HC_pct = [5, 10, 25, 50]

    # Make sure it's a list (in case someone passes a tuple/np.array)
    top_HC_pct = list(top_HC_pct)

    # Full cross-tab: counts of (node, cluster)
    ct_full = pd.crosstab(df[node_col], df[cluster_col], dropna=False)

    # Global node sizes (all clusters)
    node_totals_global = ct_full.sum(axis=1)

    # Filter nodes by min_node_size
    if min_node_size is not None:
        min_node_size = int(min_node_size)
        keep_nodes = node_totals_global[node_totals_global >= min_node_size].index
        ct = ct_full.loc[keep_nodes]
        node_totals_global = node_totals_global.loc[keep_nodes]
    else:
        ct = ct_full.copy()

    # Build the dynamic list of empty columns (for early returns)
    base_cols = [
        "node",
        "node_specialism_index",
        "node_generalism_index",
        "n_isolates",
    ]
    pct_cols = [f"pct_node_from_top{p}pct_clusters" for p in top_HC_pct]
    empty_cols = base_cols + pct_cols

    if ct.shape[0] == 0:
        # No nodes pass the size filter
        return pd.DataFrame(columns=empty_cols)

    # Global cluster sizes (all nodes)
    cluster_totals_global = ct_full.sum(axis=0)

    # Filter clusters by min_cluster_size
    if min_cluster_size is not None:
        min_cluster_size = int(min_cluster_size)
        keep_clusters = cluster_totals_global[cluster_totals_global >= min_cluster_size].index
        ct = ct.loc[:, ct.columns.isin(keep_clusters)]

    if ct.shape[1] == 0:
        # No clusters pass the size filter
        return pd.DataFrame(columns=empty_cols)

    # Recompute node totals based only on retained clusters (for probabilities)
    node_totals_for_probs = ct.sum(axis=1)

    # Drop nodes with zero mass after cluster filtering
    nonzero_nodes = node_totals_for_probs[node_totals_for_probs > 0].index
    ct = ct.loc[nonzero_nodes]
    node_totals_for_probs = node_totals_for_probs.loc[nonzero_nodes]
    node_totals_global = node_totals_global.loc[nonzero_nodes]

    if ct.shape[0] == 0:
        return pd.DataFrame(columns=empty_cols)

    # Within-node composition p(cluster | node) using retained clusters
    p_h_given_n = ct.div(node_totals_for_probs.replace(0, np.nan), axis=0).fillna(0.0)

    # Align cluster scores to ct columns
    cs = cluster_scores.set_index(cluster_col)

    if use_total:
        si_col = "specialism_index_total"
        gi_col = "generalism_index_total"
    else:
        si_col = "specialism_index_geom"
        gi_col = "generalism_index_geom"

    if si_col not in cs.columns or gi_col not in cs.columns:
        raise ValueError(f"cluster_scores must contain '{si_col}' and '{gi_col}'.")

    # Restrict to clusters that appear in both ct and cluster_scores
    common_clusters = [c for c in p_h_given_n.columns if c in cs.index]
    if len(common_clusters) == 0:
        raise ValueError("No overlap between clusters in df and cluster_scores after filtering.")

    p_sub = p_h_given_n[common_clusters]
    si_vec = cs.loc[common_clusters, si_col]
    gi_vec = cs.loc[common_clusters, gi_col]

    # Node-level indices: weighted sums over cluster composition
    node_specialism = (p_sub * si_vec.values).sum(axis=1)
    node_generalism = (p_sub * gi_vec.values).sum(axis=1)

    # ---- New: concentration metrics — fraction of node's scored isolates in top X% clusters ----
    # Dictionary keyed by the original percentage value (e.g. 5, 10, 25, 50)
    frac_top = {p: [] for p in top_HC_pct}

    for node, row in p_sub.iterrows():
        probs = row.values
        # Consider only clusters actually present in this node
        nonzero = probs[probs > 0]
        m = len(nonzero)
        if m == 0:
            for p in top_HC_pct:
                frac_top[p].append(0.0)
            continue

        nonzero_sorted = np.sort(nonzero)[::-1]  # descending by within-node share

        for p in top_HC_pct:
            p_frac = float(p) / 100.0
            k = max(1, int(np.ceil(p_frac * m)))  # at least 1 cluster
            top_sum = nonzero_sorted[:k].sum()    # fraction of node mass in top k clusters
            frac_top[p].append(float(top_sum))

    # Build output DataFrame
    data = {
        "node": p_sub.index,
        "node_specialism_index": node_specialism,
        "node_generalism_index": node_generalism,
        # total isolates in the node (all clusters, before cluster filtering)
        "n_isolates": node_totals_global.loc[p_sub.index].astype(int),
    }

    # Add the fraction columns for each requested percentile
    for p in top_HC_pct:
        col_name = f"pct_node_from_top{p}pct_clusters"
        data[col_name] = np.array(frac_top[p])

    node_scores = pd.DataFrame(data).reset_index(drop=True)

    # Sort by specialism (descending) for convenience
    node_scores = node_scores.sort_values(
        ["node_specialism_index", "n_isolates"], ascending=[False, False]
    ).reset_index(drop=True)

    return node_scores

def compute_node_specialism_generalism(
    df,
    cluster_scores,
    *,
    node_col="curated_source_region",
    cluster_col="HC50",
    use_total=True,
    min_node_size=50,
    min_cluster_size=100,
    top_HC_pct=None,
):
    """
    Compute node-level specialism/generalism indices by averaging cluster scores,
    weighted by the within-node cluster composition.

    Also computes how concentrated each node is in its largest HC50 clusters:
    for each percentage p in top_HC_pct, we compute the fraction of isolates in
    the node (within scored clusters) that come from the top p% of HC50 clusters
    in that node (ranked by within-node cluster size).

    Returns
    -------
    node_scores : DataFrame
        Columns:
        - node
        - node_specialism_index
        - node_generalism_index
        - n_isolates
        - n_scored_isolates
        - frac_isolates_scored
        - n_scored_clusters
        - effective_num_clusters
        - cluster_evenness
        - hhi_cluster_concentration
        - weighted_mean_cluster_breadth
        - pct_isolates_in_private_clusters
        - pct_node_from_topXpct_clusters  (fraction in [0, 1] for each X in top_HC_pct)
    """
    import numpy as np
    import pandas as pd

    if top_HC_pct is None:
        top_HC_pct = [5, 10, 25, 50]

    top_HC_pct = list(top_HC_pct)

    # Full cross-tab: counts of (node, cluster)
    ct_full = pd.crosstab(df[node_col], df[cluster_col], dropna=False)

    # Global node sizes (all clusters)
    node_totals_global = ct_full.sum(axis=1)

    # Filter nodes by min_node_size
    if min_node_size is not None:
        min_node_size = int(min_node_size)
        keep_nodes = node_totals_global[node_totals_global >= min_node_size].index
        ct = ct_full.loc[keep_nodes]
        node_totals_global = node_totals_global.loc[keep_nodes]
    else:
        ct = ct_full.copy()

    base_cols = [
        "node",
        "node_specialism_index",
        "node_generalism_index",
        "n_isolates",
        "n_scored_isolates",
        "frac_isolates_scored",
        "n_scored_clusters",
        "effective_num_clusters",
        "cluster_evenness",
        "hhi_cluster_concentration",
        "weighted_mean_cluster_breadth",
        "pct_isolates_in_private_clusters",
    ]
    pct_cols = [f"pct_node_from_top{p}pct_clusters" for p in top_HC_pct]
    empty_cols = base_cols + pct_cols

    if ct.shape[0] == 0:
        return pd.DataFrame(columns=empty_cols)

    # Global cluster sizes (all nodes)
    cluster_totals_global = ct_full.sum(axis=0)
    cluster_breadth_global = (ct_full > 0).sum(axis=0)

    # Filter clusters by min_cluster_size
    if min_cluster_size is not None:
        min_cluster_size = int(min_cluster_size)
        keep_clusters = cluster_totals_global[cluster_totals_global >= min_cluster_size].index
        ct = ct.loc[:, ct.columns.isin(keep_clusters)]

    if ct.shape[1] == 0:
        return pd.DataFrame(columns=empty_cols)

    # Recompute node totals based only on retained clusters
    node_totals_for_probs = ct.sum(axis=1)

    # Drop nodes with zero mass after cluster filtering
    nonzero_nodes = node_totals_for_probs[node_totals_for_probs > 0].index
    ct = ct.loc[nonzero_nodes]
    node_totals_for_probs = node_totals_for_probs.loc[nonzero_nodes]
    node_totals_global = node_totals_global.loc[nonzero_nodes]

    if ct.shape[0] == 0:
        return pd.DataFrame(columns=empty_cols)

    # Within-node composition p(cluster | node)
    p_h_given_n = ct.div(node_totals_for_probs.replace(0, np.nan), axis=0).fillna(0.0)

    # Align cluster scores to ct columns
    cs = cluster_scores.set_index(cluster_col)

    if use_total:
        si_col = "specialism_index_total"
        gi_col = "generalism_index_total"
    else:
        si_col = "specialism_index_geom"
        gi_col = "generalism_index_geom"

    if si_col not in cs.columns or gi_col not in cs.columns:
        raise ValueError(f"cluster_scores must contain '{si_col}' and '{gi_col}'.")

    # Restrict to clusters that appear in both ct and cluster_scores
    common_clusters = [c for c in p_h_given_n.columns if c in cs.index]
    if len(common_clusters) == 0:
        raise ValueError("No overlap between clusters in df and cluster_scores after filtering.")

    p_sub = p_h_given_n[common_clusters]
    si_vec = cs.loc[common_clusters, si_col]
    gi_vec = cs.loc[common_clusters, gi_col]
    breadth_vec = cluster_breadth_global.loc[common_clusters]

    # Node-level indices: weighted sums over cluster composition
    node_specialism = (p_sub * si_vec.values).sum(axis=1)
    node_generalism = (p_sub * gi_vec.values).sum(axis=1)

    # Concentration metrics
    frac_top = {p: [] for p in top_HC_pct}

    # Additional node-level output metrics
    n_scored_clusters = []
    effective_num_clusters = []
    cluster_evenness = []
    hhi_cluster_concentration = []
    weighted_mean_cluster_breadth = []
    pct_isolates_in_private_clusters = []

    for node, row in p_sub.iterrows():
        probs = row.values.astype(float)
        nonzero = probs[probs > 0]
        m = len(nonzero)

        n_scored_clusters.append(int(m))

        if m == 0:
            effective_num_clusters.append(0.0)
            cluster_evenness.append(np.nan)
            hhi_cluster_concentration.append(np.nan)
            weighted_mean_cluster_breadth.append(np.nan)
            pct_isolates_in_private_clusters.append(np.nan)
            for p in top_HC_pct:
                frac_top[p].append(0.0)
            continue

        # Existing top-X% metrics
        nonzero_sorted = np.sort(nonzero)[::-1]
        for p in top_HC_pct:
            p_frac = float(p) / 100.0
            k = max(1, int(np.ceil(p_frac * m)))
            top_sum = nonzero_sorted[:k].sum()
            frac_top[p].append(float(top_sum))

        # New output metrics
        H = -np.sum(nonzero * np.log(nonzero))
        effective_num_clusters.append(float(np.exp(H)))
        cluster_evenness.append(float(H / np.log(m)) if m > 1 else 1.0)
        hhi_cluster_concentration.append(float(np.sum(nonzero ** 2)))
        weighted_mean_cluster_breadth.append(float(np.sum(probs * breadth_vec.values)))
        pct_isolates_in_private_clusters.append(float(np.sum(probs[breadth_vec.values == 1])))

    # Build output DataFrame
    data = {
        "node": p_sub.index,
        "node_specialism_index": node_specialism,
        "node_generalism_index": node_generalism,
        "n_isolates": node_totals_global.loc[p_sub.index].astype(int),
        "n_scored_isolates": node_totals_for_probs.loc[p_sub.index].astype(int),
        "frac_isolates_scored": (
            node_totals_for_probs.loc[p_sub.index] / node_totals_global.loc[p_sub.index]
        ).astype(float),
        "n_scored_clusters": np.array(n_scored_clusters),
        "effective_num_clusters": np.array(effective_num_clusters),
        "cluster_evenness": np.array(cluster_evenness),
        "hhi_cluster_concentration": np.array(hhi_cluster_concentration),
        "weighted_mean_cluster_breadth": np.array(weighted_mean_cluster_breadth),
        "pct_isolates_in_private_clusters": np.array(pct_isolates_in_private_clusters),
    }

    for p in top_HC_pct:
        col_name = f"pct_node_from_top{p}pct_clusters"
        data[col_name] = np.array(frac_top[p])

    node_scores = pd.DataFrame(data).reset_index(drop=True)

    node_scores = node_scores.sort_values(
        ["node_specialism_index", "n_isolates"], ascending=[False, False]
    ).reset_index(drop=True)

    return node_scores