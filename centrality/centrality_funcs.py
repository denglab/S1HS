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


# ============================================
# Single-node Step 2 (DM-based synthetic nodes)
# ============================================

EPS = 1e-8
EPS_DOT = 1e-12


# -----------------
# Utility functions
# -----------------
def _ensure_id_col(df: pd.DataFrame, id_col: str = "biosample_acc") -> pd.DataFrame:
    """Ensure id_col exists and is unique per row."""
    if id_col not in df.columns:
        df = df.copy()
        df[id_col] = df.index.astype(str)
    if df[id_col].duplicated().any():
        dups = df[df[id_col].duplicated()][id_col].unique()[:5]
        raise ValueError(
            f"{id_col} must be unique per row; duplicates found (up to 5 shown): {dups}"
        )
    return df


def _hist(levels: List[str], s: pd.Series) -> np.ndarray:
    """Histogram over a fixed level ordering."""
    c = pd.Series(s.astype(str)).value_counts(sort=False)
    return c.reindex(levels, fill_value=0).astype(int).values


def _normalize(x: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """Normalize a nonnegative vector to sum to 1 (or all zeros if total == 0)."""
    x = np.asarray(x, float)
    if eps > 0:
        x = x + float(eps)
    s = x.sum()
    return x / s if s > 0 else np.zeros_like(x, dtype=float)


def _cosine_distance(p: np.ndarray, q: np.ndarray, eps: float = EPS_DOT) -> float:
    """Cosine distance = 1 - cos(theta). Adds a tiny eps to avoid 0-vectors."""
    p = np.asarray(p, float) + eps
    q = np.asarray(q, float) + eps
    num = float(np.dot(p, q))
    den = float(np.linalg.norm(p) * np.linalg.norm(q))
    if den == 0.0:
        return 0.0
    return float(1.0 - num / den)


def _shannon_evenness(counts: np.ndarray) -> Tuple[float, float]:
    """Return (Shannon entropy H, evenness J) for a count vector."""
    total = counts.sum()
    if total <= 0:
        return 0.0, 0.0
    p = counts / total
    p = p[p > 0]
    H = float(-(p * np.log(p)).sum())
    K_obs = int((counts > 0).sum())
    J = float(H / np.log(K_obs)) if K_obs > 1 else 0.0
    return H, J


def _iqr(arr_like) -> float:
    """Interquartile range."""
    arr = np.asarray(list(arr_like), float)
    if arr.size == 0:
        return np.nan
    return float(np.percentile(arr, 75) - np.percentile(arr, 25))


def _validate_counts_mode(counts_mode: str) -> str:
    """Ensure counts_mode is one of the supported options."""
    valid = {"positive", "nonnegative", "capacity_tempered"}
    if counts_mode not in valid:
        raise ValueError(f"counts_mode must be one of {valid}, got {counts_mode!r}")
    return counts_mode


# -----------------------------
# DM & capacity helper methods
# -----------------------------


def sample_support(
    rng: np.random.Generator, weights: np.ndarray, K_star: int
) -> np.ndarray:
    """
    Choose K_star distinct indices without replacement with probabilities ∝ weights.
    Raises ValueError if fewer than K_star positive-weight bins.
    """
    w = np.array(weights, float)
    pos = np.flatnonzero(w > 0)
    if pos.size < K_star:
        raise ValueError("Insufficient positive-weight bins for K_star.")
    probs = w[pos] / w[pos].sum()
    chosen = rng.choice(pos, size=K_star, replace=False, p=probs)
    return np.sort(chosen)


def positive_multinomial_counts(
    rng: np.random.Generator, n: int, theta: np.ndarray
) -> np.ndarray:
    """Multinomial counts with each category receiving at least 1 (requires n >= K)."""
    K = len(theta)
    if n < K:
        raise ValueError("positive_multinomial_counts requires n >= number of categories")
    base = np.ones(K, dtype=int)
    rem = n - K
    if rem == 0:
        return base
    add = rng.multinomial(rem, theta / np.clip(theta.sum(), 1e-12, None))
    return base + add


def nonnegative_multinomial_counts(
    rng: np.random.Generator, n: int, theta: np.ndarray
) -> np.ndarray:
    """Plain multinomial; zero counts allowed."""
    n_int = int(n)
    return rng.multinomial(n_int, theta / np.clip(theta.sum(), 1e-12, None)).astype(int)


def _bounded_multinomial_once(
    total: int, probs: np.ndarray, cap: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """
    Single-shot bounded allocation ≤ cap that approximately preserves probs.

    Returns allocation summing to min(total, cap.sum()).
    """
    cap = np.asarray(cap, int)
    if total <= 0 or cap.sum() == 0:
        return np.zeros_like(cap, dtype=int)
    w = np.asarray(probs, float)
    if w.sum() <= 0:
        w = (cap > 0).astype(float)
    w = w / w.sum()
    want = w * min(total, int(cap.sum()))
    flo = np.floor(want).astype(int)
    flo = np.minimum(flo, cap)
    rem = int(min(total, int(cap.sum())) - flo.sum())
    if rem <= 0:
        return flo
    frac = want - flo
    mask_spare = (cap - flo) > 0
    if not np.any(mask_spare):
        return flo
    idx = np.flatnonzero(mask_spare)
    p = frac[idx]
    if p.sum() == 0:
        p = np.ones_like(p, float)
    p = p / p.sum()
    add = np.zeros_like(flo)
    for _ in range(rem):
        j = int(rng.choice(idx.size, p=p))
        g = idx[j]
        if add[g] + flo[g] < cap[g]:
            add[g] += 1
        else:
            spare_idx = np.flatnonzero((cap - (flo + add)) > 0)
            if spare_idx.size == 0:
                break
            g2 = int(rng.choice(spare_idx.size))
            add[spare_idx[g2]] += 1
    return flo + add


def capacity_tempered_counts(
    rng: np.random.Generator, n: int, theta: np.ndarray, capacity: np.ndarray
) -> Optional[np.ndarray]:
    """
    Draw counts with total n, but clipped by per-category capacity (without replacement).

    Returns
    -------
    np.ndarray or None
        If capacity.sum() < n, returns None (impossible to allocate).
        Otherwise returns integer counts <= capacity, summing to n.
    """
    cap = np.asarray(capacity, int)
    if int(cap.sum()) < int(n):
        return None
    return _bounded_multinomial_once(int(n), np.asarray(theta, float), cap, rng)


def build_ids_by_level(
    df_bg: pd.DataFrame, level_col: str, id_col: str
) -> Dict[str, List[str]]:
    """Map HC50 level -> list of isolate IDs in the background."""
    ids_by_level: Dict[str, List[str]] = {}
    for lev, iso in zip(df_bg[level_col].astype(str).values,
                        df_bg[id_col].astype(str).values):
        ids_by_level.setdefault(lev, []).append(iso)
    return ids_by_level


def sample_isolates_without_replacement(
    rng: np.random.Generator,
    ids_by_level: Dict[str, List[str]],
    counts_by_level_index: Dict[int, int],
    levels: List[str],
) -> Optional[List[str]]:
    """
    Given desired counts per level index, sample that many isolate IDs without replacement.

    Returns
    -------
    list[str] or None
        If any requested count exceeds available isolates for that level, returns None.
    """
    chosen: List[str] = []
    for idx, c in counts_by_level_index.items():
        lev = levels[idx]
        pool = ids_by_level.get(lev, [])
        if len(pool) < c:
            return None
        take_idx = rng.choice(len(pool), size=c, replace=False)
        chosen.extend([pool[i] for i in take_idx])
    return chosen


# -----------------------------------------
# 1) Calibrate τ for a single target node
# -----------------------------------------


def calibrate_tau_for_single_node(
    df: pd.DataFrame,
    target_node: str,
    taus: Tuple[float, ...],
    *,
    B_calib: int,
    node_col: str,
    level_col: str,
    id_col: str,
    alpha_support: float = 0.10,
    counts_mode: str = "positive",
    min_avail_per_level: int = 1,
    rng_seed: int = 123,
) -> Tuple[Optional[float], Dict[float, float], Dict[str, object]]:
    """
    Calibrate τ for a single node by matching L1O cosine distance.

    For each τ in `taus`, we:
      1. Build the leave-one-node-out background distribution p_bg(level).
      2. Repeatedly simulate pseudo-nodes with the same (n, richness) as
         the target node under a Dirichlet–Multinomial model.
      3. Compute the cosine distance between pseudo-node frequency and p_bg,
         aggregating the distances across B_calib draws.
      4. Take the median distance for that τ.

    We then select τ* that minimizes |median_sim(τ) - observed_L1O|.

    Parameters
    ----------
    df : DataFrame
        Full dataset with columns [node_col, level_col, id_col].
    target_node : str
        Node name whose τ we are calibrating.
    taus : tuple of float
        Grid of τ values to test.
    B_calib : int
        Number of pseudo-draws per τ.
    node_col, level_col, id_col : str
        Column names for node, HC50 level, and isolate ID.
    alpha_support : float, default 0.10
        Exponent controlling how strongly background counts tilt support selection.
    counts_mode : {"positive","nonnegative","capacity_tempered"}
        How to sample counts given θ.
    min_avail_per_level : int, default 1
        Minimum background isolates per level to consider that level eligible.
    rng_seed : int, default 123
        Seed for numpy RNG.

    Returns
    -------
    tau_star : float or None
        Best τ* for this node, or None if calibration fails badly.
    median_by_tau : dict
        Mapping τ -> median simulated distance for that τ.
    calib_info : dict
        Diagnostics, including:
          - "D_obs" (observed L1O distance)
          - "n" (observed count)
          - "K_obs" (observed richness)
          - "taus" (grid)
          - "median_by_tau"
          - "n_sim_accepted_by_tau" (effective draws per τ)
    """
    counts_mode = _validate_counts_mode(counts_mode)

    df = _ensure_id_col(df, id_col=id_col)
    target_node = str(target_node)

    # Extract observed node and background
    df_node = df.loc[df[node_col] == target_node].copy()
    df_bg = df.loc[df[node_col] != target_node].copy()
    if df_node.empty:
        raise ValueError(f"Target node {target_node!r} has no rows in df.")

    levels = sorted(df[level_col].astype(str).unique().tolist())
    n_v = int(df_node.shape[0])
    K_obs = int(df_node[level_col].nunique())

    # Observed L1O distance for the node
    obs_counts = _hist(levels, df_node[level_col])
    p_node = _normalize(obs_counts, eps=0.0)
    bg_counts_full = _hist(levels, df_bg[level_col])
    p_bg_full = _normalize(bg_counts_full, eps=0.0)
    D_obs = _cosine_distance(p_node, p_bg_full)

    # Background available counts & support weights
    bg_counts = (
        df_bg[level_col].astype(str).value_counts().reindex(levels, fill_value=0).astype(int)
    )
    avail = bg_counts.values
    # eligible levels must have at least min_avail_per_level isolates
    eligible_mask = (avail >= int(min_avail_per_level)).astype(float)
    if eligible_mask.sum() == 0 or avail.sum() == 0:
        # Degenerate background
        return None, {}, {
            "D_obs": float(D_obs),
            "n": n_v,
            "K_obs": K_obs,
            "taus": list(taus),
            "median_by_tau": {},
            "n_sim_accepted_by_tau": {},
            "warning": "No eligible background levels for calibration.",
        }

    p_bg = _normalize(bg_counts.values, eps=0.0)
    K_bg = int((p_bg > 0).sum())
    C_eligible = int(eligible_mask.sum())
    # K_star limited by what we can actually support
    K_star = int(min(K_obs, K_bg, C_eligible, n_v))

    # Availability-tempered support weights
    w_support = (p_bg * np.clip(avail, 1, None) ** float(alpha_support)) * eligible_mask
    if (w_support > 0).sum() < K_star:
        # fallback: uniform over eligible bins
        w_support = eligible_mask.copy()

    rng = np.random.default_rng(rng_seed)

    median_by_tau: Dict[float, float] = {}
    n_sim_accepted_by_tau: Dict[float, int] = {}

    for tau in taus:
        tau = float(tau)
        d_vals: List[float] = []

        for _ in range(B_calib):
            try:
                support_idx = sample_support(rng, w_support, K_star)
            except ValueError:
                # No valid support for this draw; skip
                continue

            p_bg_support = p_bg[support_idx]
            if p_bg_support.sum() <= 0:
                continue

            alpha_vec = tau * (p_bg_support / p_bg_support.sum())
            theta = rng.dirichlet(alpha_vec)

            if counts_mode == "positive":
                if n_v < len(support_idx):
                    # can't give each category at least 1
                    continue
                counts_support = positive_multinomial_counts(rng, n_v, theta)
            elif counts_mode == "capacity_tempered":
                cap_support = avail[support_idx].astype(int)
                c = capacity_tempered_counts(rng, n_v, theta, cap_support)
                if c is None:
                    continue
                counts_support = c
            else:  # "nonnegative"
                counts_support = nonnegative_multinomial_counts(rng, n_v, theta)

            # Must not exceed capacity
            if np.any(counts_support > avail[support_idx]):
                continue

            # Embed into full length and compute distance
            pseudo_counts = np.zeros(len(levels), dtype=int)
            pseudo_counts[support_idx] = counts_support
            d = _cosine_distance(_normalize(pseudo_counts, eps=0.0), p_bg_full + EPS)
            d_vals.append(float(d))

        n_sim_accepted_by_tau[tau] = len(d_vals)
        median_by_tau[tau] = float(np.median(d_vals)) if d_vals else np.nan

    # Warn if calibration has very few effective draws (can be noisy)
    max_eff = max(n_sim_accepted_by_tau.values()) if n_sim_accepted_by_tau else 0
    if max_eff < B_calib * 0.5:
        print(
            f"[calibrate_tau_for_single_node] Warning: only {max_eff} accepted "
            f"simulations for best τ; calibration may be noisy."
        )

    # Choose best τ *
    tau_star = None
    best_gap = None
    for tau in taus:
        tau = float(tau)
        dmed = median_by_tau.get(tau, np.nan)
        if np.isnan(dmed):
            continue
        gap = abs(dmed - D_obs)
        if (best_gap is None) or (gap < best_gap):
            best_gap = gap
            tau_star = tau

    calib_info = {
        "D_obs": float(D_obs),
        "n": n_v,
        "K_obs": K_obs,
        "taus": list(map(float, taus)),
        "median_by_tau": {float(k): float(v) for k, v in median_by_tau.items()},
        "n_sim_accepted_by_tau": {float(k): int(v) for k, v in n_sim_accepted_by_tau.items()},
        "K_bg": K_bg,
        "C_eligible": C_eligible,
        "K_star": K_star,
        "alpha_support": float(alpha_support),
        "counts_mode": counts_mode,
        "min_avail_per_level": int(min_avail_per_level),
    }

    return tau_star, median_by_tau, calib_info


# ---------------------------------------------
# 2) Generate pseudo-nodes for a single target
# ---------------------------------------------


def generate_pseudonodes_single_node(
    df: pd.DataFrame,
    target_node: str,
    tau: float,
    *,
    B_pseudo: int,
    node_col: str,
    level_col: str,
    id_col: str,
    alpha_support: float = 0.10,
    counts_mode: str = "positive",
    min_avail_per_level: int = 1,
    rng_seed: int = 123,
    max_resample_tries: int = 200,
) -> Tuple[List[List[str]], Dict[str, object]]:
    """
    Generate DM-based pseudo-nodes for a single target node.

    The pseudo-nodes:
      - Have the same n (number of isolates) as the observed node.
      - Have richness <= observed richness and bounded by the background supply.
      - Are built by:
          1) Selecting a support of HC50 levels from the leave-one-node-out
             background with availability-tempered weights.
          2) Drawing θ ~ Dirichlet(τ * p_bg_support).
          3) Drawing counts over that support using counts_mode.
          4) Sampling actual isolate IDs without replacement, respecting capacity.

    Parameters
    ----------
    df : DataFrame
        Full dataset with columns [node_col, level_col, id_col].
    target_node : str
        Node for which we generate synthetic pseudo-nodes.
    tau : float
        Calibrated τ*(node).
    B_pseudo : int
        Number of pseudo-nodes to attempt to generate.
    node_col, level_col, id_col : str
        Column names.
    alpha_support : float, default 0.10
        Availability-tempering exponent.
    counts_mode : {"positive","nonnegative","capacity_tempered"}
        How to simulate counts.
    min_avail_per_level : int, default 1
        Minimum background isolates per level to consider eligible.
    rng_seed : int, default 123
        RNG seed.
    max_resample_tries : int, default 200
        Max attempts per pseudo-node to find a feasible draw (capacity, etc.).

    Returns
    -------
    pseudo_nodes : list[list[str]]
        Each element is a list of isolate IDs for one pseudo-node.
    diag : dict
        Diagnostics, including:
          - n_obs, K_obs, K_bg, C_eligible, K_star
          - draws_requested, draws_generated
          - supply_fail_rate, cap_binding_rate
          - distance_to_bg_summary: median + IQR of distances to background.
    """
    counts_mode = _validate_counts_mode(counts_mode)

    df = _ensure_id_col(df, id_col=id_col)
    target_node = str(target_node)

    df_node = df.loc[df[node_col] == target_node].copy()
    df_bg = df.loc[df[node_col] != target_node].copy()
    if df_node.empty:
        raise ValueError(f"Target node {target_node!r} has no rows in df.")

    levels = sorted(df[level_col].astype(str).unique().tolist())
    n_v = int(df_node.shape[0])
    K_obs = int(df_node[level_col].nunique())

    # Background frequencies and availability
    bg_counts_full = (
        df_bg[level_col].astype(str).value_counts().reindex(levels, fill_value=0).astype(int)
    )
    avail = bg_counts_full.values
    p_bg_full = _normalize(bg_counts_full.values, eps=0.0)

    eligible_mask = (avail >= int(min_avail_per_level)).astype(float)
    K_bg = int((p_bg_full > 0).sum())
    C_eligible = int(eligible_mask.sum())
    K_star = int(min(K_obs, K_bg, C_eligible, n_v))

    if K_star <= 0 or p_bg_full.sum() == 0:
        return [], {
            "n_obs": n_v,
            "K_obs": K_obs,
            "K_bg": K_bg,
            "C_eligible": C_eligible,
            "K_star": K_star,
            "draws_requested": B_pseudo,
            "draws_generated": 0,
            "supply_fail_rate": 1.0,
            "cap_binding_rate": np.nan,
            "distance_to_bg_summary": {"median": np.nan, "iqr": np.nan},
            "warning": "No feasible support for pseudo-node generation.",
        }

    # Availability-tempered support weights
    w_support = (p_bg_full * np.clip(avail, 1, None) ** float(alpha_support)) * eligible_mask
    if (w_support > 0).sum() < 1:
        w_support = eligible_mask.copy()

    ids_by_level = build_ids_by_level(df_bg, level_col=level_col, id_col=id_col)

    rng = np.random.default_rng(rng_seed)

    pseudo_nodes: List[List[str]] = []
    d_bg_vals: List[float] = []
    total_attempts = 0
    total_failures = 0
    cap_bind_hits = 0
    cap_bind_trials = 0

    # Observed node frequency for reference (L1O background distance)
    obs_counts = _hist(levels, df_node[level_col])
    p_node = _normalize(obs_counts, eps=0.0)
    # distance to global background (L1O) for reference
    # (not required here, but useful to interpret diag)
    _ = _cosine_distance(p_node, p_bg_full + EPS)

    for _draw in range(B_pseudo):
        success = False
        for _try in range(max_resample_tries):
            total_attempts += 1

            # 1) choose support
            try:
                support_idx = sample_support(rng, w_support, K_star)
            except ValueError:
                total_failures += 1
                continue

            p_bg_support = p_bg_full[support_idx]
            if p_bg_support.sum() <= 0:
                total_failures += 1
                continue

            # 2) DM parameters
            alpha_vec = float(tau) * (p_bg_support / p_bg_support.sum())
            theta = rng.dirichlet(alpha_vec)

            # 3) sample counts
            if counts_mode == "positive":
                if n_v < len(support_idx):
                    total_failures += 1
                    continue
                counts_support = positive_multinomial_counts(rng, n_v, theta)
                # capacity constraint
                if np.any(counts_support > avail[support_idx]):
                    total_failures += 1
                    cap_bind_trials += 1
                    cap_bind_hits += 1
                    continue
            elif counts_mode == "capacity_tempered":
                cap_support = avail[support_idx].astype(int)
                cap_bind_trials += 1
                c = capacity_tempered_counts(rng, n_v, theta, cap_support)
                if c is None:
                    total_failures += 1
                    cap_bind_hits += 1
                    continue
                counts_support = c
                if np.any(counts_support >= cap_support):
                    cap_bind_hits += 1
            else:  # "nonnegative"
                counts_support = nonnegative_multinomial_counts(rng, n_v, theta)
                if np.any(counts_support > avail[support_idx]):
                    total_failures += 1
                    cap_bind_trials += 1
                    cap_bind_hits += 1
                    continue

            # 4) embed into full vector and compute distance to background
            pseudo_counts = np.zeros(len(levels), dtype=int)
            pseudo_counts[support_idx] = counts_support
            d_bg = _cosine_distance(_normalize(pseudo_counts, eps=0.0), p_bg_full + EPS)

            # 5) sample actual isolate IDs consistent with counts
            counts_by_idx = {int(support_idx[j]): int(counts_support[j]) for j in range(len(support_idx))}
            chosen = sample_isolates_without_replacement(
                rng=rng,
                ids_by_level=ids_by_level,
                counts_by_level_index=counts_by_idx,
                levels=levels,
            )
            if chosen is None:
                total_failures += 1
                continue

            pseudo_nodes.append(chosen)
            d_bg_vals.append(float(d_bg))
            success = True
            break

        if not success:
            # couldn't make this pseudo-node in allotted resamples
            continue

    draws_generated = len(pseudo_nodes)
    fail_rate = (total_failures / max(1, total_attempts)) if total_attempts else 0.0
    if len(d_bg_vals) >= 1:
        med = float(np.median(d_bg_vals))
        iqr = _iqr(d_bg_vals)
    else:
        med, iqr = (np.nan, np.nan)

    cap_binding_rate = float(cap_bind_hits / cap_bind_trials) if cap_bind_trials > 0 else 0.0

    diag = {
        "n_obs": n_v,
        "K_obs": K_obs,
        "K_bg": K_bg,
        "C_eligible": C_eligible,
        "K_star": K_star,
        "tau": float(tau),
        "alpha_support": float(alpha_support),
        "counts_mode": counts_mode,
        "min_avail_per_level": int(min_avail_per_level),
        "draws_requested": int(B_pseudo),
        "draws_generated": int(draws_generated),
        "supply_fail_rate": float(fail_rate),
        "cap_binding_rate": float(cap_binding_rate),
        "distance_to_bg_summary": {"median": med, "iqr": iqr},
    }
    if draws_generated < 0.8 * B_pseudo:
        diag["warning"] = (
            "Low yield: consider reducing K_star (e.g. lower richness), "
            "using capacity_tempered counts, or pooling rare HC50s."
        )

    return pseudo_nodes, diag


# ---------------------------------
# 3) QC and readiness for Step 3
# ---------------------------------


def compile_qc_single_node(
    df: pd.DataFrame,
    df_bg: pd.DataFrame,
    target_node: str,
    *,
    node_col: str,
    level_col: str,
    id_col: str,
    pseudo_nodes: List[List[str]],
    tau: float,
    alpha_support: float,
    counts_mode: str,
    min_avail_per_level: int,
    diag_gen: Dict[str, object],
) -> pd.DataFrame:
    """
    Build a QC table for a single node comparing observed vs synthetic.

    The output has one row with:
      - observed n, richness, Shannon, evenness, L1O cosine
      - background richness / availability
      - synthetic richness median/IQR, L1O cosine median/IQR
      - gaps in medians (delta_med_l1o, delta_med_richness)
    """
    df = _ensure_id_col(df, id_col=id_col)
    target_node = str(target_node)

    levels = sorted(df[level_col].astype(str).unique().tolist())

    df_node = df.loc[df[node_col] == target_node].copy()
    if df_node.empty:
        raise ValueError(f"Target node {target_node!r} has no rows in df.")

    # Observed stats
    obs_n = int(df_node.shape[0])
    obs_K = int(df_node[level_col].nunique())
    obs_counts = _hist(levels, df_node[level_col])
    obs_H, obs_J = _shannon_evenness(obs_counts)

    # Background
    bg_counts = (
        df_bg[level_col].astype(str).value_counts().reindex(levels, fill_value=0).astype(int)
    )
    K_bg = int((bg_counts.values > 0).sum())
    C_eligible = int((bg_counts.values >= int(min_avail_per_level)).sum())

    p_node = _normalize(obs_counts, eps=0.0)
    p_bg = _normalize(bg_counts.values, eps=0.0)
    obs_l1o = _cosine_distance(p_node, p_bg)

    # Synthetic summaries
    id_to_level = dict(zip(df[id_col].astype(str), df[level_col].astype(str)))

    syn_rich_vals: List[int] = []
    d_l1o_vals: List[float] = []
    pseudo_H_vals: List[float] = []
    pseudo_J_vals: List[float] = []

    for ids in pseudo_nodes:
        levs = [id_to_level.get(str(x)) for x in ids]
        counts = (
            pd.Series(levs, dtype=str)
            .value_counts()
            .reindex(levels, fill_value=0)
            .astype(int)
            .values
        )
        syn_rich_vals.append(int((counts > 0).sum()))
        H, J = _shannon_evenness(counts)
        pseudo_H_vals.append(H)
        pseudo_J_vals.append(J)

        p_syn = _normalize(counts, eps=0.0)
        d = _cosine_distance(p_syn, p_bg)
        d_l1o_vals.append(float(d))

    syn_rich_med = float(np.median(syn_rich_vals)) if syn_rich_vals else np.nan
    syn_rich_iqr = _iqr(syn_rich_vals) if syn_rich_vals else np.nan
    syn_l1o_med = float(np.median(d_l1o_vals)) if d_l1o_vals else np.nan
    syn_l1o_iqr = _iqr(d_l1o_vals) if d_l1o_vals else np.nan
    pseudo_H_med = float(np.median(pseudo_H_vals)) if pseudo_H_vals else np.nan
    pseudo_H_iqr = _iqr(pseudo_H_vals) if pseudo_H_vals else np.nan
    pseudo_J_med = float(np.median(pseudo_J_vals)) if pseudo_J_vals else np.nan
    pseudo_J_iqr = _iqr(pseudo_J_vals) if pseudo_J_vals else np.nan

    row = dict(
        node=target_node,
        obs_n=obs_n,
        obs_K=obs_K,
        obs_shannon=obs_H,
        obs_evenness=obs_J,
        obs_l1o_cosine=obs_l1o,
        K_bg=K_bg,
        C_eligible=C_eligible,
        tau=float(tau),
        alpha_support=float(alpha_support),
        counts_mode=counts_mode,
        min_avail_per_level=int(min_avail_per_level),
        draws_requested=int(diag_gen.get("draws_requested", 0)),
        draws_generated=int(diag_gen.get("draws_generated", 0)),
        yield_rate=float(
            int(diag_gen.get("draws_generated", 0)) / max(1, int(diag_gen.get("draws_requested", 1)))
        ),
        supply_fail_rate=float(diag_gen.get("supply_fail_rate", np.nan)),
        syn_richness_med=syn_rich_med,
        syn_richness_iqr=syn_rich_iqr,
        syn_l1o_cosine_med=syn_l1o_med,
        syn_l1o_cosine_iqr=syn_l1o_iqr,
        delta_med_l1o=(
            abs(syn_l1o_med - obs_l1o)
            if (not np.isnan(syn_l1o_med) and not np.isnan(obs_l1o))
            else np.nan
        ),
        delta_med_richness=(
            abs(syn_rich_med - obs_K)
            if not np.isnan(syn_rich_med)
            else np.nan
        ),
        pseudo_shannon_med=pseudo_H_med,
        pseudo_shannon_iqr=pseudo_H_iqr,
        pseudo_evenness_med=pseudo_J_med,
        pseudo_evenness_iqr=pseudo_J_iqr,
    )

    qc_df = pd.DataFrame([row])
    return qc_df


def evaluate_ready_single_node(
    qc_df: pd.DataFrame,
    *,
    median_tol: float = 0.10,
    require_min_yield: float = 0.80,
) -> bool:
    """
    Decide whether the synthetic node ensemble is "good enough" for Step 3.

    Criteria (all must pass):
      - |syn_l1o_cosine_med - obs_l1o_cosine| <= median_tol
      - yield_rate >= require_min_yield
      - (optional) you can add further checks here later.
    """
    if qc_df.empty:
        return False
    row = qc_df.iloc[0]

    delta = float(row["delta_med_l1o"])
    yield_rate = float(row["yield_rate"])

    distance_ok = (not np.isnan(delta)) and (abs(delta) <= float(median_tol))
    yield_ok = (yield_rate >= float(require_min_yield))

    ready = bool(distance_ok and yield_ok)
    return ready


# ---------------------------------------------
# 4) Top-level single-node Step 2 convenience
# ---------------------------------------------


def run_step2_dm_single_node(
    df: pd.DataFrame,
    *,
    target_node: str,
    taus: Tuple[float, ...],
    B_calib: int,
    B_pseudo: int,
    node_col: str,
    level_col: str,
    id_col: str,
    alpha_support: float = 0.10,
    counts_mode: str = "positive",
    min_avail_per_level: int = 1,
    rng_seed: int = 123,
    median_tol: float = 0.10,
    require_min_yield: float = 0.80,
) -> Tuple[pd.DataFrame, bool, Dict[str, object]]:
    """
    Single-node Step 2 pipeline (DM-based synthetic node generation).

    This is a stripped-down version of the multi-node Step 2 logic, intended
    for focused evaluation of a single node (e.g. "animal_feed"):

      1) Calibrate τ*(node) on a τ-grid via L1O cosine distance.
      2) Generate B_pseudo pseudo-nodes under a Dirichlet–Multinomial model
         using the leave-one-node-out background.
      3) Build a one-row QC table summarizing observed vs synthetic.
      4) Decide if the synthetic ensemble is "good enough" for Step 3, based on:
           - distance match (delta_med_l1o <= median_tol)
           - yield >= require_min_yield

    Parameters
    ----------
    df : DataFrame
        Full dataset with columns [node_col, level_col, id_col].
    target_node : str
        Node to evaluate (e.g. "animal_feed").
    taus : tuple of float
        Grid of τ values for calibration.
    B_calib : int
        Number of calibration pseudo-draws per τ.
    B_pseudo : int
        Number of pseudo-nodes to generate after calibration.
    node_col, level_col, id_col : str
        Column names in `df`.
    alpha_support : float, default 0.10
        Availability-tempering exponent.
    counts_mode : {"positive","nonnegative","capacity_tempered"}
        How to draw counts in DM sampling.
    min_avail_per_level : int, default 1
        Minimum isolates required in background for a level to be eligible.
    rng_seed : int, default 123
        Base RNG seed (used for both calibration and generation).
    median_tol : float, default 0.10
        Allowed absolute difference between observed and synthetic L1O medians.
    require_min_yield : float, default 0.80
        Minimum acceptable pseudo-node yield fraction.

    Returns
    -------
    qc_df : DataFrame
        One-row QC table with observed vs synthetic stats for `target_node`.
    ready_for_step3 : bool
        Whether this synthetic ensemble is acceptable for use in Step 3.
    info : dict
        Misc diagnostics including:
          - "tau_star" (best τ)
          - "calibration" (output from calibrate_tau_for_single_node)
          - "generation_diag" (output from generate_pseudonodes_single_node)

    Example
    -------
    >>> TAU_GRID = (0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20,
    ...             0.30, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0)
    >>> qc_df_af, ready_af, info_af = run_step2_dm_single_node(
    ...     df=df_all_training_testing_no_singletons_suffixed,
    ...     target_node="animal_feed",
    ...     taus=TAU_GRID,
    ...     B_calib=1200,
    ...     B_pseudo=1000,
    ...     node_col="curated_source_region",
    ...     level_col="HC50",
    ...     id_col="biosample_acc",
    ...     alpha_support=0.10,
    ...     counts_mode="positive",
    ...     min_avail_per_level=1,
    ...     rng_seed=123,
    ...     median_tol=0.10,
    ...     require_min_yield=0.80,
    ... )
    >>> print(ready_af)
    True
    >>> qc_df_af[[
    ...   "obs_l1o_cosine", "syn_l1o_cosine_med", "delta_med_l1o",
    ...   "obs_K", "syn_richness_med", "delta_med_richness",
    ...   "yield_rate"
    ... ]]
    """
    counts_mode = _validate_counts_mode(counts_mode)

    df = _ensure_id_col(df, id_col=id_col)
    target_node = str(target_node)

    # 1) separate background and node
    df_node = df.loc[df[node_col] == target_node].copy()
    df_bg = df.loc[df[node_col] != target_node].copy()
    if df_node.empty:
        raise ValueError(f"Target node {target_node!r} has no rows in df.")

    # 2) calibrate τ*(node)
    tau_star, median_by_tau, calib_info = calibrate_tau_for_single_node(
        df=df,
        target_node=target_node,
        taus=taus,
        B_calib=B_calib,
        node_col=node_col,
        level_col=level_col,
        id_col=id_col,
        alpha_support=alpha_support,
        counts_mode=counts_mode,
        min_avail_per_level=min_avail_per_level,
        rng_seed=rng_seed,
    )
    if tau_star is None:
        # Calibration failed badly; build a minimal QC row and mark as not ready.
        qc_df = compile_qc_single_node(
            df=df,
            df_bg=df_bg,
            target_node=target_node,
            node_col=node_col,
            level_col=level_col,
            id_col=id_col,
            pseudo_nodes=[],
            tau=np.nan,
            alpha_support=alpha_support,
            counts_mode=counts_mode,
            min_avail_per_level=min_avail_per_level,
            diag_gen={
                "draws_requested": B_pseudo,
                "draws_generated": 0,
                "supply_fail_rate": 1.0,
            },
        )
        return qc_df, False, {
            "tau_star": None,
            "calibration": calib_info,
            "generation_diag": {
                "draws_requested": B_pseudo,
                "draws_generated": 0,
                "supply_fail_rate": 1.0,
            },
        }

    # 3) generate pseudo-nodes
    pseudo_nodes, diag_gen = generate_pseudonodes_single_node(
        df=df,
        target_node=target_node,
        tau=tau_star,
        B_pseudo=B_pseudo,
        node_col=node_col,
        level_col=level_col,
        id_col=id_col,
        alpha_support=alpha_support,
        counts_mode=counts_mode,
        min_avail_per_level=min_avail_per_level,
        rng_seed=rng_seed + 7,
    )

    # 4) QC table
    qc_df = compile_qc_single_node(
        df=df,
        df_bg=df_bg,
        target_node=target_node,
        node_col=node_col,
        level_col=level_col,
        id_col=id_col,
        pseudo_nodes=pseudo_nodes,
        tau=tau_star,
        alpha_support=alpha_support,
        counts_mode=counts_mode,
        min_avail_per_level=min_avail_per_level,
        diag_gen=diag_gen,
    )

    # 5) readiness flag
    ready = evaluate_ready_single_node(
        qc_df=qc_df,
        median_tol=median_tol,
        require_min_yield=require_min_yield,
    )

    info = {
        "tau_star": float(tau_star),
        "calibration": calib_info,
        "generation_diag": diag_gen,
    }
    return qc_df, ready, info


## Identify testable and untestable nodes for H0 and H1
# --------------------------------------------
# Global tau grid 
# --------------------------------------------
TAU_GRID_GENERIC = (
    0.1, 0.3, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 30, 40, 50
)
def screen_nodes_for_dm_testability(
    df,
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
    nodes=None,
):
    """
    Run Step 2 for many nodes and identify those that are not testable for H0/H1.

    "Human-like" structural non-testability:
      - tau_star is None
      - OR draws_generated == 0
      - OR supply_fail_rate == 1.0

    Additional QC non-testability (even if structural DM is feasible):
      - ready_step3 == False
      - OR yield_rate < require_min_yield
      - OR delta_med_l1o > median_tol

    Parameters
    ----------
    df : pd.DataFrame
        Isolate-level data (same preprocessing as Step 2, e.g.
        df_all_training_testing_no_singletons_suffixed).
    node_col : str
        Column with node labels (e.g. "curated_source_region").
    level_col : str
        Column with HC levels (e.g. "HC50").
    id_col : str
        Column with isolate IDs (e.g. "biosample_acc").
    taus, B_calib, B_pseudo, alpha_support, counts_mode,
    min_avail_per_level, median_tol, require_min_yield, rng_seed :
        Same as your run_step2_dm_single_node configuration.
    nodes : list or None
        Subset of nodes to test. If None, test all unique values in node_col.

    Returns
    -------
    results_df : pd.DataFrame
        One row per node with:
          - obs_n, obs_K, K_bg, K_star, tau_star, yield_rate, delta_med_l1o
          - draws_requested, draws_generated, supply_fail_rate
          - ready_step3
          - structural_not_testable (boolean, "like human")
          - qc_not_testable (boolean, Step 2 QC failure)
          - not_testable_overall (structural OR qc)
          - reason_structural, reason_qc
          - error (if Step 2 threw an exception)
    """

    # which nodes to screen
    if nodes is None:
        nodes = sorted(df[node_col].unique().tolist())

    records = []

    for node in nodes:
        print(f"\n[Step 2 screening] Node: {node}")
        try:
            qc_df, ready_step3, info = run_step2_dm_single_node(
                df=df,
                target_node=node,
                taus=taus,
                B_calib=B_calib,
                B_pseudo=B_pseudo,
                node_col=node_col,
                level_col=level_col,
                id_col=id_col,
                alpha_support=alpha_support,
                counts_mode=counts_mode,
                min_avail_per_level=min_avail_per_level,
                rng_seed=rng_seed,
                median_tol=median_tol,
                require_min_yield=require_min_yield,
            )
        except Exception as e:
            # Hard failure for this node
            records.append(
                dict(
                    node=node,
                    obs_n=np.nan,
                    obs_K=np.nan,
                    K_bg=np.nan,
                    K_star=np.nan,
                    tau_star=np.nan,
                    yield_rate=np.nan,
                    delta_med_l1o=np.nan,
                    draws_requested=np.nan,
                    draws_generated=np.nan,
                    supply_fail_rate=np.nan,
                    ready_step3=False,
                    structural_not_testable=True,
                    qc_not_testable=True,
                    not_testable_overall=True,
                    reason_structural=f"Exception in Step 2: {e}",
                    reason_qc="Step 2 did not complete",
                    error=str(e),
                )
            )
            continue

        # Extract QC row
        if qc_df.shape[0] != 1:
            raise ValueError(f"Expected QC df to have 1 row for node={node}, got {qc_df.shape[0]}")

        qc_row = qc_df.iloc[0]

        # Extract generation diagnostics
        gen_diag = info.get("generation_diag", {})
        calib = info.get("calibration", {})

        tau_star = info.get("tau_star", None)
        draws_requested = gen_diag.get("draws_requested", np.nan)
        draws_generated = gen_diag.get("draws_generated", np.nan)
        supply_fail_rate = gen_diag.get("supply_fail_rate", np.nan)

        obs_n = int(qc_row.get("obs_n", np.nan))
        obs_K = int(qc_row.get("obs_K", np.nan))
        K_bg = int(qc_row.get("K_bg", calib.get("K_bg", np.nan)))
        K_star = int(calib.get("K_star", gen_diag.get("K_star", np.nan)))

        yield_rate = float(qc_row.get("yield_rate", np.nan))
        delta_med_l1o = float(qc_row.get("delta_med_l1o", np.nan))

        # -----------------------------
        # Structural non-testability
        # -----------------------------
        structural_reasons = []
        if tau_star is None:
            structural_reasons.append("tau_star is None")
        if draws_generated == 0:
            structural_reasons.append("draws_generated == 0")
        if isinstance(supply_fail_rate, (int, float)) and supply_fail_rate == 1.0:
            structural_reasons.append("supply_fail_rate == 1.0")

        structural_not_testable = len(structural_reasons) > 0
        reason_structural = "; ".join(structural_reasons) if structural_reasons else ""

        # -----------------------------
        # QC non-testability (Step 2)
        # -----------------------------
        qc_reasons = []
        if not bool(ready_step3):
            qc_reasons.append("ready_step3 == False")
        # (These two are effectively part of ready_step3 in your code, but
        #  we record them explicitly for clarity.)
        if not np.isnan(yield_rate) and yield_rate < require_min_yield:
            qc_reasons.append(f"yield_rate < require_min_yield ({yield_rate:.3f} < {require_min_yield})")
        if not np.isnan(delta_med_l1o) and delta_med_l1o > median_tol:
            qc_reasons.append(f"delta_med_l1o > median_tol ({delta_med_l1o:.3f} > {median_tol})")

        qc_not_testable = len(qc_reasons) > 0
        reason_qc = "; ".join(qc_reasons) if qc_reasons else ""

        records.append(
            dict(
                node=node,
                obs_n=obs_n,
                obs_K=obs_K,
                K_bg=K_bg,
                K_star=K_star,
                tau_star=tau_star,
                yield_rate=yield_rate,
                delta_med_l1o=delta_med_l1o,
                draws_requested=draws_requested,
                draws_generated=draws_generated,
                supply_fail_rate=supply_fail_rate,
                ready_step3=bool(ready_step3),
                structural_not_testable=structural_not_testable,
                qc_not_testable=qc_not_testable,
                not_testable_overall=(structural_not_testable or qc_not_testable),
                reason_structural=reason_structural,
                reason_qc=reason_qc,
                error=None,
            )
        )

    results_df = pd.DataFrame(records)
    return results_df


### A wrapper function to run Step 2 and stability analysis for all testable nodes
import numpy as np
import pandas as pd

# ------------------------------------------------
# Shared setup
# ------------------------------------------------
TAU_GRID_GENERIC = (
    0.1, 0.3, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 30, 40, 50
)

STABILITY_SEEDS = [
    101, 123, 202, 303, 404, 505, 606, 707,
    808, 909, 1010, 2025, 3030, 4040, 5050, 6060
]


def run_step2_and_stability_for_node(
    target_node,
    df,
    taus=TAU_GRID_GENERIC,
    B_calib=1200,
    B_pseudo=2000,
    node_col="curated_source_region",
    level_col="HC50",
    id_col="biosample_acc",
    alpha_support=0.10,
    counts_mode="positive",
    min_avail_per_level=1,
    median_tol=0.10,
    require_min_yield=0.80,
    base_seed=123,
):
    """
    Run Step 2 calibration + stability test for a single node.

    Returns
    -------
    out : dict with keys
      - 'qc'      : Step 2 QC dataframe (1 row)
      - 'ready'   : bool, ready for Step 3 under base_seed
      - 'info'    : Step 2 info dict for base_seed
      - 'stability_df' : DataFrame of stability runs over STABILITY_SEEDS
    """

    print("\n" + "#" * 70)
    print(f"[Step 2] Node: {target_node}")
    print("#" * 70)

    # -------------------------
    # 1) Main Step 2 run (base_seed)
    # -------------------------
    qc, ready, info = run_step2_dm_single_node(
        df=df,
        target_node=target_node,
        taus=taus,
        B_calib=B_calib,
        B_pseudo=B_pseudo,
        node_col=node_col,
        level_col=level_col,
        id_col=id_col,
        alpha_support=alpha_support,
        counts_mode=counts_mode,
        min_avail_per_level=min_avail_per_level,
        rng_seed=base_seed,
        median_tol=median_tol,
        require_min_yield=require_min_yield,
    )

    print(f"\n[{target_node}] Ready for Step 3? {ready}")
    print(f"[{target_node}] Calibrated tau*: {info.get('tau_star', None)}")

    cols_to_show = [
        "obs_n", "obs_K",
        "obs_l1o_cosine", "syn_l1o_cosine_med", "delta_med_l1o",
        "obs_shannon", "obs_evenness",
        "syn_richness_med", "delta_med_richness",
        "yield_rate",
    ]
    print(f"\n[{target_node}] QC summary:")
    cols_to_show_existing = [c for c in cols_to_show if c in qc.columns]
    print(qc[cols_to_show_existing].to_string(index=False))

    print(f"\n[{target_node}] Generation diagnostics:")
    print(info.get("generation_diag", {}))

    # -------------------------
    # 2) Stability runs over STABILITY_SEEDS
    # -------------------------
    stability_results = []

    print(f"\n[{target_node}] Stability test over {len(STABILITY_SEEDS)} seeds")

    for i, seed in enumerate(STABILITY_SEEDS, start=1):
        qc_s, ready_s, info_s = run_step2_dm_single_node(
            df=df,
            target_node=target_node,
            taus=taus,
            B_calib=B_calib,
            B_pseudo=B_pseudo,
            node_col=node_col,
            level_col=level_col,
            id_col=id_col,
            alpha_support=alpha_support,
            counts_mode=counts_mode,
            min_avail_per_level=min_avail_per_level,
            rng_seed=seed,
            median_tol=median_tol,
            require_min_yield=require_min_yield,
        )

        tau_star = info_s.get("tau_star", None)
        cal = info_s.get("calibration", {})
        D_obs = float(cal.get("D_obs", np.nan))

        syn_med = float(qc_s["syn_l1o_cosine_med"].iloc[0]) if "syn_l1o_cosine_med" in qc_s else np.nan
        delta = float(qc_s["delta_med_l1o"].iloc[0]) if "delta_med_l1o" in qc_s else np.nan
        yield_rate = float(qc_s["yield_rate"].iloc[0]) if "yield_rate" in qc_s else np.nan

        print(
            f"[Stability {target_node}] ({i}/{len(STABILITY_SEEDS)}) "
            f"seed={seed} -> tau_star={tau_star}, ready={ready_s}, "
            f"D_obs={D_obs:.4f}, syn_med={syn_med:.4f}, "
            f"delta_med_l1o={delta:.4f}, yield={yield_rate:.3f}"
        )

        stability_results.append(dict(
            node=target_node,
            seed=int(seed),
            tau_star=float(tau_star) if tau_star is not None else np.nan,
            D_obs=D_obs,
            syn_l1o_cosine_med=syn_med,
            delta_med_l1o=delta,
            yield_rate=yield_rate,
            ready_step3=bool(ready_s),
        ))

    stability_df = pd.DataFrame(stability_results)

    print(f"\n[Stability {target_node}] Summary statistics:")
    print(stability_df.describe(include="all"))

    print(f"\n[Stability {target_node}] Tau* distribution across seeds:")
    print(stability_df["tau_star"].value_counts().sort_index())

    print(f"\n[Stability {target_node}] Ready_step3, delta, and yield by seed:")
    print(stability_df[["seed", "ready_step3", "delta_med_l1o", "yield_rate"]])

    return dict(
        qc=qc,
        ready=ready,
        info=info,
        stability_df=stability_df,
    )


def run_step2_and_stability_for_all(
    testable_nodes,
    df,
    taus=TAU_GRID_GENERIC,
    B_calib=1200,
    B_pseudo=2000,
    node_col="curated_source_region",
    level_col="HC50",
    id_col="biosample_acc",
    alpha_support=0.10,
    counts_mode="positive",
    min_avail_per_level=1,
    median_tol=0.10,
    require_min_yield=0.80,
    base_seed=123,
):
    """
    Wrapper to run Step 2 + stability test for all nodes in testable_nodes.

    For each node N in testable_nodes, this will create three
    top-level variables:

      N_res  : full result dict (qc, ready, info, stability_df)
      N_info : Step 2 info dict (for Step 3)
      N_stab : stability DataFrame

    Returns
    -------
    all_results : dict
        {
          node_name: {
             'qc': qc_df,
             'ready': bool,
             'info': info_dict,
             'stability_df': DataFrame
          },
          ...
        }
    """
    all_results = {}

    for node in testable_nodes:
        res = run_step2_and_stability_for_node(
            target_node=node,
            df=df,
            taus=taus,
            B_calib=B_calib,
            B_pseudo=B_pseudo,
            node_col=node_col,
            level_col=level_col,
            id_col=id_col,
            alpha_support=alpha_support,
            counts_mode=counts_mode,
            min_avail_per_level=min_avail_per_level,
            median_tol=median_tol,
            require_min_yield=require_min_yield,
            base_seed=base_seed,
        )
        all_results[node] = res

        # Make sure node name is a valid Python identifier (your nodes already look fine)
        var_prefix = node  # if needed: node.replace("-", "_").replace(" ", "_")

        # Create: <node>_res, <node>_info, <node>_stab in the global namespace
        globals()[f"{var_prefix}_res"] = res
        globals()[f"{var_prefix}_info"] = res["info"]
        globals()[f"{var_prefix}_stab"] = res["stability_df"]

        print(f"\nCreated variables: {var_prefix}_res, {var_prefix}_info, {var_prefix}_stab")

    return all_results


#### Functions for Step 3 for a single node
import numpy as np
import pandas as pd
import networkx as nx

# ============================================================
# Step 3: Single-node network test (example: "animal_feed")
# ============================================================

def build_step3_config_from_step2(
    qc_df: pd.DataFrame,
    info: dict,
    target_node: str,
    ready_step3: bool = True,
) -> dict:
    """
    Extract the minimal configuration needed for Step 3 from
    the Step 2 outputs for a single target node.
    """
    if qc_df.shape[0] != 1:
        raise ValueError("Expected qc_df to have exactly 1 row for a single node.")

    qc_row = qc_df.iloc[0].to_dict()
    gen = info["generation_diag"]

    config = dict(
        node=target_node,
        ready_step3=bool(ready_step3),
        # Observed DM / richness info
        obs_n=int(gen["n_obs"]),          # total isolates for the node
        obs_K=int(gen["K_obs"]),          # observed richness
        K_star=int(gen["K_star"]),        # richness used in DM generator (should match obs_K)
        K_bg=int(gen["K_bg"]),            # total background HC50
        C_eligible=int(gen["C_eligible"]),
        tau_star=float(info["tau_star"]), # calibrated tau*
        alpha_support=float(gen["alpha_support"]),
        counts_mode=str(gen["counts_mode"]),
        min_avail_per_level=int(gen["min_avail_per_level"]),
        # Some sanity / QC information from Step 2
        delta_med_l1o=float(qc_row["delta_med_l1o"]),
        syn_richness_med=float(qc_row["syn_richness_med"]),
        yield_rate=float(qc_row["yield_rate"]),
    )
    return config


def build_background_level_counts(
    df: pd.DataFrame,
    target_node: str,
    node_col: str,
    level_col: str,
    min_avail_per_level: int = 1,
) -> pd.Series:
    """
    Build background HC50 count distribution for all nodes except target_node.

    Returns a Series indexed by HC-level (level_col), with counts of isolates
    per HC, filtered to levels with count >= min_avail_per_level.
    """
    df_bg = df[df[node_col] != target_node]
    level_counts = df_bg[level_col].value_counts()
    if min_avail_per_level > 1:
        level_counts = level_counts[level_counts >= min_avail_per_level]
    return level_counts.sort_index()


def sample_dm_pseudo_composition(
    bg_counts: pd.Series,
    n_obs: int,
    K_star: int,
    tau_star: float,
    rng: np.random.RandomState,
) -> pd.Series:
    """
    Sample a single Dirichlet–multinomial pseudo-node composition.

    Simplified DM model:
      1. Choose K_star distinct HC-levels from background, without replacement,
         proportional to bg_counts.
      2. Let p_sub be the normalized weights for these chosen HCs.
      3. Draw p ~ Dirichlet(tau_star * p_sub).
      4. Draw counts ~ Multinomial(n_obs, p).
    """
    if len(bg_counts) < K_star:
        raise ValueError(f"Background has only {len(bg_counts)} levels, but K_star={K_star}.")

    levels_all = bg_counts.index.to_numpy()
    weights_all = bg_counts.values.astype(float)
    weights_all /= weights_all.sum()

    # 1) sample K_star levels without replacement
    chosen_idx = rng.choice(len(levels_all), size=K_star, replace=False, p=weights_all)
    chosen_levels = levels_all[chosen_idx]
    chosen_weights = weights_all[chosen_idx]
    chosen_weights /= chosen_weights.sum()

    # 2–4) Dirichlet–Multinomial draw
    alpha_vec = tau_star * chosen_weights
    p = rng.dirichlet(alpha_vec)
    counts = rng.multinomial(n_obs, p)

    return pd.Series(counts, index=chosen_levels)


def composition_series_to_vector(
    comp: pd.Series,
    feature_cols: pd.Index,
) -> np.ndarray:
    """
    Map a composition over HC-levels (index = HC IDs) to a fixed-length
    vector aligned to 'feature_cols' (the columns used in the network).

    Any HCs in comp that are not in feature_cols are ignored.
    """
    vec = np.zeros(len(feature_cols), dtype=float)
    loc = feature_cols.get_indexer(comp.index)
    mask = loc >= 0
    if mask.any():
        vec[loc[mask]] = comp.values[mask]
    return vec


# step 3 function with p-based z-scores
import numpy as np
from scipy.stats import norm   # make sure scipy is installed

def run_step3_dm_single_node(
    df: pd.DataFrame,
    G_obs: nx.Graph,
    target_node: str,
    step2_qc_df: pd.DataFrame,
    step2_info: dict,
    feature_col: str = "HC50",
    node_col: str = "curated_source_region",
    id_col: str = "biosample_acc",
    n_null: int = 2000,
    edge_threshold: float = 0.35, # cosine similarity threshold for edge creation
    rng_seed: int = 123,
) -> dict:
    """
    Step 3 network test for a single node (e.g. 'animal_feed').

    Uses the calibrated DM generator from Step 2 (tau_star, K_star, n_obs)
    to simulate null pseudo-nodes, embeds them in the observed graph, and
    compares the observed node’s centrality metrics to the null distributions
    via Monte Carlo p-values and **p-based z-scores**.
    """
    # ---- 1) Extract Step 3 config from Step 2 ----
    cfg = build_step3_config_from_step2(
        qc_df=step2_qc_df,
        info=step2_info,
        target_node=target_node,
        ready_step3=True,  # or pass your actual ready flag here
    )

    if not cfg["ready_step3"]:
        raise ValueError(f"Node {target_node} is not ready for Step 3 according to Step 2.")

    n_obs = cfg["obs_n"]
    K_star = cfg["K_star"]
    tau_star = cfg["tau_star"]
    min_avail_per_level = cfg["min_avail_per_level"]

    # ---- 2) Observed centrality metrics for the target node ----
    metrics_obs_all = node_metrics_weighted(G_obs, weighted=True, degree_normalize=True)
    if target_node not in metrics_obs_all.index:
        raise ValueError(f"Target node '{target_node}' not found in G_obs when computing metrics.")
    obs_metrics = metrics_obs_all.loc[target_node].copy()

    # ---- 3) Feature matrix for all nodes in G_obs ----
    feature_matrix = pd.pivot_table(
        df,
        index=node_col,
        columns=feature_col,
        aggfunc="size",
        fill_value=0,
    )

    # Restrict to nodes actually present in G_obs (and in the same order)
    node_order = list(G_obs.nodes())
    feature_matrix = feature_matrix.reindex(index=node_order).fillna(0)
    feature_cols = feature_matrix.columns

    # ---- 4) Background HC50 distribution excluding the target node ----
    bg_counts = build_background_level_counts(
        df=df,
        target_node=target_node,
        node_col=node_col,
        level_col=feature_col,
        min_avail_per_level=min_avail_per_level,
    )

    # ---- 5) Simulate null pseudo-nodes and compute their network metrics ----
    rng = np.random.RandomState(rng_seed)
    null_records = []

    X_nodes = feature_matrix.values  # shape: (n_nodes, n_HC)

    for b in range(n_null):
        # 5a) DM pseudo composition
        pseudo_comp = sample_dm_pseudo_composition(
            bg_counts=bg_counts,
            n_obs=n_obs,
            K_star=K_star,
            tau_star=tau_star,
            rng=rng,
        )

        # 5b) Map to HC50 feature vector aligned with feature_matrix columns
        pseudo_vec = composition_series_to_vector(pseudo_comp, feature_cols)

        # 5c) Cosine similarity between pseudo-node and all existing nodes
        stacked = np.vstack([pseudo_vec, X_nodes])
        sim_full = compute_similarity(stacked, similarity="cosine")
        # row 0 vs rows 1..N
        sim_row = sim_full[0, 1:]

        # 5d) Embed pseudo-node into a *copy* of the observed consensus graph
        pseudo_name = f"{target_node}__null_{b}"
        G_pseudo = G_obs.copy()
        G_pseudo.add_node(pseudo_name)

        for node_name, sim_val in zip(node_order, sim_row):
            if sim_val >= edge_threshold:
                G_pseudo.add_edge(pseudo_name, node_name, weight=float(sim_val))

        # 5e) Compute weighted centrality metrics for the pseudo-node
        metrics_pseudo = node_metrics_weighted(G_pseudo, weighted=True, degree_normalize=True)
        if pseudo_name not in metrics_pseudo.index:
            # Pseudo-node completely isolated: fall back to zeros.
            mrow = pd.Series(
                {
                    "degree": 0.0,
                    "strength": 0.0,
                    "clustering": 0.0,
                    "betweenness": 0.0,
                    "closeness": 0.0,
                    "eigenvector": 0.0,
                }
            )
        else:
            mrow = metrics_pseudo.loc[pseudo_name]

        null_records.append(
            dict(
                null_id=b,
                degree=float(mrow["degree"]),
                strength=float(mrow["strength"]),
                clustering=float(mrow["clustering"]),
                betweenness=float(mrow["betweenness"]),
                closeness=float(mrow["closeness"]),
                eigenvector=float(mrow["eigenvector"]),
            )
        )

    null_df = pd.DataFrame(null_records)

    # ---- 6) Empirical p-values and z-scores (REVISED) ----

    def one_sided_p(obs_val, null_vals, higher=True):
        """Upper- or lower-tail Monte Carlo p with +1 smoothing."""
        null_vals = np.asarray(null_vals, dtype=float)
        if higher:
            return float(((null_vals >= obs_val).sum() + 1) / (len(null_vals) + 1))
        else:
            return float(((null_vals <= obs_val).sum() + 1) / (len(null_vals) + 1))

    def two_sided_p(obs_val, null_vals):
        """Two-sided Monte Carlo p from both tails."""
        null_vals = np.asarray(null_vals, dtype=float)
        p_up = ((null_vals >= obs_val).sum() + 1) / (len(null_vals) + 1)
        p_lo = ((null_vals <= obs_val).sum() + 1) / (len(null_vals) + 1)
        return float(min(1.0, 2.0 * min(p_up, p_lo))), float(p_up), float(p_lo)

    def z_from_p_two_sided(obs_val, null_vals, p_two_sided):
        """
        Define a z-score from the *Monte Carlo* two-sided p-value:
            z = sign(obs - median(null)) * Phi^{-1}(1 - p/2)
        This avoids division by tiny SD when the null is nearly degenerate.
        """
        null_vals = np.asarray(null_vals, dtype=float)
        med = float(np.median(null_vals))

        diff = obs_val - med
        if np.isclose(diff, 0.0):
            return 0.0

        sign = np.sign(diff)

        # Guard against p == 0 or p == 1
        eps = 1e-12
        p_adj = min(max(p_two_sided, eps), 1 - eps)

        z_mag = norm.ppf(1.0 - p_adj / 2.0)
        return float(sign * z_mag)

    metrics = ["degree", "strength", "betweenness", "closeness", "eigenvector", "clustering"]

    pvals = {}
    zscores = {}

    for m in metrics:
        obs_val = float(obs_metrics[m])
        null_vals = null_df[m]

        p_two, p_up, p_lo = two_sided_p(obs_val, null_vals)
        z = z_from_p_two_sided(obs_val, null_vals, p_two)

        # Store detailed p's
        pvals[f"{m}_p_upper"] = p_up       # P(null >= obs)
        pvals[f"{m}_p_lower"] = p_lo       # P(null <= obs)
        pvals[f"{m}_p_two_sided"] = p_two  # symmetric deviation

        # Backward compatibility: keep original one-sided keys
        if m == "clustering":
            # originally lower-tail
            pvals[f"{m}_p"] = p_lo
        else:
            pvals[f"{m}_p"] = p_up

        zscores[f"{m}_z"] = z

    results = dict(
        target_node=target_node,
        config_step3=cfg,
        obs_metrics=obs_metrics,
        null_metrics=null_df,
        p_values=pvals,
        z_scores=zscores,
    )
    return results

#### Step 3 for all testable nodes
from typing import Dict, List, Union, Optional
import pandas as pd
import networkx as nx

# ------------------------------------------------------------
# Internal logger
# ------------------------------------------------------------
def _log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg, flush=True)

# ------------------------------------------------------------
# Helper: normalize target_nodes input
# ------------------------------------------------------------
def _parse_target_nodes(target_nodes: Union[str, List[str]]) -> List[str]:
    if target_nodes is None:
        return []
    if isinstance(target_nodes, (list, tuple, pd.Index)):
        return [str(x) for x in target_nodes]
    if isinstance(target_nodes, str):
        parts = [p.strip() for p in target_nodes.split(",")]
        return [p for p in parts if p]
    return [str(target_nodes)]


# ------------------------------------------------------------
# Helper: extract QC DataFrame from whatever *_res is
# ------------------------------------------------------------
def _extract_qc_df_from_res(
    qc_obj,
    node: str,
    qc_var_name: str,
    qc_subkey: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Turn whatever <node>_res is into the QC DataFrame expected by Step 3.

    qc_obj: the object stored in e.g. animal_feed_res
    qc_subkey: if not None, treat qc_obj as a dict and use qc_obj[qc_subkey]
    """
    # 1) If caller told us exactly which key to use
    if qc_subkey is not None:
        if not isinstance(qc_obj, dict):
            raise TypeError(
                f"{qc_var_name} for node '{node}' is not a dict, "
                f"but qc_subkey='{qc_subkey}' was provided. "
                f"Got type={type(qc_obj)}."
            )
        if qc_subkey not in qc_obj:
            raise KeyError(
                f"qc_subkey='{qc_subkey}' not found in {qc_var_name} keys: "
                f"{list(qc_obj.keys())}"
            )
        qc_df = qc_obj[qc_subkey]
        if not isinstance(qc_df, pd.DataFrame):
            raise TypeError(
                f"{qc_var_name}['{qc_subkey}'] for node '{node}' is not a DataFrame; "
                f"got type={type(qc_df)}."
            )
        _log(f"[Step3][{node}] Using QC DataFrame from {qc_var_name}['{qc_subkey}'] "
             f"with shape={qc_df.shape}.", verbose)
        return qc_df

    # 2) No qc_subkey: try to be smart
    # Direct DataFrame
    if isinstance(qc_obj, pd.DataFrame):
        _log(f"[Step3][{node}] {qc_var_name} is already a DataFrame with shape={qc_obj.shape}.",
             verbose)
        return qc_obj

    # Dict containing one or more DataFrames
    if isinstance(qc_obj, dict):
        df_candidates = [v for v in qc_obj.values() if isinstance(v, pd.DataFrame)]
        if not df_candidates:
            raise TypeError(
                f"{qc_var_name} for node '{node}' is a dict but contains no DataFrames. "
                f"Keys: {list(qc_obj.keys())}"
            )

        required_cols = {"delta_med_l1o", "syn_richness_med", "yield_rate"}
        for k, v in qc_obj.items():
            if isinstance(v, pd.DataFrame) and required_cols.issubset(set(v.columns)):
                _log(f"[Step3][{node}] Auto-detected QC DataFrame as {qc_var_name}['{k}'] "
                     f"with shape={v.shape}.", verbose)
                return v

        # Otherwise just return the first DataFrame we find
        qc_df = df_candidates[0]
        _log(f"[Step3][{node}] Auto-detected QC DataFrame as first DataFrame in {qc_var_name} "
             f"with shape={qc_df.shape}.", verbose)
        return qc_df

    # 3) Completely unexpected type
    raise TypeError(
        f"{qc_var_name} for node '{node}' must be a DataFrame or a dict of DataFrames; "
        f"got type={type(qc_obj)}."
    )


# ------------------------------------------------------------
# Main wrapper with logging
# ------------------------------------------------------------
def run_step3_for_target_nodes(
    target_nodes: Union[str, List[str]],
    df: pd.DataFrame,
    G_obs: nx.Graph,
    qc_suffix: str = "_res",
    info_suffix: str = "_info",
    *,
    qc_subkey: Optional[str] = None,  # e.g. "qc_final" if your *_res is a dict
    feature_col: str = "HC50",
    node_col: str = "curated_source_region",
    id_col: str = "biosample_acc",
    n_null: int = 2000,
    edge_threshold: float = 0.35,
    rng_seed: int = 123,
    verbose: bool = True,
) -> Dict[str, dict]:
    """
    Run Step 3 (run_step3_dm_single_node) for a set of nodes, using existing
    per-node Step 2 variables in the global namespace.

    For each node name N in target_nodes, expects the notebook to have:
        N + qc_suffix   -> Step 2 result object (DataFrame or dict), e.g. animal_feed_res
        N + info_suffix -> Step 2 info dict,           e.g. animal_feed_info

    If N + qc_suffix is a dict, this function will:
      - if qc_subkey is provided: use that key, e.g. res['qc_final']
      - else: auto-detect a DataFrame with columns
              {delta_med_l1o, syn_richness_med, yield_rate},
              or fall back to the first DataFrame in the dict.
    """
    nodes = _parse_target_nodes(target_nodes)
    if not nodes:
        raise ValueError("No target_nodes provided.")

    _log(f"[Step3] Starting Step 3 for {len(nodes)} node(s): {nodes}", verbose)
    _log(f"[Step3] Parameters: n_null={n_null}, edge_threshold={edge_threshold}, "
         f"rng_seed={rng_seed}", verbose)

    results_by_node: Dict[str, dict] = {}
    global_ns = globals()  # notebook-level namespace

    for i, node in enumerate(nodes, start=1):
        qc_name = f"{node}{qc_suffix}"
        info_name = f"{node}{info_suffix}"

        _log(f"\n[Step3][{node}] ({i}/{len(nodes)}) Processing node.", verbose)
        _log(f"[Step3][{node}] Expecting QC var='{qc_name}', info var='{info_name}'.", verbose)

        if qc_name not in global_ns:
            raise KeyError(f"Expected QC result variable '{qc_name}' not found in globals().")
        if info_name not in global_ns:
            raise KeyError(f"Expected info dict variable '{info_name}' not found in globals().")

        qc_obj = global_ns[qc_name]
        step2_info = global_ns[info_name]

        _log(f"[Step3][{node}] Found {qc_name} (type={type(qc_obj)}), "
             f"{info_name} (type={type(step2_info)}).", verbose)

        # Extract the actual QC DataFrame for Step 3
        step2_qc_df = _extract_qc_df_from_res(
            qc_obj=qc_obj,
            node=node,
            qc_var_name=qc_name,
            qc_subkey=qc_subkey,
            verbose=verbose,
        )

        _log(f"[Step3][{node}] Calling run_step3_dm_single_node() ...", verbose)

        # Call your existing Step 3 function
        res = run_step3_dm_single_node(
            df=df,
            G_obs=G_obs,
            target_node=node,
            step2_qc_df=step2_qc_df,
            step2_info=step2_info,
            feature_col=feature_col,
            node_col=node_col,
            id_col=id_col,
            n_null=n_null,
            edge_threshold=edge_threshold,
            rng_seed=rng_seed,
        )

        _log(f"[Step3][{node}] Finished. Stored results.", verbose)
        results_by_node[node] = res

    _log(f"\n[Step3] Completed Step 3 for all {len(nodes)} node(s).", verbose)
    return results_by_node


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