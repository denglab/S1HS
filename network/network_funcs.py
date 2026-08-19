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

# names for all 59 nodes
all_node_labels = {'animal_feed': 'Animal Feed',
 'aquatic_animal': 'Aquatic Animals',
 'companion_animal': 'Companion Animals',
 'equine': 'Equine',
 'fruit_and_vegetable': 'Fruits/Vegetables',
 'grains': 'Grains',
 'herbs': 'Herbs',
 'nuts': 'Nuts',
 'production_env_produce': 'FPE-Produce',
 'soil_northeast': 'Soil-NE',
 'wild_animal': 'Wild Animals',
 'dairy': 'Dairy',
 'food_production_env_pacific': 'FPE-PAC',
 'food_production_env_mountain': 'FPE-MTN',
 'food_production_env_southern_plains': 'FPE-SPLNS',
 'food_production_env_corn_belt': 'FPE-CB',
 'food_production_env_northeast': 'FPE-NE',
 'food_production_env_appalachian': 'FPE-APPAL',
 'food_production_env_na': 'FPE-N/A',
 'food_production_env_southeast': 'FPE-SE',
 'food_production_env_lake_states': 'FPE-LS',
 'food_production_env_northern_plains': 'FPE-NPLNS',
 'water_northeast': 'WTR-NE',
 'water_southeast': 'WTR-SE',
 'water_pacific': 'WTR-PAC',
 'water_appalachian': 'WTR-APPAL',
 'sewage_hi': 'Sewage-HI',
 'water_northern_plains': 'WTR-NPLNS',
 'water_southern_plains': 'WTR-SPLNS',
 'water_na': 'WTR-N/A',
 'forest': 'Forest',
 'multi_ingredient_other': 'MIF-Other',
 'multi_ingredient_poultry': 'MIF-Poultry',
 'rodents': 'Rodents',
 'rte': 'RTE',
 'poultry': 'Poultry',
 'bovine': 'Bovine',
 'swine': 'Swine',
 'human': 'Humans',
 #'produce_outbreak': 'Produce Outbreaks',
 'bird': 'Birds',
 'plant_based_low_water_activity_food': 'PLWF',
 'bsaaos': 'BSAAOs',
 'goat_sheep': 'Caprinae',
 'soil_southeast': 'Soil-SE',
 'soil_appalachian': 'Soil-APPAL',
 'water_corn_belt': 'WTR-CB',
 'multi_ingredient_pork_beef_dairy': 'MIF-PBD',
 'seeds': 'Seeds',
 'seasoning': 'Seasoning',
 'water_mountain': 'WTR-MTN',
 'soil_corn_belt': 'Soil-CB',
 'soil_pacific': 'Soil-PAC',
 'root_undergroudn': 'Root/Underground',
 'confectionery': 'Confectionery',
 'camelids': 'Camelids',
 'beans': 'Beans',
 'food_production_env_delta_states': 'FPE-DELTA',
 'soil_na': 'Soil-N/A',
 'water_lake_states': 'WTR-LS',
 'water': 'WTR-ALL',
 'soil': 'Soil-ALL',
 'food_production_env': 'FPE-ALL',
 # additional labels for AUC vs coverage plot
 'env_water': 'WTR-All',
 'aquatic_animals': 'Aquatic Animals',
 'env_unclear': 'FPE-All',
 'vegetable_snack_others': 'PLWF',
}

# names for nodes in consensus network
node_labels = {'animal_feed': 'Animal Feed',
 'aquatic_animal': 'Aquatic Animals',
 'companion_animal': 'Companion Animals',
 'equine': 'Equine',
 'fruit_and_vegetable': 'Fruits Vegetables',
 'grains': 'Grains',
 'herbs': 'Herbs',
 'nuts': 'Nuts',
 'production_env_produce': 'FPE-Produce',
 'soil_northeast': 'Soil-NE',
 'soil_appalachian': 'Soil-APPAL', # singleton community
 'wild_animal': 'Wild Animals', # singleton community
 'dairy': 'Dairy',
 'food_production_env_pacific': 'FPE-PAC',
 'food_production_env_mountain': 'FPE-MTN',
 'food_production_env_southern_plains': 'FPE-SPLNS',
 'food_production_env_corn_belt': 'FPE-CB',
 'food_production_env_northeast': 'FPE-NE',
 'food_production_env_appalachian': 'FPE-APPAL',
 'food_production_env_na': 'FPE-N/A',
 'food_production_env_southeast': 'FPE-SE',
 'food_production_env_lake_states': 'FPE-LS',
 'food_production_env_northern_plains': 'FPE-NPLNS',
 'food_production_env_delta_states': 'FPE-DELTA', # singleton community
 'root_underground': 'Root Underground', # singleton community
 'water_northeast': 'WTR-NE',
 'water_southeast': 'WTR-SE',
 'water_pacific': 'WTR-PAC',
 'water_appalachian': 'WTR-APPAL',
 'sewage_hi': 'Sewage-HI',
 'water_northern_plains': 'WTR-NPLNS',
 'water_southern_plains': 'WTR-SPLNS',
 'water_na': 'WTR-N/A',
 'forest': 'Forest',
 'multi_ingredient_other': 'MIF-Other',
 'multi_ingredient_poultry': 'MIF-Poultry',
 'rodents': 'Rodents',
 'rte': 'RTE',
 'poultry': 'Poultry',
 'bovine': 'Bovine',
 'swine': 'Swine',
 'human': 'Humans',
 }

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


def community_importance(
    G, communities, 
    density_thr=None, external_thr=None, 
    weight='weight', 
    density_mode='global_avg',
    distance_mode='inv',
    return_metagraph=False
):
    from collections import defaultdict
    import numpy as np
    import pandas as pd
    import networkx as nx
    comm_nodes = defaultdict(list)
    for n, c in communities.items():
        comm_nodes[c].append(n)
    comm_ids = list(comm_nodes.keys())
    all_w = [d.get(weight, 1.0) for _,_,d in G.edges(data=True)]
    global_max_w = max(all_w) if all_w else 1.0
    global_q95_w = np.quantile(all_w, 0.95) if all_w else 1.0
    records = []
    for comm, nodes in comm_nodes.items():
        subG = G.subgraph(nodes)
        n = len(nodes)
        internal_w = subG.size(weight=weight)
        if n > 1:
            max_possible_edges = n * (n - 1) / 2
            if density_mode == 'global_avg':
                dens = internal_w / max_possible_edges
            elif density_mode == 'global_max':
                ref = global_q95_w if global_q95_w > 0 else global_max_w
                dens = internal_w / (max_possible_edges * ref)
            elif density_mode == 'per_node_strength':
                dens = (2.0 * internal_w) / n
            else:
                raise ValueError("density_mode not recognized")
        else:
            dens = 0.0
        external_w = 0.0
        for u, v in nx.edge_boundary(G, nodes):
            external_w += G[u][v].get(weight, 1.0)
        denom = external_w + 2.0 * internal_w
        external_ratio = external_w / denom if denom > 0 else 0.0
        records.append(dict(
            community=comm, size=n,
            internal_weight=internal_w,
            internal_weighted_density=dens,
            external_weight=external_w,
            external_ratio=external_ratio
        ))
    df = pd.DataFrame(records)
    H = nx.Graph()
    H.add_nodes_from(comm_ids)
    for u, v, d in G.edges(data=True):
        cu, cv = communities[u], communities[v]
        if cu == cv:
            continue
        w = d.get(weight, 1.0)
        if H.has_edge(cu, cv):
            H[cu][cv]['weight'] += w
        else:
            H.add_edge(cu, cv, weight=w)
    for u, v, d in H.edges(data=True):
        w = d.get('weight', 0.0)
        if distance_mode == 'inv':
            d['inv_weight'] = (1.0 / w) if w > 0 else np.inf
        elif distance_mode == 'inv_log1p':
            d['inv_weight'] = (1.0 / np.log1p(w)) if w > 0 else np.inf
        else:
            raise ValueError("distance_mode not recognized")
    comm_bt = nx.betweenness_centrality(H, weight='inv_weight', normalized=True) if H.number_of_nodes() > 1 else {c:0 for c in comm_ids}
    df['community_betweenness'] = df['community'].map(lambda c: comm_bt.get(c, 0.0)).fillna(0.0)
    def norm(col):
        mx = df[col].max()
        return df[col] / mx if mx and mx > 0 else 0.0
    df['size_norm'] = norm('size')
    df['density_norm'] = norm('internal_weighted_density')
    df['external_weight_norm'] = norm('external_weight')
    df['community_betweenness_norm'] = norm('community_betweenness')
    df['importance_score'] = (
        0.4 * df['size_norm'] +
        0.2 * df['density_norm'] +
        0.2 * df['external_weight_norm'] +
        0.2 * df['community_betweenness_norm']
    )
    if density_thr is None:
        density_thr = float(df['internal_weighted_density'].median())
    if external_thr is None:
        external_thr = float(df['external_ratio'].median())
    def classify(density, ext_ratio):
        if density >= density_thr and ext_ratio < external_thr:
            return "Core"
        elif density >= density_thr and ext_ratio >= external_thr:
            return "Hub"
        elif density < density_thr and ext_ratio >= external_thr:
            return "Bridge"
        else:
            return "Peripheral"
    df['role'] = [classify(d, e) for d, e in zip(df['internal_weighted_density'], df['external_ratio'])]
    df = df.sort_values('importance_score', ascending=False).reset_index(drop=True)
    df = df[['community','size','internal_weight','internal_weighted_density',
             'external_weight','external_ratio','community_betweenness',
             'importance_score','role']]
    df = df.rename(columns={'external_ratio':'external_ratio(conductance)'})
    return (df, H) if return_metagraph else df


def calculate_correlations(df, columns, correlations=['pearson', 'spearman', 'kendall']):
    """
    Calculate correlations between specified columns in a DataFrame and return results as a DataFrame.
    """
    from scipy.stats import pearsonr, spearmanr, kendalltau
    
    df_subset = df[columns].dropna()
    pairwise_results = []
    
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col1, col2 = columns[i], columns[j]
            row = {'Columns': f'{col1} vs {col2}'}
            
            if 'pearson' in correlations:
                pearson_r, pearson_p = pearsonr(df_subset[col1], df_subset[col2])
                row.update({'Pearson': pearson_r, 'Pearson_p': pearson_p})
            
            if 'spearman' in correlations:
                spearman_r, spearman_p = spearmanr(df_subset[col1], df_subset[col2])
                row.update({'Spearman': spearman_r, 'Spearman_p': spearman_p})
            
            if 'kendall' in correlations:
                kendall_r, kendall_p = kendalltau(df_subset[col1], df_subset[col2])
                row.update({'Kendall': kendall_r, 'Kendall_p': kendall_p})
                
            pairwise_results.append(row)
    
    return pd.DataFrame(pairwise_results)


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



def remove_disconnected_nodes(G, communities=None, weight="weight"):
    """
    Remove disconnected nodes (isolates: degree == 0). Keep singleton *communities*
    if the node is connected (i.e., do NOT remove nodes just because their community size is 1).

    Returns:
      (result_dict, G_filtered, disconnected_nodes_removed)

    Notes:
      - If `communities` is provided, modularity and per-community stats are computed
        on the filtered graph using community labels restricted to remaining nodes.
      - We do NOT return the filtered community mapping; instead we report which nodes
        were removed and (optionally) their original community IDs.
    """
    from collections import Counter, defaultdict
    import networkx as nx
    from networkx.algorithms.community.quality import modularity

    # --- identify disconnected nodes (isolates) ---
    disconnected_nodes = list(nx.isolates(G))  # degree == 0

    # --- filter graph ---
    G_filtered = G.copy()
    if disconnected_nodes:
        G_filtered.remove_nodes_from(disconnected_nodes)

    # --- prepare community mappings (for stats only) ---
    communities_in_G = {}
    communities_for_stats = {}
    removed_node_communities = {}

    if communities is not None:
        # only keep labels for nodes actually in G
        communities_in_G = {n: c for n, c in communities.items() if n in G}

        # record which communities the removed nodes belonged to (if labeled)
        removed_node_communities = {n: communities_in_G.get(n) for n in disconnected_nodes if n in communities_in_G}

        # labels for remaining nodes (used for modularity + community stats)
        communities_for_stats = {n: c for n, c in communities_in_G.items() if n not in disconnected_nodes}

        # relabel IDs to 0..K-1 for stable reporting (optional, used only internally)
        if communities_for_stats:
            unique_comms = sorted(set(communities_for_stats.values()), key=str)
            relabel_map = {old: new for new, old in enumerate(unique_comms)}
            communities_for_stats = {n: relabel_map[c] for n, c in communities_for_stats.items()}

    def dict_to_partition(comm_dict):
        comm_map = defaultdict(set)
        for n, c in comm_dict.items():
            comm_map[c].add(n)
        return list(comm_map.values())

    # --- modularity (only if communities provided) ---
    if communities_in_G:
        partition_all = dict_to_partition(communities_in_G)
        modularity_all = modularity(G, partition_all, weight=weight) if partition_all else 0.0
    else:
        modularity_all = 0.0

    if communities_for_stats and G_filtered.number_of_nodes() > 0:
        partition_filtered = dict_to_partition(communities_for_stats)
        modularity_after = modularity(G_filtered, partition_filtered, weight=weight) if partition_filtered else 0.0
    else:
        modularity_after = 0.0

    # --- per-community stats on the filtered graph ---
    records = []
    if communities_for_stats:
        comm_nodes = defaultdict(list)
        for n, c in communities_for_stats.items():
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
                "community": comm,
                "size": n_nodes,
                "internal_weighted_density": internal_density,
                "external_ratio(conductance)": conductance,
                "low_density_flag": internal_density < 0.3,  # matches your earlier style
            })

        records.sort(key=lambda r: r["size"], reverse=True)

        comm_sizes_after = Counter(communities_for_stats.values())
        num_communities_after = len(comm_sizes_after)
        num_singletons_after = sum(1 for s in comm_sizes_after.values() if s == 1)
        num_large_after = sum(1 for s in comm_sizes_after.values() if s > 1)
    else:
        num_communities_after = 0
        num_singletons_after = 0
        num_large_after = 0

    result = {
        "num_nodes": G_filtered.number_of_nodes(),
        "num_edges": G_filtered.number_of_edges(),
        "num_nodes_before": G.number_of_nodes(),
        "num_edges_before": G.number_of_edges(),

        # what was filtered out
        "num_disconnected_removed": len(disconnected_nodes),
        "disconnected_nodes_removed": disconnected_nodes,

        # optional: what communities those removed nodes belonged to (if `communities` provided)
        "disconnected_nodes_removed_communities": removed_node_communities,
        "disconnected_nodes_removed_community_counts": dict(Counter(removed_node_communities.values())) if removed_node_communities else {},

        # community stats AFTER removing disconnected nodes
        "num_communities": num_communities_after,
        "num_singletons": num_singletons_after,
        "num_large_communities": num_large_after,

        # modularity before/after isolate removal (if `communities` provided)
        "modularity_all": modularity_all,
        "modularity_after_removing_disconnected": modularity_after,

        "communities": records,
    }

    return result, G_filtered, disconnected_nodes



def renumber_communities(partition):
    """
    Renumber communities based on anchor nodes (deterministic).
    """
    from typing import Dict, List, Tuple
    from collections import defaultdict

    old_to_nodes: Dict[int, List[str]] = defaultdict(list)
    for node, old_c in partition.items():
        old_to_nodes[old_c].append(node)

    anchor_specs = [
        (1, ["animal_feed"]),
        (2, ["human"]),
        (3, ["fruit_and_vegetable"]),
        (4, ["nuts"]),
        (5, ["sewage_hi"]),
        (6, ["bovine"]),
        (7, ["forest"]),
    ]

    def locate_old_comm(candidates: list):
        for name in candidates:
            if name in partition:
                return partition[name]
        return None

    old_to_new: Dict[int, int] = {}
    used_new_labels = set()

    for new_label, candidates in anchor_specs:
        old_comm = locate_old_comm(candidates)
        if old_comm is None:
            continue
        if old_comm in old_to_new:
            continue
        old_to_new[old_comm] = new_label
        used_new_labels.add(new_label)

    next_label = 8
    remaining_old = [oc for oc in old_to_nodes.keys() if oc not in old_to_new]

    def sort_key(oc):
        nodes = sorted(old_to_nodes[oc])
        size = len(nodes)
        smallest_name = nodes[0] if nodes else ""
        return (-size, smallest_name)

    remaining_old.sort(key=sort_key)
    for oc in remaining_old:
        while next_label in used_new_labels:
            next_label += 1
        old_to_new[oc] = next_label
        used_new_labels.add(next_label)
        next_label += 1

    new_partition = {node: old_to_new[old_c] for node, old_c in partition.items()}
    return new_partition, old_to_new


def plot_community_network_selected_edges(
    G,
    communities,
    df,
    *,
    layout="sfdp",
    layout_params=None,
    modularity_score=None,
    cliques=None,
    size_attribute="HC_count",
    seed=GLOBAL_SEED,
    color_scheme=None,
    figsize=(10, 8),
    dpi=600,
    selected_edges=None,
    selected_edge_color=None,
    node_edge_color="grey",
    node_edge_width=1.0,
    node_lables=None,  # None=default labels; True=use global dict; dict=use directly
):
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
    max_node_size = 2000
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

    # ---- Clique highlighting optional ----
    edge_color_map, node_color_map = {}, {}
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

    # ---- Selected edge highlighting optional ----
    selected_set = set()
    if selected_edges:
        sel_color = selected_edge_color if selected_edge_color is not None else "red"
        for u, v in selected_edges:
            if G.has_edge(u, v):
                edge_color_map[frozenset((u, v))] = sel_color
                selected_set.add(frozenset((u, v)))

    all_edges = list(G.edges())
    non_selected_edges = [e for e in all_edges if frozenset(e) not in selected_set]
    selected_edges_draw = [e for e in all_edges if frozenset(e) in selected_set]

    def edge_color(e):
        return edge_color_map.get(frozenset(e), "#adb5bd")

    def edge_width(e):
        idx = all_edges.index(e)
        return edge_widths[idx]

    non_sel_colors = [edge_color(e) for e in non_selected_edges]
    sel_colors = [edge_color(e) for e in selected_edges_draw]

    non_sel_widths = [edge_width(e) for e in non_selected_edges]
    sel_widths = [edge_width(e) for e in selected_edges_draw]

    # ---- Draw ----
    plt.figure(figsize=figsize, dpi=dpi)

    if non_selected_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=non_selected_edges,
            edge_color=non_sel_colors,
            width=non_sel_widths,
            alpha=0.9,
        )

    if selected_edges_draw:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=selected_edges_draw,
            edge_color=sel_colors,
            width=sel_widths,
            alpha=0.9,
        )

    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=1.0,
    )
    nodes.set_edgecolor(node_edge_color)
    nodes.set_linewidth(node_edge_width)

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
                alpha=1,
            )

    # ---- Labels ----
    def _two_row_label(text):
        text = str(text)
        parts = text.split(" ")
        if len(parts) == 2:
            return parts[0] + "\n" + parts[1]
        return text

    if node_lables is None:
        label_map = {node: _two_row_label(node) for node in G.nodes()}
    elif node_lables is True:
        raw_map = globals().get("node_lables")
        if not isinstance(raw_map, dict):
            raise ValueError("node_lables=True requires a global dict named node_lables.")
        label_map = {node: _two_row_label(raw_map.get(node, node)) for node in G.nodes()}
    elif isinstance(node_lables, dict):
        label_map = {node: _two_row_label(node_lables.get(node, node)) for node in G.nodes()}
    else:
        raise ValueError("node_lables must be None, True, or a dict.")

    nx.draw_networkx_labels(
        G,
        pos,
        labels=label_map,
        font_size=12,
        bbox=dict(
            facecolor="none",
            edgecolor="none",
            alpha=0.7,
            boxstyle="round,pad=0.2",
        ),
    )

    # ---- Legend ----
    legend_elements = []
    for comm_id, color in color_map.items():
        legend_elements.append(
            plt.Line2D(
                [0], [0],
                marker="o",
                color="w",
                label=f"Community {comm_id}",
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0,
                markersize=20,
            )
        )

    if legend_elements:
        leg = plt.legend(
            handles=legend_elements,
            loc="upper right",
            bbox_to_anchor=(0.94, 0.95),
            frameon=True,
            fontsize=12,
        )
        leg.get_frame().set_edgecolor("#adb5bd")
        leg.get_frame().set_linewidth(1.0)

    title_parts = []
    if modularity_score is not None:
        title_parts.append(f"Modularity: {modularity_score:.3f}")
    plt.title("\n".join(title_parts), pad=20)

    plt.axis("off")
    plt.tight_layout()
    plt.show()

# Evaluate modularity/partition significance (Inverse-normal z score)
import numpy as np
import networkx as nx
import random as _py_random
from collections import defaultdict
from statistics import NormalDist


def _communities_from_partition_dict(partition_dict):
    """Convert {node: community_id} -> list[set(nodes)]"""
    comms = defaultdict(set)
    for node, cid in partition_dict.items():
        comms[cid].add(node)
    return list(comms.values())


def _extract_edge_weights(G, weight_key="weight", default=1.0):
    """Extract weights in the current G.edges() iteration order."""
    w = []
    for u, v in G.edges():
        data = G.get_edge_data(u, v) or {}
        w.append(data.get(weight_key, default))
    return np.asarray(w, dtype=float)


def _assign_edge_weights_in_order(G, weights, weight_key="weight"):
    """Assign weights to edges in G.edges() iteration order."""
    for (u, v), w in zip(G.edges(), weights):
        G[u][v][weight_key] = float(w)
    return G


def _empirical_p_value(null_vals, observed, tail="upper", plus_one=True):
    """One-sided empirical p-value with optional +1 correction."""
    null_vals = np.asarray(null_vals, dtype=float)
    n = null_vals.size
    if n == 0:
        raise RuntimeError("Null distribution is empty.")

    if tail == "upper":
        k = int(np.sum(null_vals >= observed))
    elif tail == "lower":
        k = int(np.sum(null_vals <= observed))
    else:
        raise ValueError("tail must be 'upper' or 'lower'.")

    if plus_one:
        return (k + 1) / (n + 1)
    else:
        return k / n


def _invnorm_z_from_p(p, n=None):
    """
    Convert one-sided p to Gaussian-equivalent z: z = Phi^{-1}(1 - p).
    Clipped to avoid +/-inf.
    """
    if n is not None and n > 0:
        # natural clip based on Monte Carlo resolution
        eps = 1.0 / (n + 1)
    else:
        eps = 1e-12
    p = float(np.clip(p, eps, 1.0 - eps))
    return NormalDist().inv_cdf(1.0 - p)


def _make_weighted_degree_preserving_null(
    G,
    weights,
    weight_key="weight",
    seed=None,
    nswap_factor=5,
    max_tries_factor=1000,
    adapt_rounds=4,
):
    """
    Weighted null generator:
      - Degree-preserving: double_edge_swap on topology-only graph.
      - Weight-preserving (multiset): randomly permute original weights onto rewired edges.

    If swapping fails (dense/constrained graphs), falls back to:
      - keep topology, permute weights only.
    """
    if G.is_directed() or G.is_multigraph():
        raise TypeError("Only supports undirected simple Graphs (nx.Graph).")

    m = G.number_of_edges()
    if m == 0:
        H = G.copy()
        return H, "no_edges"

    # Topology-only copy (no attributes)
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from(G.edges())

    rng = np.random.default_rng(seed)

    target = max(1, int(nswap_factor * m))
    max_tries = max(1, int(max_tries_factor * m))

    method = "swap+weight_permute"
    swapped = False

    # Adaptive attempts: reduce target swaps if graph is too constrained
    for _ in range(adapt_rounds):
        try:
            # networkx requires nswap <= max_tries
            nswap = min(target, max_tries)
            nx.double_edge_swap(H, nswap=nswap, max_tries=max_tries)
            swapped = True
            break
        except Exception:
            target = max(1, target // 2)

    if not swapped:
        method = "weight_permute_only"
        # H currently equals original topology (since swaps never succeeded)

    # Permute weights and assign to H
    w = np.array(weights, copy=True)
    rng.shuffle(w)
    _assign_edge_weights_in_order(H, w, weight_key=weight_key)

    return H, method


def evaluate_partition_significance_modularity(
    G,
    observed_partition=None,    # dict node->community_id; if None, detect below
    n_iter=1000,
    score_type="z",             # 'z' (inverse-normal), 'p', or 'z_classic'
    plot=True,
    weight="weight",
    resolution=1.0,
    seed=None,
    nswap_factor=5,
    max_tries_factor=1000,
    plus_one=True,
    tail="upper",
    verbose=True,
    community_detection="leiden",  # 'louvain' or 'leiden'
):
    """
    Weighted modularity significance test using chosen community detection (louvain/leiden).

    - If observed_partition is None, detect communities using the chosen method.
    - Nulls: degree-preserving edge swaps on topology + weight permutation.
    """
    if G.is_directed() or G.is_multigraph():
        raise TypeError("This function supports only undirected simple Graphs (nx.Graph).")

    community_detection = (community_detection or "louvain").lower()
    if community_detection not in {"louvain", "leiden"}:
        raise ValueError("community_detection must be 'louvain' or 'leiden'")

    def _partition_to_sets(part, nodes):
        """Normalize partition to list[set], accepting mapping, flat labels, iterable-of-iterables, or (part, meta) tuple."""
        if isinstance(part, tuple) and len(part) == 2:
            part = part[0]

        # Reject scalars
        if np.isscalar(part):
            raise ValueError(f"Partition is scalar ({part}); expected mapping, labels, or iterable of iterables.")

        # Mapping
        if hasattr(part, "items"):
            m = dict(part)
            return [{n for n, c in m.items() if c == cid} for cid in set(m.values())]

        # Flat labels aligned to nodes
        if isinstance(part, (list, tuple)) or (hasattr(part, "shape") and len(getattr(part, "shape", ())) == 1):
            labels = list(part)
            if len(labels) != len(nodes):
                raise ValueError(f"Partition length {len(labels)} != number of nodes {len(nodes)}.")
            return [{n for n, c in zip(nodes, labels) if c == cid} for cid in set(labels)]

        # Iterable of iterables (communities)
        try:
            sets_out = []
            for comm in part:
                if np.isscalar(comm):
                    raise ValueError(f"Encountered scalar community label {comm}; expected iterable of nodes.")
                sets_out.append(set(comm))
            if not sets_out:
                raise ValueError("Empty partition provided.")
            return sets_out
        except Exception as exc:
            raise ValueError(f"Partition must be mapping, flat labels, or iterable of iterables: {exc}")

    # Detect if not provided
    if observed_partition is None:
        if community_detection == "louvain":
            observed_partition = detect_communities_louvain(G, seed=seed)
        else:
            observed_partition = detect_communities_leiden(G, seed=seed)

    nodes_order = list(G.nodes)
    observed_sets = _partition_to_sets(observed_partition, nodes_order)

    # Validate coverage
    covered = set().union(*observed_sets) if observed_sets else set()
    if covered != set(G.nodes):
        missing = set(G.nodes()) - covered
        extra = covered - set(G.nodes())
        raise ValueError(
            "observed_partition must assign every node in G exactly once. "
            f"missing={len(missing)}, extra={len(extra)}"
        )

    # Observed modularity
    q_obs = nx.community.modularity(G, observed_sets, weight=weight, resolution=resolution)

    # Cache original weights once
    w0 = _extract_edge_weights(G, weight_key=weight, default=1.0)

    null_q = []
    null_methods = {"swap+weight_permute": 0, "weight_permute_only": 0, "no_edges": 0}

    for i in range(n_iter):
        iter_seed = None if seed is None else (seed + i)

        H, method = _make_weighted_degree_preserving_null(
            G,
            weights=w0,
            weight_key=weight,
            seed=iter_seed,
            nswap_factor=nswap_factor,
            max_tries_factor=max_tries_factor,
        )
        null_methods[method] = null_methods.get(method, 0) + 1

        # Detect communities on null graph
        if community_detection == "louvain":
            null_part = detect_communities_louvain(H, seed=iter_seed)
        else:
            null_part = detect_communities_leiden(H, seed=iter_seed)

        part_sets = _partition_to_sets(null_part, list(H.nodes()))
        q = nx.community.modularity(H, part_sets, weight=weight, resolution=resolution)
        null_q.append(q)

    null_q = np.asarray(null_q, dtype=float)

    # Empirical p from null
    p_emp = _empirical_p_value(null_q, q_obs, tail=tail, plus_one=plus_one)

    # Score selection
    if score_type == "z":
        score = _invnorm_z_from_p(p_emp, n=(len(null_q) if plus_one else None))
        label = "Inverse-normal Z (from empirical p)"
    elif score_type == "p":
        score = p_emp
        label = "Empirical p-value"
    elif score_type == "z_classic":
        mu = float(null_q.mean())
        sd = float(null_q.std(ddof=1))
        score = (q_obs - mu) / sd if sd > 0 else np.inf
        label = "Classic Z (obs-mean)/std"
    else:
        raise ValueError("score_type must be 'z', 'p', or 'z_classic'.")

    if verbose:
        mu = float(null_q.mean())
        sd = float(null_q.std(ddof=1))
        print(f"Observed modularity Q:  {q_obs:.7f}")
        print(f"Null mean / std:        {mu:.7f} / {sd:.7f}")
        print(f"Empirical p-value:      {p_emp:.7g}")
        print(f"{label}:               {score:.7f}")
        print(f"Null method counts:     {null_methods}")

    if plot:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(8, 8), dpi=300)
        sns.histplot(null_q, bins=50, kde=True, color="lightgray", label="Null modularity")
        plt.axvline(q_obs, color="red", linestyle="--", label=f"Observed (Q={q_obs:.3f})")
        #plt.title(f"Modularity Null Distribution (weighted, {community_detection.title()})")
        plt.xlabel("Modularity")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return score, q_obs, null_q


## Node classification by clustering of centrality metrics
import numpy as np
def add_node_role(
    node_metrics_no_singletons,
    eig_col="eigenvector",
    betw_col="betweenness",
    q_high=0.75,
    role_col="node_role",
):
    df = node_metrics_no_singletons.copy()

    b_thr = np.percentile(df[betw_col].dropna().values, q_high * 100)
    e_thr = np.percentile(df[eig_col].dropna().values, q_high * 100)

    def classify_role(row):
        hb = row[betw_col] >= b_thr
        he = row[eig_col] >= e_thr

        if hb and he:
            return "Global Connector"
        elif hb and not he:
            return "Broker/Bridge"
        elif he and not hb:
            return "Core Influencer"
        else:
            return "Peripheral/Local"

    df[role_col] = df.apply(classify_role, axis=1)
    return df


### Function for alpha diversity calculation
import pandas as pd
import numpy as np
from skbio import DistanceMatrix
from skbio.tree import nj
from skbio.diversity.alpha import faith_pd, shannon, simpson, chao1

def build_nj_tree(dist_df):
    """
    Build a midpoint-rooted NJ tree from a distance matrix.

    Parameters:
        dist_df (pd.DataFrame): Allelic distance matrix (rows & columns = genome IDs).

    Returns:
        skbio.tree.TreeNode: Midpoint-rooted NJ tree.
    """
    dist_df = dist_df.copy()
    dist_df.index = dist_df.index.astype(str)
    dist_df.columns = dist_df.columns.astype(str)

    dm = DistanceMatrix(dist_df.values, ids=dist_df.index.tolist())
    tree = nj(dm).root_at_midpoint()

    return tree

def compute_full_alpha_diversity(counts_df, tree=None):
    """
    Compute multiple alpha diversity metrics for each source.

    Metrics included:
        - Faith_PD (requires tree, uses presence/absence)
        - Observed_HCs / Richness (presence/absence)
        - Shannon index (raw counts)
        - Simpson index (raw counts)
        - Chao1 estimator (raw counts)
        - Evenness (Shannon / log(Richness), raw counts)
        - Isolate_N (total isolates / counts per source)

    Parameters:
        counts_df (pd.DataFrame): Raw counts table (rows = sources, columns = feature IDs)
        tree (skbio.tree.TreeNode, optional): NJ tree for Faith_PD. If None, Faith_PD will be NaN.

    Returns:
        pd.DataFrame: Alpha diversity metrics per source.
    """
    counts_df = counts_df.copy()
    counts_df.columns = counts_df.columns.astype(str)

    # 0️⃣ Isolate counts
    isolate_counts = counts_df.sum(axis=1)

    # 1️⃣ Observed OTUs / Richness (presence/absence)
    presence_absence = counts_df.gt(0).astype(int)
    observed_hcs = presence_absence.sum(axis=1)

    # 2️⃣ Shannon, Simpson, Chao1 (raw counts)
    shannon_vals = [shannon(row.values) for idx, row in counts_df.iterrows()]
    simpson_vals = [simpson(row.values) for idx, row in counts_df.iterrows()]
    chao1_vals = [chao1(row.values) for idx, row in counts_df.iterrows()]

    # 3️⃣ Evenness (Shannon / log(Richness))
    evenness_vals = [s / np.log(r) if r > 1 else 0 for s, r in zip(shannon_vals, observed_hcs)]

    # 4️⃣ Faith_PD (presence/absence + tree)
    if tree is not None:
        tip_names = [t.name for t in tree.tips()]
        pd_values = {}
        for source, row in counts_df.iterrows():
            presence = np.array([1 if row.get(t, 0) > 0 else 0 for t in tip_names], dtype=int)
            pd_values[source] = faith_pd(presence, tip_names, tree)
        faith_pd_vals = pd.Series(pd_values)
    else:
        faith_pd_vals = pd.Series([np.nan]*counts_df.shape[0], index=counts_df.index)

    # ✅ Build result table
    alpha_df = pd.DataFrame({
        'Isolate_N': isolate_counts,
        'Faith_PD': faith_pd_vals,
        'Observed_HCs': observed_hcs,
        'Shannon': shannon_vals,
        'Simpson': simpson_vals,
        'Chao1': chao1_vals,
        'Evenness': evenness_vals
    })
    alpha_df.index.name = "curated_source_region"

    # Sort by Faith_PD if tree given
    if tree is not None:
        alpha_df = alpha_df.sort_values("Faith_PD", ascending=False)

    return alpha_df


## Build community meta-graph 
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch, ConnectionStyle

color_scheme = {
    "Community 1": "#dd6670",
    "Community 2": "#edae93",
    "Community 3": "#93b071",
    "Community 4": "#ede2cc",
    "Community 5": "#598f91",
    "Community 6": "#88a2c4",
    "Community 7": "#14698D",
}

def build_metagraph(G, partition, weight="weight"):
    missing = [n for n in G.nodes if n not in partition]
    if missing:
        raise ValueError(
            f"partition is missing {len(missing)} nodes from G (e.g., {missing[:5]})."
        )

    comm_ids = sorted(set(partition.values()))
    comm_size = {c: 0 for c in comm_ids}
    for n in G.nodes:
        comm_size[partition[n]] += 1

    H = nx.Graph()
    H.add_nodes_from(comm_ids)

    for u, v, d in G.edges(data=True):
        cu, cv = partition[u], partition[v]
        if cu == cv:
            continue

        a, b = (cu, cv) if cu <= cv else (cv, cu)
        w = float(d.get(weight, 1.0))

        if H.has_edge(a, b):
            H[a][b]["weight"] += w
            H[a][b]["n_edges"] += 1
        else:
            H.add_edge(a, b, weight=w, n_edges=1)

    return H, comm_size


# Edge characterizaton and classification
## All node pair characterization by cosine similarity and node richness
def build_pairwise_cosine_df_tidy(
    sim_matrix_single_cosine,
    df_all_training_testing,
    node_col="curated_source_region",
    hc_col="HC50",
    pair_sep="-",
    drop_self_pairs=True,
    upper_triangle_only=True,
    node_name_override=None,   # dict: {old_name: new_name}
):
    # --- 1) Node labels in same order as similarity matrix ---
    if isinstance(sim_matrix_single_cosine, pd.DataFrame):
        sim_df = sim_matrix_single_cosine
        nodes = sim_df.index.to_list()
        if sim_df.shape[0] != sim_df.shape[1]:
            raise ValueError("Similarity matrix DataFrame must be square.")
        if list(sim_df.columns) != nodes:
            raise ValueError("Similarity matrix DataFrame must have matching index and columns order.")
        sim_values = sim_df.values
    else:
        sim_values = np.asarray(sim_matrix_single_cosine)
        if sim_values.ndim != 2 or sim_values.shape[0] != sim_values.shape[1]:
            raise ValueError("sim_matrix_single_cosine must be a square 2D array or a square DataFrame.")
        nodes = sorted(df_all_training_testing[node_col].dropna().unique().tolist())
        if len(nodes) != sim_values.shape[0]:
            raise ValueError(
                f"Got {sim_values.shape[0]}x{sim_values.shape[0]} matrix but inferred {len(nodes)} nodes "
                f"from df_all_training_testing[{node_col!r}]. If the matrix order differs, pass a labeled DataFrame."
            )

    n = len(nodes)

    # --- 2) Node-level summaries ---
    node_stats = (
        df_all_training_testing
        .dropna(subset=[node_col, hc_col])
        .groupby(node_col)
        .agg(
            isolate_count=(hc_col, "size"),
            hc50_richness=(hc_col, pd.Series.nunique),
        )
        .reindex(nodes)
    )
    node_stats["isolate_count"] = node_stats["isolate_count"].fillna(0).astype(int)
    node_stats["hc50_richness"] = node_stats["hc50_richness"].fillna(0).astype(int)

    # --- 2b) Node -> set of HC50 for binary Jaccard ---
    node_to_hcset = (
        df_all_training_testing
        .dropna(subset=[node_col, hc_col])
        .groupby(node_col)[hc_col]
        .apply(lambda s: set(s.tolist()))
        .reindex(nodes)
    )
    node_to_hcset = node_to_hcset.apply(lambda s: s if isinstance(s, set) else set())

    # --- helper for renaming ---
    def _rename(n):
        if isinstance(node_name_override, dict):
            return node_name_override.get(n, n)
        return n

    # --- 3) Build distinct pairs ---
    rows = []
    for i in range(n):
        j_start = i + 1 if upper_triangle_only else 0
        for j in range(j_start, n):
            if drop_self_pairs and i == j:
                continue

            node1, node2 = nodes[i], nodes[j]
            A = node_to_hcset.loc[node1]
            B = node_to_hcset.loc[node2]
            union = A | B
            jaccard = len(A & B) / len(union) if len(union) > 0 else 0.0

            r1 = int(node_stats.loc[node1, "hc50_richness"])
            r2 = int(node_stats.loc[node2, "hc50_richness"])
            rmax = max(r1, r2)
            richness_ratio = (min(r1, r2) / rmax) if rmax > 0 else np.nan

            node1_r = _rename(node1)
            node2_r = _rename(node2)

            rows.append({
                "node_pair": f"{node1_r}{pair_sep}{node2_r}",
                "node1": node1_r,
                "node2": node2_r,
                "cosine_similarity": float(sim_values[i, j]),
                "binary_jaccard": float(jaccard),
                "isolates_node1": int(node_stats.loc[node1, "isolate_count"]),
                "isolates_node2": int(node_stats.loc[node2, "isolate_count"]),
                "richness_node1": r1,
                "richness_node2": r2,
                "richness_ratio": float(richness_ratio),
            })

    out = pd.DataFrame(rows).sort_values("cosine_similarity", ascending=False).reset_index(drop=True)
    return out

### Edge direction inference
## Shared source test: vector projection method
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# Core helpers
# ============================================================

def build_node_hc50_matrix(
    df: pd.DataFrame,
    node_col: str = "curated_source_region",
    hc_col: str = "HC50",
) -> pd.DataFrame:
    """
    Build a node x HC50 count matrix from isolate-level data.
    Rows are nodes, columns are HC50 terms, values are isolate counts.
    """
    return (
        df.groupby([node_col, hc_col])
        .size()
        .unstack(fill_value=0)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )


def compute_node_richness(mat: pd.DataFrame) -> pd.Series:
    """
    HC50 richness per node = number of distinct nonzero HC50 terms.
    """
    return (mat > 0).sum(axis=1).astype(int)


def compute_node_richness_rank(node_richness: pd.Series) -> pd.Series:
    """
    Rank nodes by richness (descending). Rank 1 is richest.
    Ties use minimum rank.
    """
    return node_richness.rank(method="min", ascending=False).astype(int)


def safe_cosine(x, y, zero_value=np.nan) -> float:
    """
    Cosine similarity with guard for zero vectors.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)
    if nx == 0 or ny == 0:
        return zero_value

    return float(cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))[0, 0])


def residual_against_source(x, s):
    """
    Remove projection of x onto s and return residual vector.
    """
    x = np.asarray(x, dtype=float)
    s = np.asarray(s, dtype=float)

    ss = np.dot(s, s)
    if ss == 0:
        raise ValueError("Source vector is all zeros; cannot project against it.")

    coef = np.dot(x, s) / ss
    return x - coef * s


def required_delta(original_cosine, max_residual=0.30, base_min_delta=0.15):
    """
    Conservative required drop:
      delta >= max(base_min_delta, original_cosine - max_residual)
    """
    return max(base_min_delta, original_cosine - max_residual)


def hc50_count_from_vector(x):
    """
    Number of nonzero HC50 terms in vector.
    """
    x = np.asarray(x)
    return int(np.sum(x > 0))


def validate_triangle(triangle):
    """
    Ensure triangle has exactly 3 distinct nodes.
    """
    triangle = tuple(triangle)
    if len(triangle) != 3 or len(set(triangle)) != 3:
        raise ValueError(f"Triangle must contain 3 distinct nodes: {triangle}")
    return triangle


def canonical_triangle(triangle):
    """
    Canonical triangle representation for deterministic seeding.
    """
    triangle = validate_triangle(triangle)
    return tuple(sorted(map(str, triangle)))


def triangle_bootstrap_seed(base_seed: int, triangle) -> int:
    """
    Deterministic seed from base_seed + triangle identity.
    This makes single-triangle and batch runs consistent.
    """
    tri_key = "|".join(canonical_triangle(triangle))
    msg = f"{int(base_seed)}::{tri_key}".encode("utf-8")
    digest = hashlib.blake2b(msg, digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**32 - 1)


def validate_source_richness_top_n(source_richness_top_n):
    """
    source_richness_top_n must be None or integer in [0, 60].
    """
    if source_richness_top_n is None:
        return None
    if not isinstance(source_richness_top_n, (int, np.integer)):
        raise ValueError("source_richness_top_n must be None or an integer in [0, 60].")
    source_richness_top_n = int(source_richness_top_n)
    if not (0 <= source_richness_top_n <= 60):
        raise ValueError("source_richness_top_n must be in [0, 60].")
    return source_richness_top_n


def validate_bootstrap_params(
    n_boot,
    bootstrap_frac_range,
    bootstrap_min_total_per_node,
    bootstrap_stability_threshold,
):
    """
    Validate bootstrap configuration.
    """
    n_boot = int(n_boot)
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1.")

    if (
        not isinstance(bootstrap_frac_range, (tuple, list))
        or len(bootstrap_frac_range) != 2
    ):
        raise ValueError("bootstrap_frac_range must be a length-2 tuple/list, e.g. (0.7, 1.2).")

    frac_min, frac_max = float(bootstrap_frac_range[0]), float(bootstrap_frac_range[1])
    if frac_min <= 0 or frac_max <= 0 or frac_min > frac_max:
        raise ValueError("bootstrap_frac_range must satisfy 0 < frac_min <= frac_max.")

    bootstrap_min_total_per_node = int(bootstrap_min_total_per_node)
    if bootstrap_min_total_per_node < 1:
        raise ValueError("bootstrap_min_total_per_node must be >= 1.")

    bootstrap_stability_threshold = float(bootstrap_stability_threshold)
    if not (0 <= bootstrap_stability_threshold <= 1):
        raise ValueError("bootstrap_stability_threshold must be between 0 and 1.")

    return n_boot, frac_min, frac_max, bootstrap_min_total_per_node, bootstrap_stability_threshold


# ============================================================
# Candidate evaluation from matrix
# ============================================================

def evaluate_source_pair_from_mat(
    mat: pd.DataFrame,
    source_node,
    node_a,
    node_b,
    max_residual: float = 0.30,
    base_min_delta: float = 0.15,
    min_fraction: float = 0.50,
    zero_value=np.nan,
    node_richness: pd.Series = None,
    node_richness_rank: pd.Series = None,
    source_richness_top_n=None,
):
    """
    Evaluate one candidate source for one focal pair using a prebuilt matrix.
    """
    source_richness_top_n = validate_source_richness_top_n(source_richness_top_n)

    required = [source_node, node_a, node_b]
    missing = [x for x in required if x not in mat.index]
    if missing:
        raise ValueError(f"Missing nodes in matrix: {missing}")

    if len({source_node, node_a, node_b}) < 3:
        raise ValueError("source_node, node_a, and node_b must be three distinct nodes.")

    if node_richness is None:
        node_richness = compute_node_richness(mat)
    if node_richness_rank is None:
        node_richness_rank = compute_node_richness_rank(node_richness)

    source_vec = mat.loc[source_node].to_numpy(dtype=float)
    a_vec = mat.loc[node_a].to_numpy(dtype=float)
    b_vec = mat.loc[node_b].to_numpy(dtype=float)

    a_resid = residual_against_source(a_vec, source_vec)
    b_resid = residual_against_source(b_vec, source_vec)

    original_cos = safe_cosine(a_vec, b_vec, zero_value=zero_value)
    residual_cos = safe_cosine(a_resid, b_resid, zero_value=zero_value)

    if np.isnan(original_cos) or np.isnan(residual_cos):
        raise ValueError(
            "Observed cosine is undefined because one original or residual vector is zero."
        )

    delta_cos = original_cos - residual_cos
    delta_fraction = delta_cos / original_cos if original_cos > 0 else np.nan
    req_delta = required_delta(
        original_cosine=original_cos,
        max_residual=max_residual,
        base_min_delta=base_min_delta,
    )

    source_hc50_count = hc50_count_from_vector(source_vec)
    node_a_hc50_count = hc50_count_from_vector(a_vec)
    node_b_hc50_count = hc50_count_from_vector(b_vec)

    passes_size_filter = source_hc50_count >= max(node_a_hc50_count, node_b_hc50_count)

    source_rank = int(node_richness_rank.loc[source_node])
    passes_source_node_test = (
        True if source_richness_top_n is None else source_rank <= source_richness_top_n
    )

    return {
        "source_node": source_node,
        "node_a": node_a,
        "node_b": node_b,
        "original_cosine": original_cos,
        "residual_cosine": residual_cos,
        "delta_cosine": delta_cos,
        "delta_fraction": delta_fraction,
        "required_delta": req_delta,
        "passes_residual": residual_cos <= max_residual,
        "passes_delta": delta_cos >= req_delta,
        "passes_fraction": delta_fraction >= min_fraction if pd.notna(delta_fraction) else False,
        "source_hc50_count": source_hc50_count,
        "node_a_hc50_count": node_a_hc50_count,
        "node_b_hc50_count": node_b_hc50_count,
        "passes_size_filter": passes_size_filter,
        "source_node_richness": int(node_richness.loc[source_node]),
        "node_a_richness": int(node_richness.loc[node_a]),
        "node_b_richness": int(node_richness.loc[node_b]),
        "source_richness_rank": source_rank,
        "passes_source_node_test": passes_source_node_test,
    }


def evaluate_triangle_from_matrix(
    mat: pd.DataFrame,
    triangle,
    max_residual: float = 0.30,
    base_min_delta: float = 0.15,
    min_fraction: float = 0.50,
    min_gap: float = 0.10,
    require_size_filter: bool = False,
    source_richness_top_n=None,
    zero_value=np.nan,
    node_richness: pd.Series = None,
    node_richness_rank: pd.Series = None,
):
    """
    Evaluate all 3 candidate sources for one triangle using a prebuilt matrix.
    """
    triangle = validate_triangle(triangle)
    source_richness_top_n = validate_source_richness_top_n(source_richness_top_n)

    missing_nodes = [n for n in triangle if n not in mat.index]
    if missing_nodes:
        raise ValueError(f"Missing triangle nodes in matrix: {missing_nodes}")

    if node_richness is None:
        node_richness = compute_node_richness(mat)
    if node_richness_rank is None:
        node_richness_rank = compute_node_richness_rank(node_richness)

    rows = []
    for source_node in triangle:
        node_a, node_b = [n for n in triangle if n != source_node]
        row = evaluate_source_pair_from_mat(
            mat=mat,
            source_node=source_node,
            node_a=node_a,
            node_b=node_b,
            max_residual=max_residual,
            base_min_delta=base_min_delta,
            min_fraction=min_fraction,
            zero_value=zero_value,
            node_richness=node_richness,
            node_richness_rank=node_richness_rank,
            source_richness_top_n=source_richness_top_n,
        )
        row["triangle"] = triangle
        rows.append(row)

    candidate_results_df = (
        pd.DataFrame(rows)
        .sort_values("delta_cosine", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    candidate_results_df["rank_in_triangle"] = np.arange(1, len(candidate_results_df) + 1)

    best_delta = candidate_results_df.loc[0, "delta_cosine"]
    second_delta = candidate_results_df.loc[1, "delta_cosine"]
    gap_to_second = (
        best_delta - second_delta
        if pd.notna(best_delta) and pd.notna(second_delta)
        else np.nan
    )

    candidate_results_df["gap_to_second"] = np.nan
    candidate_results_df.loc[candidate_results_df["rank_in_triangle"] == 1, "gap_to_second"] = gap_to_second

    candidate_results_df["passes_gap"] = False
    if pd.notna(gap_to_second):
        candidate_results_df.loc[candidate_results_df["rank_in_triangle"] == 1, "passes_gap"] = gap_to_second >= min_gap

    structural_support = (
        (candidate_results_df["rank_in_triangle"] == 1)
        & candidate_results_df["passes_residual"]
        & candidate_results_df["passes_delta"]
        & candidate_results_df["passes_fraction"]
        & candidate_results_df["passes_gap"]
    )

    if require_size_filter:
        structural_support = structural_support & candidate_results_df["passes_size_filter"]

    candidate_results_df["source_supported"] = (
        structural_support & candidate_results_df["passes_source_node_test"]
    )

    best_source_row = candidate_results_df.iloc[0].copy()
    supported_source = (
        best_source_row["source_node"] if bool(best_source_row["source_supported"]) else None
    )

    return candidate_results_df, best_source_row, supported_source


# ============================================================
# Fast matrix bootstrap
# ============================================================

def bootstrap_node_vector_variable_total(
    counts_vec: np.ndarray,
    rng: np.random.Generator,
    frac_min: float,
    frac_max: float,
    bootstrap_min_total_per_node: int,
) -> np.ndarray:
    """
    Bootstrap a single node's HC50 count vector without raw-row resampling.
    """
    counts_vec = np.asarray(counts_vec, dtype=float)
    total = counts_vec.sum()

    if total <= 0:
        return np.zeros_like(counts_vec, dtype=int)

    probs = counts_vec / total
    u = float(rng.uniform(frac_min, frac_max))
    n_boot = max(bootstrap_min_total_per_node, int(round(total * u)))
    return rng.multinomial(n=n_boot, pvals=probs)


def bootstrap_triangle_matrix_variable_n(
    mat: pd.DataFrame,
    triangle,
    rng: np.random.Generator,
    frac_min: float,
    frac_max: float,
    bootstrap_min_total_per_node: int,
) -> pd.DataFrame:
    """
    Fast bootstrap directly from the precomputed node x HC50 count matrix.
    """
    triangle = validate_triangle(triangle)

    missing_nodes = [n for n in triangle if n not in mat.index]
    if missing_nodes:
        raise ValueError(f"Missing triangle nodes in matrix: {missing_nodes}")

    boot_rows = []
    for node in triangle:
        counts_vec = mat.loc[node].to_numpy(dtype=float)
        boot_vec = bootstrap_node_vector_variable_total(
            counts_vec=counts_vec,
            rng=rng,
            frac_min=frac_min,
            frac_max=frac_max,
            bootstrap_min_total_per_node=bootstrap_min_total_per_node,
        )
        boot_rows.append(boot_vec)

    return pd.DataFrame(
        data=np.vstack(boot_rows),
        index=list(triangle),
        columns=mat.columns,
    )


def bootstrap_triangle_stability(
    mat: pd.DataFrame,
    triangle,
    max_residual: float,
    base_min_delta: float,
    min_fraction: float,
    min_gap: float,
    require_size_filter: bool,
    source_richness_top_n,
    n_boot: int,
    random_state: int,
    bootstrap_frac_range=(0.7, 1.2),
    bootstrap_min_total_per_node: int = 1,
    bootstrap_stability_threshold: float = 0.80,
    zero_value=np.nan,
):
    """
    Fast bootstrap stability analysis for one triangle using the precomputed count matrix.
    """
    triangle = validate_triangle(triangle)
    source_richness_top_n = validate_source_richness_top_n(source_richness_top_n)
    n_boot, frac_min, frac_max, bootstrap_min_total_per_node, bootstrap_stability_threshold = (
        validate_bootstrap_params(
            n_boot=n_boot,
            bootstrap_frac_range=bootstrap_frac_range,
            bootstrap_min_total_per_node=bootstrap_min_total_per_node,
            bootstrap_stability_threshold=bootstrap_stability_threshold,
        )
    )

    missing_nodes = [n for n in triangle if n not in mat.index]
    if missing_nodes:
        raise ValueError(f"Cannot bootstrap triangle; missing nodes in matrix: {missing_nodes}")

    rng = np.random.default_rng(random_state)

    selected_counts = {s: 0 for s in triangle}
    supported_counts = {s: 0 for s in triangle}
    delta_values = {s: [] for s in triangle}
    long_rows = []
    none_supported_count = 0
    n_valid = 0
    n_invalid = 0

    for b in range(n_boot):
        try:
            boot_mat = bootstrap_triangle_matrix_variable_n(
                mat=mat,
                triangle=triangle,
                rng=rng,
                frac_min=frac_min,
                frac_max=frac_max,
                bootstrap_min_total_per_node=bootstrap_min_total_per_node,
            )
            boot_node_richness = compute_node_richness(boot_mat)
            boot_node_richness_rank = compute_node_richness_rank(boot_node_richness)

            boot_candidate_df, boot_best_row, boot_supported_source = evaluate_triangle_from_matrix(
                mat=boot_mat,
                triangle=triangle,
                max_residual=max_residual,
                base_min_delta=base_min_delta,
                min_fraction=min_fraction,
                min_gap=min_gap,
                require_size_filter=require_size_filter,
                source_richness_top_n=source_richness_top_n,
                zero_value=zero_value,
                node_richness=boot_node_richness,
                node_richness_rank=boot_node_richness_rank,
            )
        except Exception:
            n_invalid += 1
            continue

        n_valid += 1
        selected_counts[boot_best_row["source_node"]] += 1

        if boot_supported_source is None:
            none_supported_count += 1
        else:
            supported_counts[boot_supported_source] += 1

        boot_candidate_df = boot_candidate_df.copy()
        boot_candidate_df["bootstrap_rep"] = b
        boot_candidate_df["triangle"] = triangle
        long_rows.append(boot_candidate_df)

        for s in triangle:
            d = boot_candidate_df.loc[boot_candidate_df["source_node"] == s, "delta_cosine"].iloc[0]
            if pd.notna(d):
                delta_values[s].append(float(d))

    if n_valid == 0:
        raise ValueError("All bootstrap replicates were invalid.")

    bootstrap_long_df = pd.concat(long_rows, ignore_index=True)

    candidate_summaries = []
    for s in triangle:
        g = bootstrap_long_df.loc[bootstrap_long_df["source_node"] == s]

        deltas = delta_values[s]
        if len(deltas) > 0:
            delta_mean = float(np.mean(deltas))
            ci_low = float(np.percentile(deltas, 2.5))
            ci_high = float(np.percentile(deltas, 97.5))
        else:
            delta_mean, ci_low, ci_high = np.nan, np.nan, np.nan

        candidate_summaries.append({
            "triangle": triangle,
            "source_node": s,
            "bootstrap_selected_rate": selected_counts[s] / n_valid,
            "bootstrap_supported_rate": supported_counts[s] / n_valid,
            "bootstrap_delta_mean": delta_mean,
            "bootstrap_delta_ci_low": ci_low,
            "bootstrap_delta_ci_high": ci_high,
            "bootstrap_pass_residual_rate": g["passes_residual"].mean(),
            "bootstrap_pass_delta_rate": g["passes_delta"].mean(),
            "bootstrap_pass_fraction_rate": g["passes_fraction"].mean(),
            "bootstrap_pass_gap_rate": g["passes_gap"].mean(),
            "bootstrap_pass_source_node_rate": g["passes_source_node_test"].mean(),
            "bootstrap_pass_size_filter_rate": g["passes_size_filter"].mean(),
            "bootstrap_stable_selected": (selected_counts[s] / n_valid) >= bootstrap_stability_threshold,
            "bootstrap_stable_supported": (supported_counts[s] / n_valid) >= bootstrap_stability_threshold,
            "n_boot_valid": n_valid,
            "n_boot_invalid": n_invalid,
        })

    bootstrap_candidate_summary_df = pd.DataFrame(candidate_summaries)

    bootstrap_none_supported_rate = none_supported_count / n_valid

    # mode-related fields intentionally removed
    bootstrap_triangle_summary_df = pd.DataFrame([{
        "triangle": triangle,
        "run_bootstrap": True,
        "n_boot_requested": n_boot,
        "n_boot_valid": n_valid,
        "n_boot_invalid": n_invalid,
        "bootstrap_frac_min": frac_min,
        "bootstrap_frac_max": frac_max,
        "bootstrap_min_total_per_node": bootstrap_min_total_per_node,
        "bootstrap_stability_threshold": bootstrap_stability_threshold,
        "bootstrap_none_supported_rate": bootstrap_none_supported_rate,
    }])

    return {
        "bootstrap_candidate_summary_df": bootstrap_candidate_summary_df,
        "bootstrap_triangle_summary_df": bootstrap_triangle_summary_df,
        "bootstrap_long_df": bootstrap_long_df,
    }


# ============================================================
# User-facing triangle evaluation
# ============================================================

def evaluate_triangle_sources(
    df: pd.DataFrame,
    triangle,
    node_col: str = "curated_source_region",
    hc_col: str = "HC50",
    run_bootstrap: bool = False,
    n_boot: int = 1000,
    random_state: int = 186,
    bootstrap_frac_range=(0.7, 1.2),
    bootstrap_min_total_per_node: int = 1,
    bootstrap_stability_threshold: float = 0.80,
    max_residual: float = 0.30,
    base_min_delta: float = 0.15,
    min_fraction: float = 0.50,
    min_gap: float = 0.10,
    require_size_filter: bool = False,
    source_richness_top_n=None,
    zero_value=np.nan,
    mat: pd.DataFrame = None,
    node_richness: pd.Series = None,
    node_richness_rank: pd.Series = None,
    print_summary: bool = False,
):
    """
    Evaluate source support in one triangle.
    Uses deterministic triangle-specific bootstrap seed so batch and single runs match.
    """
    triangle = validate_triangle(triangle)
    source_richness_top_n = validate_source_richness_top_n(source_richness_top_n)

    if mat is None:
        mat = build_node_hc50_matrix(df, node_col=node_col, hc_col=hc_col)

    if node_richness is None:
        node_richness = compute_node_richness(mat)
    if node_richness_rank is None:
        node_richness_rank = compute_node_richness_rank(node_richness)

    candidate_results_df, best_source_row, supported_source = evaluate_triangle_from_matrix(
        mat=mat,
        triangle=triangle,
        max_residual=max_residual,
        base_min_delta=base_min_delta,
        min_fraction=min_fraction,
        min_gap=min_gap,
        require_size_filter=require_size_filter,
        source_richness_top_n=source_richness_top_n,
        zero_value=zero_value,
        node_richness=node_richness,
        node_richness_rank=node_richness_rank,
    )

    bootstrap_seed_used = triangle_bootstrap_seed(random_state, triangle)

    if run_bootstrap:
        boot = bootstrap_triangle_stability(
            mat=mat,
            triangle=triangle,
            max_residual=max_residual,
            base_min_delta=base_min_delta,
            min_fraction=min_fraction,
            min_gap=min_gap,
            require_size_filter=require_size_filter,
            source_richness_top_n=source_richness_top_n,
            n_boot=n_boot,
            random_state=bootstrap_seed_used,
            bootstrap_frac_range=bootstrap_frac_range,
            bootstrap_min_total_per_node=bootstrap_min_total_per_node,
            bootstrap_stability_threshold=bootstrap_stability_threshold,
            zero_value=zero_value,
        )
        bootstrap_candidate_summary_df = boot["bootstrap_candidate_summary_df"]
        bootstrap_triangle_summary_df = boot["bootstrap_triangle_summary_df"]
        bootstrap_long_df = boot["bootstrap_long_df"]
    else:
        bootstrap_candidate_summary_df = pd.DataFrame({
            "triangle": [triangle] * 3,
            "source_node": list(triangle),
            "bootstrap_selected_rate": [np.nan] * 3,
            "bootstrap_supported_rate": [np.nan] * 3,
            "bootstrap_delta_mean": [np.nan] * 3,
            "bootstrap_delta_ci_low": [np.nan] * 3,
            "bootstrap_delta_ci_high": [np.nan] * 3,
            "bootstrap_pass_residual_rate": [np.nan] * 3,
            "bootstrap_pass_delta_rate": [np.nan] * 3,
            "bootstrap_pass_fraction_rate": [np.nan] * 3,
            "bootstrap_pass_gap_rate": [np.nan] * 3,
            "bootstrap_pass_source_node_rate": [np.nan] * 3,
            "bootstrap_pass_size_filter_rate": [np.nan] * 3,
            "bootstrap_stable_selected": [False] * 3,
            "bootstrap_stable_supported": [False] * 3,
            "n_boot_valid": [0] * 3,
            "n_boot_invalid": [0] * 3,
        })
        bootstrap_triangle_summary_df = pd.DataFrame([{
            "triangle": triangle,
            "run_bootstrap": False,
            "n_boot_requested": 0,
            "n_boot_valid": 0,
            "n_boot_invalid": 0,
            "bootstrap_frac_min": float(bootstrap_frac_range[0]),
            "bootstrap_frac_max": float(bootstrap_frac_range[1]),
            "bootstrap_min_total_per_node": int(bootstrap_min_total_per_node),
            "bootstrap_stability_threshold": float(bootstrap_stability_threshold),
            "bootstrap_none_supported_rate": np.nan,
        }])
        bootstrap_long_df = pd.DataFrame()

    candidate_results_df = candidate_results_df.merge(
        bootstrap_candidate_summary_df.drop(columns=["triangle"]),
        on="source_node",
        how="left",
    )

    best_source_row = candidate_results_df.iloc[0].copy()

    if supported_source is None:
        supported_source_bootstrap_selected_rate = np.nan
        supported_source_bootstrap_supported_rate = np.nan
        bootstrap_stable = False
    else:
        supported_row = candidate_results_df.loc[candidate_results_df["source_node"] == supported_source].iloc[0]
        supported_source_bootstrap_selected_rate = float(supported_row["bootstrap_selected_rate"])
        supported_source_bootstrap_supported_rate = float(supported_row["bootstrap_supported_rate"])
        bootstrap_stable = bool(
            run_bootstrap
            and (supported_source_bootstrap_selected_rate >= bootstrap_stability_threshold)
            and (supported_source_bootstrap_supported_rate >= bootstrap_stability_threshold)
        )

    triangle_summary_df = pd.DataFrame([{
        "triangle": triangle,
        "best_source_node": best_source_row["source_node"],
        "best_node_a": best_source_row["node_a"],
        "best_node_b": best_source_row["node_b"],
        "best_original_cosine": best_source_row["original_cosine"],
        "best_residual_cosine": best_source_row["residual_cosine"],
        "best_delta_cosine": best_source_row["delta_cosine"],
        "best_delta_fraction": best_source_row["delta_fraction"],
        "best_required_delta": best_source_row["required_delta"],
        "best_source_hc50_count": best_source_row["source_hc50_count"],
        "best_node_a_hc50_count": best_source_row["node_a_hc50_count"],
        "best_node_b_hc50_count": best_source_row["node_b_hc50_count"],
        "best_source_richness_rank": best_source_row["source_richness_rank"],
        "passes_size_filter": bool(best_source_row["passes_size_filter"]),
        "passes_source_node_test": bool(best_source_row["passes_source_node_test"]),
        "gap_to_second": best_source_row["gap_to_second"],
        "passes_residual": bool(best_source_row["passes_residual"]),
        "passes_delta": bool(best_source_row["passes_delta"]),
        "passes_fraction": bool(best_source_row["passes_fraction"]),
        "passes_gap": bool(best_source_row["passes_gap"]),
        "source_supported": bool(best_source_row["source_supported"]),
        "supported_source": supported_source,

        "run_bootstrap": bool(bootstrap_triangle_summary_df.loc[0, "run_bootstrap"]),
        "n_boot_requested": int(bootstrap_triangle_summary_df.loc[0, "n_boot_requested"]),
        "n_boot_valid": int(bootstrap_triangle_summary_df.loc[0, "n_boot_valid"]),
        "n_boot_invalid": int(bootstrap_triangle_summary_df.loc[0, "n_boot_invalid"]),
        "bootstrap_frac_min": float(bootstrap_triangle_summary_df.loc[0, "bootstrap_frac_min"]),
        "bootstrap_frac_max": float(bootstrap_triangle_summary_df.loc[0, "bootstrap_frac_max"]),
        "bootstrap_min_total_per_node": int(bootstrap_triangle_summary_df.loc[0, "bootstrap_min_total_per_node"]),
        "bootstrap_stability_threshold": float(bootstrap_triangle_summary_df.loc[0, "bootstrap_stability_threshold"]),
        "bootstrap_none_supported_rate": bootstrap_triangle_summary_df.loc[0, "bootstrap_none_supported_rate"],
        "bootstrap_seed_used": int(bootstrap_seed_used),

        "best_bootstrap_selected_rate": best_source_row["bootstrap_selected_rate"],
        "best_bootstrap_supported_rate": best_source_row["bootstrap_supported_rate"],
        "best_bootstrap_delta_mean": best_source_row["bootstrap_delta_mean"],
        "best_bootstrap_delta_ci_low": best_source_row["bootstrap_delta_ci_low"],
        "best_bootstrap_delta_ci_high": best_source_row["bootstrap_delta_ci_high"],

        "supported_source_bootstrap_selected_rate": supported_source_bootstrap_selected_rate,
        "supported_source_bootstrap_supported_rate": supported_source_bootstrap_supported_rate,
        "bootstrap_stable": bootstrap_stable,

        "max_residual": max_residual,
        "base_min_delta": base_min_delta,
        "min_fraction": min_fraction,
        "min_gap": min_gap,
        "require_size_filter": require_size_filter,
        "source_richness_top_n": source_richness_top_n,
    }])

    result = {
        "triangle": triangle,
        "candidate_results_df": candidate_results_df,
        "triangle_summary_df": triangle_summary_df,
        "bootstrap_candidate_summary_df": bootstrap_candidate_summary_df,
        "bootstrap_triangle_summary_df": bootstrap_triangle_summary_df,
        "bootstrap_long_df": bootstrap_long_df,
        "best_source_row": best_source_row,
        "supported_source": supported_source,
    }

    if print_summary:
        s = triangle_summary_df.loc[0]
        print(f"Triangle: {s['triangle']}")
        print("Rule:")
        print("  rank_in_triangle == 1")
        print(f"  residual_cosine <= {s['max_residual']:.2f}")
        print(f"  delta_cosine >= max({s['base_min_delta']:.2f}, original_cosine - {s['max_residual']:.2f})")
        print(f"  delta_fraction >= {s['min_fraction']:.2f}")
        print(f"  gap_to_second >= {s['min_gap']:.2f}")
        print(f"  require_size_filter = {bool(s['require_size_filter'])}")
        if pd.notna(s["source_richness_top_n"]):
            print(f"  source_richness_rank <= {int(s['source_richness_top_n'])}")
        if bool(s["run_bootstrap"]):
            print(
                f"  bootstrap: n_boot={int(s['n_boot_requested'])}, "
                f"valid={int(s['n_boot_valid'])}, "
                f"frac_range=({s['bootstrap_frac_min']:.2f}, {s['bootstrap_frac_max']:.2f}), "
                f"stable_if_rates>={s['bootstrap_stability_threshold']:.2f}, "
                f"seed={int(s['bootstrap_seed_used'])}"
            )
        print()

        display_cols = [
            "source_node",
            "node_a",
            "node_b",
            "source_hc50_count",
            "node_a_hc50_count",
            "node_b_hc50_count",
            "passes_size_filter",
            "source_node_richness",
            "source_richness_rank",
            "passes_source_node_test",
            "original_cosine",
            "residual_cosine",
            "delta_cosine",
            "delta_fraction",
            "required_delta",
            "rank_in_triangle",
            "gap_to_second",
            "passes_residual",
            "passes_delta",
            "passes_fraction",
            "passes_gap",
            "source_supported",
        ]

        if bool(s["run_bootstrap"]):
            display_cols += [
                "bootstrap_selected_rate",
                "bootstrap_supported_rate",
                "bootstrap_delta_mean",
                "bootstrap_delta_ci_low",
                "bootstrap_delta_ci_high",
                "bootstrap_stable_selected",
                "bootstrap_stable_supported",
            ]

        print(candidate_results_df[display_cols].to_string(index=False))
        print()

        if bool(s["run_bootstrap"]):
            print("Bootstrap stability summary:")
            print(f"  none_supported_rate: {s['bootstrap_none_supported_rate']:.4f}")
            print(f"  bootstrap_stable: {bool(s['bootstrap_stable'])}")
            print()

        if s["supported_source"] is None:
            print("Interpretation: no uniquely supported source in this triangle.")
        else:
            print(f"Interpretation: supported source for this triangle = {s['supported_source']}")

    return result


## 3-node cliques in the network
def find_3_node_cliques_fast(G):
    """
    Return all unique 3-node cliques (triangles) in an undirected graph.
    """
    if G.is_directed():
        raise ValueError("This function expects an undirected graph.")

    triangles = set()

    for u in G.nodes():
        for v in G.neighbors(u):
            if u < v:
                common = set(G.neighbors(u)) & set(G.neighbors(v))
                for w in common:
                    if v < w:
                        triangles.add((u, v, w))

    return sorted(triangles)

### All triangles with at least one major zoonotic source 
# find out any triangle that contains any items in a list of nodes
def find_triangles_with_nodes(triangles, nodes):
    """
    Return all triangles that contain any of the specified nodes.
    """
    nodes_set = set(nodes)
    matching_triangles = [tri for tri in triangles if any(node in nodes_set for node in tri)]
    return matching_triangles


MANUSCRIPT_TRIANGLE_ALLOWED_SOURCES = [
    "animal_feed",
    "poultry",
    "bovine",
    "equine",
    "swine",
    "wild_animal",
    "water_southeast",
    "fruit_vegetable",
]


def filter_manuscript_supported_triangles(
    supported_triangle_summary_df: pd.DataFrame,
    min_bootstrap_supported_rate: float = 0.75,
    allowed_best_source_nodes=None,
    excluded_supported_sources=("human",),
) -> pd.DataFrame:
    """
    Apply the final manuscript filter used after the full triangle screen.

    This mirrors the sandbox notebook: retain non-human supported sources whose
    best source is in the allowed source list and whose bootstrap-supported rate
    is greater than 0.75. With the submitted data this leaves six triangles.
    """
    if allowed_best_source_nodes is None:
        allowed_best_source_nodes = MANUSCRIPT_TRIANGLE_ALLOWED_SOURCES

    required = {
        "supported_source",
        "best_bootstrap_supported_rate",
        "best_source_node",
    }
    missing = required - set(supported_triangle_summary_df.columns)
    if missing:
        raise ValueError(f"Missing required triangle-summary columns: {sorted(missing)}")

    return supported_triangle_summary_df[
        ~supported_triangle_summary_df["supported_source"].isin(excluded_supported_sources)
        & (
            supported_triangle_summary_df["best_bootstrap_supported_rate"].isna()
            | (supported_triangle_summary_df["best_bootstrap_supported_rate"] > min_bootstrap_supported_rate)
        )
        & supported_triangle_summary_df["best_source_node"].isin(allowed_best_source_nodes)
    ].copy().reset_index(drop=True)
