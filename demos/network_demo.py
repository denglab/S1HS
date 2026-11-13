"""Construct a small bootstrap consensus network using the study functions."""
import argparse
from pathlib import Path
import sys
import time
import json
import platform

START = time.perf_counter()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'network'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from network_funcs import build_similarity_network, node_metrics_weighted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/demo_network')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ROOT / 'data/demo/network_metadata.csv.gz')
    result = build_similarity_network(
        data, feature_col='HC50', class_col='curated_source_region',
        similarity='cosine', sim_threshold=0.35, bootstrap_iters=100,
        bootstrapping_strategy='stratified', community_detection_method='leiden',
        consensus_partition_threshold=0.5, consensus_graph_threshold=0.5,
        show=False, seed=186)
    graph, partition = result[2], result[6]
    nodes = node_metrics_weighted(graph)
    nodes['community'] = pd.Series(partition)
    nodes.to_csv(args.output_dir / 'nodes.csv', index_label='node')
    edges = pd.DataFrame([(a, b, d['weight']) for a, b, d in graph.edges(data=True)],
                         columns=['source', 'target', 'bootstrap_support'])
    edges.to_csv(args.output_dir / 'edges.csv', index=False)
    result[12].to_csv(args.output_dir / 'edge_support.csv', index_label='node')
    fig, ax = plt.subplots(figsize=(10, 7))
    nx.draw_networkx(graph, pos=nx.spring_layout(graph, seed=186), ax=ax,
                     node_color=[partition[n] for n in graph], cmap=plt.cm.Set2,
                     node_size=800, font_size=8,
                     labels={n: n.replace('_', '\n') for n in graph})
    ax.set_title('Demo consensus network (100 bootstrap iterations)')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(args.output_dir / 'network.png', dpi=160)
    plt.close(fig)
    summary = {'input_rows': len(data), 'nodes': len(graph), 'edges': graph.number_of_edges(),
               'bootstrap_iterations': 100, 'seed': 186, 'python': platform.python_version(),
               'elapsed_seconds': round(time.perf_counter() - START, 2)}
    (args.output_dir / 'run_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print('Outputs:', args.output_dir.resolve())


if __name__ == '__main__':
    main()
