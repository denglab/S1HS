"""Predict source labels for bundled queries with the study's weighted KNN model."""
import argparse
from pathlib import Path
import sys
import os
import time
import json
import platform

START = time.perf_counter()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'KNN'))
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from knn_funcs import WeightKNeighborsClassifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/demo_knn')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(ROOT / 'data/demo/knn_training.csv.gz')
    queries = pd.read_csv(ROOT / 'data/demo/knn_queries.csv.gz')
    assert set(train.Barcode).isdisjoint(queries.Barcode)
    labels = ['Bovine', 'Poultry', 'Swine']
    features = [c for c in train if c not in ['Barcode', 'Source', 'ST']]
    def profiles(frame):
        return (frame[features].apply(pd.to_numeric, errors='coerce')
                .replace(-1, np.nan).fillna(0).astype('int16'))
    x, q = profiles(train), profiles(queries)
    y = pd.Categorical(train.Source, categories=labels).codes
    assert (y >= 0).all()
    model = WeightKNeighborsClassifier(
        n_neighbors=11, algorithm='ball_tree', leaf_size=20, sampling=False,
        min_dist=5e-5, proba_method='softmax', delta=1, beta=0.5, gamma=2.5,
        alpha=0.0, filtering_method=None, pre_filtering=True, tree_path=None,
        tree_by_cls_path=None, n_filtered_samples=None, n_jobs=1)
    # The existing model writes its BallTree cache in the working directory.
    previous = Path.cwd()
    try:
        os.chdir(out)
        model.fit(x, y, additional_labels=train.Barcode.to_numpy())
        predictions = model.query_sample_results(
            {'features': q.to_numpy(), 'labels': queries.Source.to_numpy(),
             'barcodes': queries.Barcode.to_numpy()},
            n_neighbors=11, n_query_neighbors=11, n_query_neighbors_score=3,
            sort_by=['Trust'], n_neighbors_score=3, beta=2, query_agg_func=np.median)
    finally:
        os.chdir(previous)
    predictions['Prediction label'] = [labels[int(i)] for i in predictions.Prediction]
    predictions[labels] = pd.DataFrame(predictions['Prob (rounded)'].tolist(), index=predictions.index)
    predictions.to_csv(out / 'predictions.csv', index=False)
    summary = {'training_rows': len(train), 'query_rows': len(queries),
               'prediction_rows': len(predictions), 'loci': len(features),
               'training_classes': labels, 'neighbors': 11, 'python': platform.python_version(),
               'elapsed_seconds': round(time.perf_counter() - START, 2)}
    (out / 'run_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print('Outputs:', out)


if __name__ == '__main__':
    main()
