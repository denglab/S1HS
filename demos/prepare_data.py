"""Rebuild the small demo inputs from the full repository data."""
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 1127825
SOURCES = ['animal_feed', 'companion_animal', 'equine', 'fruit_and_vegetable',
           'food_production_env_pacific', 'water_pacific', 'rodents',
           'poultry', 'bovine', 'human']


def main():
    dest = ROOT / 'data' / 'demo'
    dest.mkdir(exist_ok=True)
    metadata = pd.read_csv(ROOT / 'data/metadata_training_testing.csv.gz',
                          usecols=['biosample_acc', 'curated_source_region', 'HC50'])
    metadata = metadata.dropna(subset=['curated_source_region', 'HC50'])
    network = pd.concat([metadata.loc[metadata.curated_source_region.eq(s)].sample(
        n=min(200, int(metadata.curated_source_region.eq(s).sum())), random_state=SEED) for s in SOURCES], ignore_index=True)
    train = pd.read_parquet(ROOT / 'data/KNN_training_cgMLST.parquet')
    train = pd.concat([train.loc[train.Source.eq(s)].sample(n=300, random_state=SEED)
                       for s in ['Bovine', 'Poultry', 'Swine']], ignore_index=True)
    query = pd.read_parquet(ROOT / 'data/KNN_testing_cgMLST.parquet').sample(
        n=30, random_state=SEED).reset_index(drop=True)
    assert set(train.Barcode).isdisjoint(query.Barcode)
    files = {}
    for filename, frame in [('network_metadata.csv.gz', network),
                            ('knn_training.csv.gz', train), ('knn_queries.csv.gz', query)]:
        path = dest / filename
        frame.to_csv(path, index=False, compression={'method': 'gzip', 'mtime': 0})
        files[filename] = {'rows': len(frame), 'columns': len(frame.columns),
                           'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {'seed': SEED, 'network_sources': SOURCES,
                'network_sampling': 'Up to 200 rows per source, after dropping missing source/HC50',
                'knn_sampling': '300 rows per training class; 30 testing rows; no replacement',
                'source_files': ['data/metadata_training_testing.csv.gz',
                                 'data/KNN_training_cgMLST.parquet',
                                 'data/KNN_testing_cgMLST.parquet'],
                'purpose': 'Small execution examples; not manuscript estimates or accuracy benchmarks.',
                'files': files}
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(files, indent=2))


if __name__ == '__main__':
    main()
