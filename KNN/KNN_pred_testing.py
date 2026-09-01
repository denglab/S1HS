"""Demo/sensitivity KNN run for non-BPS cases.

This script deliberately uses fixed-size, seeded subsamples of the training and
non-BPS query records so it can run quickly and reproducibly as a sensitivity
check. It is not the manuscript-facing full external prediction path; use
KNN_pred_BPS.py for the full external prediction workflow and KNN_CV.py for
cross-validation.
"""

import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.exceptions import DataConversionWarning
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from knn_funcs import WeightKNeighborsClassifier
from knn_funcs import DATA_PATH
from knn_utils import timer

tneighbors = 3
n_neighbors = 11

pd.options.display.max_columns = 999
warnings.simplefilter(action='ignore', category=DataConversionWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
sns.set_theme("notebook", style="whitegrid", font="sans-serif", font_scale=1.5, color_codes=True, rc=None)
DEMO_RANDOM_SEED = 1127825
DEMO_TRAIN_SAMPLE_N = 1000
DEMO_TEST_SAMPLE_N = 100

np.random.seed(DEMO_RANDOM_SEED)
print(
    "Running KNN_pred_nBPS.py as a demo/sensitivity script with "
    f"{DEMO_TRAIN_SAMPLE_N} training rows and {DEMO_TEST_SAMPLE_N} non-BPS query rows "
    f"sampled using random_state={DEMO_RANDOM_SEED}."
)

TRAIN_TYPES = ['Poultry', 'Bovine', 'Swine']
TEST_TYPES = ['Poultry', 'Bovine', 'Swine']

data_parquet = os.path.join(
    DATA_PATH, 'KNN_training_cgMLST.parquet')
test_parquet = os.path.join(
    DATA_PATH, 'KNN_testing_cgMLST.parquet')
metadata = os.path.join(
    DATA_PATH, "metadata_training_testing.csv.gz")

print(f"Loading {data_parquet}")
df = pd.read_parquet(data_parquet)

if len(df) < DEMO_TRAIN_SAMPLE_N:
    raise ValueError(f"Requested {DEMO_TRAIN_SAMPLE_N} demo training rows, but only {len(df)} are available.")
df = df.sample(n=DEMO_TRAIN_SAMPLE_N, random_state=DEMO_RANDOM_SEED)

df_meta = pd.read_csv(metadata, index_col=0, low_memory=False)
df_meta = df_meta[['Source']]

df_new = df[df['Source'].isin(TRAIN_TYPES)].copy()
df_new['Source'] = df_new['Source'].astype('category')
df_new.loc[:, 'source_emb'] = df_new['Source'].cat.codes
cat_codes = dict(zip(df_new['source_emb'], df_new['Source']))
print(cat_codes)
header = [cat_codes[x] for x in sorted(cat_codes.keys())]

y = df_new[['source_emb']]
y_label = df_new[['Source']]
barcodes = df_new[['Barcode']].values.ravel()

X = df_new.drop(['ST', 'Source', 'source_emb', 'Barcode'], axis=1)
clf = WeightKNeighborsClassifier(
        n_neighbors=n_neighbors,
        algorithm="ball_tree",
        leaf_size=20,
        sampling=False, # sampling the training data to do knn
        min_dist=5e-5,  # distance cut-off
        proba_method="softmax",
        delta=1,
        beta=0.5,
        gamma=2.5,
        alpha=0.0,  # for score confidence
        filtering_method=None,  # for score confidence
        pre_filtering=True,  # for score confidence
        tree_path=None,
        tree_by_cls_path=None,
        n_filtered_samples=None,  # minimum trust score clustering
        n_jobs=4)
clf.fit(X, y, additional_labels=barcodes) # load the data, set up the BallTree

df_test = pd.read_parquet(test_parquet)

feature_cols = X.columns
df_test[feature_cols] = df_test[feature_cols].apply(pd.to_numeric, errors='coerce')
df_test[feature_cols] = df_test[feature_cols].replace('-', np.nan)
df_test[feature_cols] = df_test[feature_cols].replace(-1, np.nan)
df_test[feature_cols] = df_test[feature_cols].fillna(0).astype('int16')

if len(df_test) < DEMO_TEST_SAMPLE_N:
    raise ValueError(f"Requested {DEMO_TEST_SAMPLE_N} demo query rows, but only {len(df_test)} are available.")
df_test = df_test.sample(n=DEMO_TEST_SAMPLE_N, random_state=DEMO_RANDOM_SEED)

X_test = df_test[X.columns]
X_test = X_test.values
y_test = df_test[['Source']].values.ravel()
ids = df_test[['Barcode']].values.ravel()

test_sample = {"features": X_test, "labels": y_test, "barcodes": ids}
score_kws={"n_neighbors_score": tneighbors, # how many neighbors to query for each cluster in trust score
               "beta": 2, # how spread the trust score is, the bigger the more concentrated
               "query_agg_func": np.median}
df_query = clf.query_sample_results(
    test_sample,
    n_neighbors=n_neighbors, # how many neighbors from all classes to compute probability
    n_query_neighbors=n_neighbors, # how many neighbors to query for each cluster to print out
    n_query_neighbors_score=tneighbors, # how many neighbors to query each cluster in self.trees[i] for the scores
    sort_by=['Trust'],
    **score_kws
)

df_query=df_query.set_index('Barcode')
df_query[header] = df_query['Prob (rounded)'].apply(pd.Series)
y_pred = [cat_codes[x] for x in df_query['Prediction']]
df_query['Prediction label'] = y_pred
df_out = pd.concat([df_meta,df_query],axis=1,join='inner')
df_out['Trust'] = df_out['Trust'].round(3)
outfile = os.path.basename(test_parquet).split('.')[0] + "_nBPS_demo_sensitivity_pred.csv"
df_out.to_csv(outfile)
