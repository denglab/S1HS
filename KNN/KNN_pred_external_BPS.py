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

test_file = 'KNN_external_BPS'
tneighbors = 3

pd.options.display.max_columns = 999
warnings.simplefilter(action='ignore', category=DataConversionWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
np.random.seed(1127825)

TRAIN_TYPES = ['Poultry', 'Bovine', 'Swine']
TEST_TYPES = ['Poultry', 'Bovine', 'Swine']

data_parquet = os.path.join(
    DATA_PATH, 'KNN_training_cgMLST.parquet')
metadata = os.path.join(
    DATA_PATH, "metadata_training_testing.csv.gz")

print(f"Loading {data_parquet}")
df = pd.read_parquet(data_parquet)

df_meta = pd.read_csv(metadata, index_col=0, low_memory=False)
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

n_neighbors = 11
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

test_meta = os.path.join(DATA_PATH, test_file+'_metadata.txt.gz')
df_meta = pd.read_csv(test_meta, low_memory = False, sep = '\t',index_col='Barcode')
df_meta['Source'] = df_meta['Source Type'] 
test_csv = os.path.join(DATA_PATH, test_file+'_cgMLST.csv.gz')
df_test = pd.read_csv(test_csv, low_memory = False)
df_test = df_test.rename(columns={'Source Type':'Source'})

feature_cols = X.columns
df_test[feature_cols] = df_test[feature_cols].apply(pd.to_numeric, errors='coerce')
df_test[feature_cols] = df_test[feature_cols].replace('-', np.nan)
df_test[feature_cols] = df_test[feature_cols].replace(-1, np.nan)
df_test[feature_cols] = df_test[feature_cols].fillna(0).astype('int16')

X_test = df_test[X.columns]
X_test = X_test.values
y_test = df_test[['Source']].values.ravel()
ids = df_test[['Barcode']].values.ravel()

preds_test = clf.predict(X_test)
print(preds_test)

test_sample = {"features": X_test, "labels": y_test, "barcodes": ids}
score_kws={"n_neighbors_score": tneighbors, # how many neighbors to query for each cluster in trust score
               "beta": 2, # how spread the trust score is, the bigger the more concentrated
               "query_agg_func": np.median}
df_query = clf.query_sample_results(
    test_sample,
    n_neighbors=11, # how many neighbors from all classes to compute probability
    n_query_neighbors=11, # how many neighbors to query for each cluster to print out
    n_query_neighbors_score=3, # how many neighbors to query each cluster in self.trees[i] for the scores
    sort_by=['Trust'],
    **score_kws
)

df_query=df_query.set_index('Barcode')
df_query[header] = df_query['Prob (rounded)'].apply(pd.Series)
y_pred = [cat_codes[x] for x in df_query['Prediction']]
df_query['Prediction label'] = y_pred
df_query['Match'] = df_query['Label'] == df_query['Prediction label']
df_out = pd.concat([df_meta,df_query],axis=1,join='inner')
columns = [col for col in df_out.columns if col not in ['Source Type', 'Trust']] + ['Source Type', 'Trust']
df_out = df_out[columns]
df_out['Trust'] = df_out['Trust'].round(3)
outfile = test_file+"_pred.csv"
df_out.to_csv(outfile)
