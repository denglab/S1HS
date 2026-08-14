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
np.random.seed(1127825)

TRAIN_TYPES = ['Poultry', 'Bovine', 'Swine']
TEST_TYPES = ['Poultry', 'Bovine', 'Swine']

data_parquet = os.path.join(
    DATA_PATH, 'KNN_training_cgMLST.parquet')
metadata = os.path.join(
    DATA_PATH, "KNN_training_metadata.txt")

print(f"Loading {data_parquet}")
df = pd.read_parquet(data_parquet)
df = df.sample(n=10000, random_state=1127825)  # Adjust n as needed

df_meta = pd.read_csv(metadata,  delimiter="\t", index_col=0, low_memory=False)
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

df_test = df[~df['Source'].isin(TRAIN_TYPES)].copy()
df_test = df_test.sample(n=1000, random_state=1127825) 

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
columns = [col for col in df_out.columns if col not in ['Source', 'Trust']] + ['Source', 'Trust']
df_out = df_out[columns]
df_out['Trust'] = df_out['Trust'].round(3)
outfile = os.path.basename(metadata).split('.')[0]+"_pred.csv"
df_out.to_csv(outfile)
