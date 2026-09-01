import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.exceptions import DataConversionWarning
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from knn_utils import timer
from knn_funcs import DATA_PATH
from knn_funcs import WeightKNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

tneighbors = 3
n_neighbors = 11
kfold = 10

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
    DATA_PATH, "metadata_training_testing.csv.gz")

print(f"Loading {data_parquet}")
df = pd.read_parquet(data_parquet)

print(df['Source'].value_counts())
df_meta = pd.read_csv(metadata, index_col=0, low_memory=False)
df_meta = df_meta[['Source']]
df_new = df[df['Source'].isin(TRAIN_TYPES)].copy()
print(df_new.shape)

df_new['Source'] = df_new['Source'].astype('category')
df_new.loc[:, 'source_emb'] = df_new['Source'].cat.codes
cat_codes = dict(zip(df_new['source_emb'], df_new['Source']))
print(cat_codes)
header = [cat_codes[x] for x in sorted(cat_codes.keys())]

y = df_new[['source_emb']]
y_label = df_new[['Source']]
barcodes = df_new[['Barcode']]
X = df_new.drop(['ST', 'Source', 'source_emb', 'Barcode'], axis=1)

folds = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42)
test_barcodes = []
test_fold = []
test_query_df = pd.DataFrame()
for fold, (train_id, test_id) in enumerate(folds.split(X, y)):
    print(f"\nFold {fold}")
    X_train = X.iloc[train_id].values
    y_train = y.iloc[train_id].values.ravel()
    barcodes_train = barcodes.iloc[train_id].values.ravel()
    X_valid = X.iloc[test_id].values
    y_valid = y.iloc[test_id].values.ravel()
    barcodes_valid = barcodes.iloc[test_id].values.ravel()
    df_valid = df_new.iloc[test_id]
    
    folds = [str(fold)]*len(X_valid)
    test_barcodes.extend(barcodes_valid)
    test_fold.extend(folds)

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
        tree_path='./',
        tree_by_cls_path='./',
        n_filtered_samples=None,  # minimum trust score clustering
        n_jobs=4)
    clf.fit(X_train, y_train, additional_labels=barcodes_train, fold_id=fold) # load the data, set up the BallTree

    # query neighbors
    valid_sample = {"features": X_valid,
               "labels": y_valid,
               "barcodes": barcodes_valid}
    score_kws={"n_neighbors_score": tneighbors, # how many neighbors to query for each cluster in trust score
               "beta": 2, # how spread the trust score is, the bigger the more concentrated
               "query_agg_func": np.median}
    valid_df = clf.query_sample_results(
                valid_sample,
                n_neighbors=n_neighbors, # how many neighbors from all classes to compute probability
                n_query_neighbors=n_neighbors, # how many neighbors to query overall for knn
                n_query_neighbors_score=tneighbors, # how many neighbors to query each cluster in self.trees[i] for the scores
                sort_by=['Trust'],
                **score_kws)

    test_query_df = pd.concat([test_query_df, valid_df], ignore_index=True)

    # accuracy
    y_valid = valid_df['Label']
    y_preds = valid_df['Prediction']
    accuracy=accuracy_score(y_valid,y_preds)
    print('KNN Model accuracy score: {0:0.4f}.\n'.format(accuracy))

    # view confusion matrix
    cm = confusion_matrix(y_valid,y_preds)
    print('Confusion matrix\n', cm)

    # accuracy for each class
    for x in sorted(cat_codes.keys()):
        print(cat_codes[x]+' accuracy: ', cm[x][x]/sum(cm[x]))
    print('\n')

fold_df = pd.DataFrame(data=test_fold,index=test_barcodes,columns=['Fold'])
test_query_df = test_query_df.set_index('Barcode')
test_query_df[header] = test_query_df['Prob (rounded)'].apply(pd.Series)
y_pred = [cat_codes[x] for x in test_query_df['Prediction']]
test_query_df['Prediction label'] = y_pred
test_query_df['Match'] = test_query_df['Label'] == test_query_df['Prediction']
output = pd.concat([df_meta,fold_df,test_query_df],axis=1,join='inner')
output['Trust'] = output['Trust'].round(3)
outfile = os.path.basename(data_parquet).split('.')[0]+"_CV.csv"
output.to_csv(outfile)
