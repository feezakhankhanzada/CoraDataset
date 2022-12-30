import pandas as pd
import numpy as np
import json
import tarfile
import os
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import preprocessing
import torch

file = tarfile.open('./cora.tgz')
file.extractall('./')
file.close()
data_dir = os.path.expanduser("./cora")

cite = pd.read_csv(os.path.join(data_dir, "cora.cites"), sep='\t', header=None, names=["target", "source"])

feature_names = ["w_{}".format(ii) for ii in range(1433)]
column_names =  feature_names + ["subject"]
content = pd.read_csv(os.path.join(data_dir, "cora.content"), sep='\t', header=None, names=column_names)
content = content.reset_index()
content = content.rename(columns = {"index" : "paper_id"})

for i in range(len(feature_names)):
    content[feature_names[i]] = content[feature_names[i]].astype(np.float64)

def create_graph():

    x = torch.from_numpy(content.iloc[: , 1:-1].to_numpy())

    le = preprocessing.LabelEncoder()
    le.fit(content.iloc[: , -1].tolist())
    le.classes_
    label = torch.from_numpy(le.transform(content.iloc[: , -1].tolist()))

    for i in range(len(content.index.unique())):
        cite.loc[cite.source == content.iloc[i , 0] , 's'] = i
        cite.loc[cite.target == content.iloc[i , 0] , 't'] = i

    cite.s = cite.s.astype(int)
    cite.t = cite.t.astype(int)

    edge_index = torch.from_numpy(cite[['t' , 's']].T.to_numpy())

    print("The Graph has " , x.shape[0] , " nodes, " , x.shape[1] , " features, " , edge_index.shape[1] , " edges, and " , torch.unique(label).shape[0] , " labels")
    
    return x , edge_index , label , le , content

def create_folds(folds , nfolds):
    n_samples = content.shape[0]
    kf = StratifiedKFold(n_splits = nfolds , shuffle = True , random_state = 123)
    kf_split = kf.split(np.zeros(n_samples) , np.zeros(n_samples))
    splits = {'n_samples': n_samples,
              'n_splits': nfolds,
              'cross_validator': kf.__str__(),
              'dataset': 'Cora'
              }
    for i , (_ , ids) in enumerate(kf_split):
        splits[i] = ids.tolist()
    with open('splits.json' , 'w') as f:
        json.dump(splits , f)
    with open('splits.json') as f:
        splits = json.load(f)

    assert splits['dataset'] == 'Cora' , "Unexpected dataset CV splits"
    assert splits['n_samples'] == n_samples , "Dataset length does not match"
    assert splits['n_splits'] > 0 , "Fold selection out of range"
    
    k = splits['n_splits']
    
    test_ids = splits[str(folds)]
    val_ids = splits[str((folds + 1) % 10)]
    train_ids = []
    
    for i in range(10):
        if i != folds and i != (folds + 1) % k:
            train_ids.extend(splits[str(i)])
            
    train_mask = content.index.isin(train_ids)
    val_mask = content.index.isin(val_ids)
    test_mask = content.index.isin(test_ids)

    return train_mask , val_mask , test_mask , train_ids , val_ids , test_ids
  