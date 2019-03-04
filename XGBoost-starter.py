# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import pickle
import os
import gc
gc.enable()

train = pd.read_csv("../input/train.csv")

y = train.target.values
train.drop(['ID_code', 'target'], inplace=True, axis=1)
x = train.values

#Split data for training and validation
x, x_test, y, y_test = train_test_split(x, y, test_size=0.33, random_state=42, stratify=y)

#XGBoost
dtrain = xgb.DMatrix(x, label=y)
dtest = xgb.DMatrix(x_test, label=y_test)
evallist = [(dtest, 'eval'), (dtrain, 'train')]
param = {'booster': 'dart',
         'max_depth': 5,
         'learning_rate': 0.1,
         'objective': 'binary:logistic', 
         'silent': True,
         'sample_type': 'uniform',
         'normalize_type': 'tree',
         'rate_drop': 0.1,
         'skip_drop': 0.5,
         'eta': 1,
         'objective': 'binary:logistic',
         'nthread': 4,
         'eval_metric': 'auc'
        }

# Training & Validation
num_round = 50
bst = xgb.train(param, dtrain, num_round, evals=evallist, early_stopping_rounds=100, verbose_eval=10)

# Prediction
test.drop(['ID_code'], inplace=True, axis=1)
x_test = test.values

dtest = xgb.DMatrix(x_test)
ypred = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)

# Submission
submission = pd.read_csv('../input/sample_submission.csv')
submission['target'] = ypred
submission.to_csv('submission_xgb.csv', index=False)
