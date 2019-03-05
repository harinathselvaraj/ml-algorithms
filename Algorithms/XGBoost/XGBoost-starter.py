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
x, x_val, y, y_val = train_test_split(x, y, test_size=0.33, random_state=42, stratify=y)

#XGBoost
dtrain = xgb.DMatrix(x, label=y)
dval = xgb.DMatrix(x_val, label=y_val)
evallist = [(dval, 'eval'), (dtrain, 'train')]

param = {
          'booster': 'dart', 
          'silent':True,
 #        'scale_pos_weight':1,
          'learning_rate':0.01,  
          'colsample_bytree' :0.4,
          'subsample' : 0.8,
          'objective':'binary:logistic', 
          'n_estimators':100,  #100 if the size of your data is high, 1000 is if it is medium-low
          'reg_alpha' : 0.3,
          'max_depth':4, 
          'gamma':10,
          'eval_metric': 'auc'
#          'max_depth': 4, #increase to overfit
#          'learning_rate': 0.1,
#          'objective': 'binary:logistic', 
#          'silent': True,
#          'sample_type': 'uniform',
#          'normalize_type': 'tree',
#          'rate_drop': 0.1,
#          'skip_drop': 0.5,
#          'eta': 1,
#          'objective': 'binary:logistic',
#          'nthread': 4,
#          'eval_metric': 'auc'
        }

# Training & Validation
num_round = 50
bst = xgb.train(param, dtrain, num_round, evals=evallist, early_stopping_rounds=100, verbose_eval=10)

# Feature Importance Tree Plot - Use them seperately
rcParams['figure.figsize'] = 180,150
xgb.plot_importance(bst)
xgb.plot_tree(bst, num_trees=2)
xgb.to_graphviz(bst, num_trees=2)


# Prediction
test.drop(['ID_code'], inplace=True, axis=1)
x_test = test.values

dtest = xgb.DMatrix(x_test)
ypred = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)

# Submission
submission = pd.read_csv('../input/sample_submission.csv')
submission['target'] = ypred
submission.to_csv('submission_xgb.csv', index=False)