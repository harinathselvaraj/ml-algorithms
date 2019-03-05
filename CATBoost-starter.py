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
from catboost import CatBoostClassifier, Pool


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


y = train.target.values
train.drop(['ID_code', 'target'], inplace=True, axis=1)
x = train.values

test.drop(['ID_code'], inplace=True, axis=1)
x_test = test.values

#Split data for training and validation
x, x_val, y, y_val = train_test_split(x, y, test_size=0.20, random_state=42, stratify=y)

#Define categorical features, training and validation data

#CATBoost
model = CatBoostClassifier(iterations=50, 
                           depth=6, 
                           learning_rate=0.5,
                           loss_function='Logloss')
# Training & Validation
model.fit(x, y, eval_set=[(x_val, y_val)], verbose=10, early_stopping_rounds=100)

# Prediction
ypred = model.predict(x_test)
preds_proba = model.predict_proba(x_test)
print("class = ", ypred)
print("proba = ", preds_proba)

# Submission
submission = pd.read_csv('../input/sample_submission.csv')
submission['target'] = ypred
submission.to_csv('submission_xgb.csv', index=False)