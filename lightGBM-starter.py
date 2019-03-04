import numpy as np
import pandas as pd
import lightgbm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# Prepare the data
train = pd.read_csv('../input/train.csv')

# Get the labels
y = train.target.values
train.drop(['id', 'target'], inplace=True, axis=1)
x = train.values

#Split for training and validation
x, x_test, y, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

#Define categorical features, training and validation data
categorical_features = [c for c, col in enumerate(train.columns) if 'cat' in col] #prefix cat- for all categorical columns
train_data = lightgbm.Dataset(x, label=y, categorical_feature=categorical_features)
test_data = lightgbm.Dataset(x_test, label=y_test)

parameters = {
    'application': 'binary',
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': 'true',
    'boosting': 'gbdt',
    'num_leaves': 31,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'learning_rate': 0.05,
    'verbose': 0
}

# Training & Validation
model = lightgbm.train(parameters,
                       train_data,
                       valid_sets=test_data,
                       num_boost_round=5000,
                       early_stopping_rounds=100)

# Prediction
y = model.predict(x)


#Submission
submission = pd.read_csv('../input/test.csv')
ids = submission['id'].values
submission.drop('id', inplace=True, axis=1)

x = submission.values

output = pd.DataFrame({'id': ids, 'target': y})
output.to_csv("submission.csv", index=False)