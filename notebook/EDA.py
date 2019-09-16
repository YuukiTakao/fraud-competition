# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import datetime as dt
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pandas_profiling as pdp
import seaborn as sns
import xgboost as xgb
import gc
from sklearn.preprocessing import LabelEncoder

# 交差検証
from sklearn.model_selection import TimeSeriesSplit,KFold

# モデルの評価指標
from sklearn.metrics import roc_auc_score

pd.set_option('display.max_columns', 400)
pd.set_option('display.max_rows', 500)

plt.style.use('ggplot') 
font = {'family' : 'meiryo'}
matplotlib.rc('font', **font)

# +
train_id = pd.read_csv('../data/input/train_identity.csv.zip')
train_trans = pd.read_csv('../data/input/train_transaction.csv.zip')

test_id = pd.read_csv('../data/input/test_identity.csv.zip')
test_trans = pd.read_csv('../data/input/test_transaction.csv.zip')

sample_submission = pd.read_csv('../data/input/sample_submission.csv.zip', index_col='TransactionID')
# -

train_id.head()

train_trans.head()

# +
merged_train = train_trans.merge(train_id, how='left', left_index=True, right_index=True)
del train_trans, train_id

merged_test = test_trans.merge(test_id, how='left', left_index=True, right_index=True)
del test_trans, test_id

gc.collect


# +
def get_feat_names():
    for col in merged_train.columns:
        print('"'+col+'",')

# get_feat_names()
# -



print(merged_train.shape)
print(merged_test.shape)

# +
y_train = merged_train['isFraud'].copy()

X_train = merged_train.drop('isFraud', axis=1)
X_test = merged_test.copy()
del merged_train, merged_test

X_train = X_train.fillna(-999)
X_test = X_test.fillna(-999)

# -

X_train['Trans_min_mean'] = X_train['TransactionAmt'] - X_train['TransactionAmt'].mean()
X_train['Trans_min_std'] = X_train['Trans_min_mean'] / X_train['TransactionAmt'].std()
X_test['Trans_min_mean'] = X_test['TransactionAmt'] - X_test['TransactionAmt'].mean()
X_test['Trans_min_std'] = X_test['Trans_min_mean'] / X_test['TransactionAmt'].std()

# あとでじっくりこの処理確認する
X_train.groupby(['card1'])['TransactionAmt'].transform('mean')

# +
X_train['TransactionAmt_to_mean_card1'] = X_train['TransactionAmt'] / X_train.groupby(['card1'])['TransactionAmt'].transform('mean')
X_train['TransactionAmt_to_mean_card4'] = X_train['TransactionAmt'] / X_train.groupby(['card4'])['TransactionAmt'].transform('mean')
X_train['TransactionAmt_to_std_card1'] = X_train['TransactionAmt'] / X_train.groupby(['card1'])['TransactionAmt'].transform('std')
X_train['TransactionAmt_to_std_card4'] = X_train['TransactionAmt'] / X_train.groupby(['card4'])['TransactionAmt'].transform('std')

X_test['TransactionAmt_to_mean_card1'] = X_test['TransactionAmt'] / X_test.groupby(['card1'])['TransactionAmt'].transform('mean')
X_test['TransactionAmt_to_mean_card4'] = X_test['TransactionAmt'] / X_test.groupby(['card4'])['TransactionAmt'].transform('mean')
X_test['TransactionAmt_to_std_card1'] = X_test['TransactionAmt'] / X_test.groupby(['card1'])['TransactionAmt'].transform('std')
X_test['TransactionAmt_to_std_card4'] = X_test['TransactionAmt'] / X_test.groupby(['card4'])['TransactionAmt'].transform('std')
# -

cat_cols = ['card4',
            'card6',
            'P_emaildomain',
            'R_emaildomain',
           ]
for col in cat_cols:
    if col in X_train.columns:
        le = LabelEncoder()
        le.fit(list(X_train[col].astype(str).values) + list(X_test[col].astype(str).values))
        X_train[col] = le.transform(list(X_train[col].astype(str).values))
        X_test[col] = le.transform(list(X_test[col].astype(str).values)) 


# +
# データフレームの不要カラム名を配列で受け取って削除する
def drop_columns(df, col_name_array):
    for _, col_name in enumerate(col_name_array):
        df = df.drop(col_name, axis=1)
    return df

drop_col_names = [
    'id_12',
    'id_15',
    'id_16',
    'id_23',
    'id_27',
    'id_28',
    'id_29',
    'id_30',
    'id_31',
    'id_33',
    'id_34',
    'id_35',
    'id_36',
    'id_37',
    'id_38',
    'DeviceType',
    'DeviceInfo',
    'ProductCD',
    'M1',
    'M2',
    'M3',
    'M5',
    'M6',
    'M7',
    'M8',
    'M9',
    'M4',
]
# -

X_train = drop_columns(X_train, drop_col_names)
X_test = drop_columns(X_test, drop_col_names)

X_train = X_train.fillna(-999)
X_test = X_test.fillna(-999)

X_test.to_csv('../data/input/processed_test.csv')
X_train.to_csv('../data/input/processed_train.csv')

X_train.values

xgb_submission=sample_submission.copy()
xgb_submission['isFraud'] = 0

y_preds = []
models = []
scores = []


# +
n_fold = 2
folds = KFold(n_splits=n_fold,shuffle=True)

print(folds)
# -

param_dist = {
    "objective":"binary:logistic",
    "n_estimators":10,
    "max_depth":9,
    "learning_rate":0.048,
    "subsample":0.85,
    "colsample_bytree":0.85,
    "missing":-999,
    "tree_method":"hist",
    "reg_alpha":0.15,
    "reg_lamdba":0.85
}

# +
# %%time

for fold_n, (train_index, valid_index) in enumerate(kf.split(X_train)):
    xgbclf = xgb.XGBClassifier(**param_dist)

    X_train_, X_valid = X_train.iloc[train_index], X_train.iloc[valid_index]
    y_train_, y_valid = y_train.iloc[train_index], y_train.iloc[valid_index]
    xgbclf.fit(X_train_,y_train_,
              eval_set=[(X_train, y_train), (X_valid, y_valid)],
              eval_metric='auc',
              early_stopping_rounds=100,
              )

    del X_train_,y_train_    
    models.append(xgbclf)

    pred=xgbclf.predict_proba(X_test)[:,1]
#     val=xgbclf.predict_proba(X_valid)[:,1]
    y_preds.append(pred)

#     scores.append(roc_auc_score(y_valid, val))


#     print('ROC accuracy: {}'.format(roc_auc_score(y_valid, val)))

# +
# # 調整するハイパーパラメータの値の候補を設定
# test_params = {
#     'n_estimators':[100,1000]
# }

# +
# # %%time
# from sklearn.model_selection import GridSearchCV    
# # グリッドサーチCVの実行
# gs = GridSearchCV(estimator = xgb.XGBClassifier(**param_dist),
#                            param_grid = test_params, scoring='roc_auc',
#                            cv = 2, return_train_score=False)

# #     X_train_, X_valid = X_train.iloc[train_index], X_train.iloc[valid_index]
# #     y_train_, y_valid = y_train.iloc[train_index], y_train.iloc[valid_index]
# gs.fit(X_train,y_train)

# best = gs.best_estimator_
# pred=best.predict_proba(X_test)[:,1]

# y_preds.append(pred)
# -

scores = []
# for m in models:
#     print(len(m.evals_result()['validation_0']['logloss']))
#     print(max(m.evals_result()['validation_0']['logloss']))
#     print(m.evals_result()['validation_0']['logloss'])
scores = [
    m.best_score for m in models
#     gs.cv_results_['mean_test_score']
]
score = sum(scores) / len(scores)
score

best_model = models[0]

type(best_model)

_, ax = plt.subplots(figsize=(16, 14))


def plot_importance_save_img(bst, plot_param=None):
    xgb.plot_importance(bst, ax=ax, max_num_features=50)


plt.figure(figsize=(16, 12))
xgb.plot_importance(best_model, max_num_features=50)

plt.show()

plot_importance_save_img(best_model, plot_param)

plt.savefig('./figure.png')

# +
# print(gs.best_params_)
# print(gs.cv_results_['mean_test_score'])
# print(gs.best_estimator_)
# -

y_preds

y_sub = sum(y_preds) / len(y_preds)
sample_submission['isFraud'] = y_sub

import datetime 

score = str(score)
score

path = './data/input/'
print(path + 'sub_{0:%Y-%m-%d_%H:%M:%S}_{1}.csv'.format(datetime.datetime.now(), score[:8]))

sample_submission.to_csv('sub_{0:%Y-%m-%d_%H:%M:%S}_{1}.csv'.format(datetime.datetime.now(), score[:8]))

# +
# best = gs.best_estimator_

# +
# sum(gs.cv_results_['mean_test_score']) / len(gs.cv_results_['mean_test_score'])
# -


