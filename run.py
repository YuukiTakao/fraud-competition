
import pandas as pd
import datetime
import logging
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import argparse
import json
# import numpy as np
import gc

from utils import load_datasets, load_target, tmp_load_datasets, preprocess
from logs.logger import log_best
from models.xgb import model_train

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/default.json')
parser.add_argument('-p','--preprocess', default=False, action="store_true")
options = parser.parse_args()
config = json.load(open(options.config))

now = datetime.datetime.now()
logger = logging.getLogger(__name__)
fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
logging.basicConfig(
    filename='./logs/log_{0:%Y-%m-%d_%H:%M:%S}.log'.format(now),
    level=logging.DEBUG,
    format=fmt 
)

logger.debug('./logs/log_{0:%Y-%m-%d_%H:%M:%S}.log'.format(now))

feature_names = config['features']
logger.debug(feature_names)

target_name = config['target_name']

if options.preprocess:
    logger.debug('started preprocess')
    preprocess()
    logger.debug('ended preprocess')

logger.debug('start train import')
X_test = tmp_load_datasets('data/input/processed_test.csv')
X_train_all = tmp_load_datasets('data/input/processed_train.csv')
logger.debug('ended train import')

y_train_all = load_target(target_name)
logger.debug(X_train_all.shape)

y_preds = []
models = []
scores = []

# TODOモデル名ごとのパラメーターを読み込むようにする
model_params = config['xgb_params']

n_fold = config['kfold']['n_splits']
kf = KFold(n_splits=n_fold, shuffle=True)

logger.debug('cv started')
for fold_n, (train_index, valid_index) in enumerate(kf.split(X_train_all)):

    X_train_, X_valid = X_train_all.iloc[train_index], X_train_all.iloc[valid_index]
    y_train_, y_valid = y_train_all.iloc[train_index], y_train_all.iloc[valid_index]

    model = model_train(
        X_train_,
        X_valid,
        y_train_,
        y_valid,
        model_params
    )
    pred=model.predict_proba(X_test)[:,1]
    models.append(model)

    y_preds.append(pred)

    del pred, model
    gc.collect()
    
    # スコア
    # log_best(model, config['loss'])
logger.debug('cv ended')

# CVスコア
scores = [
    m.best_score for m in models
]
score = sum(scores) / len(scores)
print('===CV scores===')
print(scores)
print(score)
logger.debug('===CV scores===')
logger.debug(scores)
logger.debug(score)

# submitファイルの作成
ID_name = config['ID_name']
# 出力ファイルのid一覧取得
sub = pd.DataFrame(pd.read_csv('./data/input/sample_submission.csv.zip')[ID_name])

# 予測値の平均を取る
y_sub = sum(y_preds) / len(y_preds)

sub[target_name] = y_sub

score = str(score)
sub.to_csv(
    './data/output/sub_{0:%Y-%m-%d_%H:%M:%S}_{1}.csv'.format(now, score[:8]),
    index=False
)