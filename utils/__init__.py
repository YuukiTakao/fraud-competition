import time
import pandas as pd
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

@contextmanager
def timer(name):
    t0 = time.time()
    print(f'[{name}] start')
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def load_datasets(feats):
    dfs = [pd.read_feather(f'features/{f}_train.feather') for f in feats]
    X_train = pd.concat(dfs, axis=1)
    dfs = [pd.read_feather(f'features/{f}_test.feather') for f in feats]
    X_test = pd.concat(dfs, axis=1)
    return X_train, X_test

def preprocess():
    train_id = pd.read_csv('./data/input/train_identity.csv.zip')
    train_trans = pd.read_csv('./data/input/train_transaction.csv.zip')
    test_id = pd.read_csv('./data/input/test_identity.csv.zip')
    test_trans = pd.read_csv('./data/input/test_transaction.csv.zip')

    merged_train = train_trans.merge(train_id, how='left', left_index=True, right_index=True)
    merged_test = test_trans.merge(test_id, how='left', left_index=True, right_index=True)
    del train_trans, train_id
    del test_trans, test_id

    X_train = merged_train.fillna(-999)
    merged_test = merged_test.fillna(-999)
    X_test = merged_test.copy()
    del merged_train, merged_test

    X_train['Trans_min_mean'] = X_train['TransactionAmt'] - X_train['TransactionAmt'].mean()
    X_train['Trans_min_std'] = X_train['Trans_min_mean'] / X_train['TransactionAmt'].std()
    X_test['Trans_min_mean'] = X_test['TransactionAmt'] - X_test['TransactionAmt'].mean()
    X_test['Trans_min_std'] = X_test['Trans_min_mean'] / X_test['TransactionAmt'].std()

    X_train['TransactionAmt_to_mean_card1'] = X_train['TransactionAmt'] / X_train.groupby(['card1'])['TransactionAmt'].transform('mean')
    X_train['TransactionAmt_to_mean_card4'] = X_train['TransactionAmt'] / X_train.groupby(['card4'])['TransactionAmt'].transform('mean')
    X_train['TransactionAmt_to_std_card1'] = X_train['TransactionAmt'] / X_train.groupby(['card1'])['TransactionAmt'].transform('std')
    X_train['TransactionAmt_to_std_card4'] = X_train['TransactionAmt'] / X_train.groupby(['card4'])['TransactionAmt'].transform('std')

    X_test['TransactionAmt_to_mean_card1'] = X_test['TransactionAmt'] / X_test.groupby(['card1'])['TransactionAmt'].transform('mean')
    X_test['TransactionAmt_to_mean_card4'] = X_test['TransactionAmt'] / X_test.groupby(['card4'])['TransactionAmt'].transform('mean')
    X_test['TransactionAmt_to_std_card1'] = X_test['TransactionAmt'] / X_test.groupby(['card1'])['TransactionAmt'].transform('std')
    X_test['TransactionAmt_to_std_card4'] = X_test['TransactionAmt'] / X_test.groupby(['card4'])['TransactionAmt'].transform('std')

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
        'card4',
        'card6',
        'M1',
        'M2',
        'M3',
        'M5',
        'M6',
        'M7',
        'M8',
        'M9',
        'M4',
        'P_emaildomain',
        'R_emaildomain',
    ]
    X_train = drop_columns(X_train, drop_col_names)
    X_train = X_train.drop('isFraud', axis=1)
    X_test = drop_columns(X_test, drop_col_names)

    X_test.to_csv('../data/input/processed_test.csv')
    X_train.to_csv('../data/input/processed_train.csv')


def tmp_load_datasets(import_path):
    return  pd.read_csv(import_path)


def load_target(target_name):
    train = pd.read_csv('./data/input/train_transaction.csv.zip')
    y_train = train[target_name]
    return y_train

# データフレームの不要カラム名を配列で受け取って削除する
def drop_columns(df, col_name_array):
    for _, col_name in enumerate(col_name_array):
        df = df.drop(col_name, axis=1)
    return df