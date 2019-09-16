import xgboost as xgb
import logging

from logs.logger import log_evaluation

logger = logging.getLogger(__name__)

def model_train(X_train, X_valid, y_train, y_valid, model_params):
    logger.debug('model_train started')

    xgbclf = xgb.XGBClassifier(**model_params)
    xgbclf.fit(X_train,y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_metric='auc',
            early_stopping_rounds=100
            )
    del X_train,y_train

    logger.debug('model_train ended')
    return xgbclf