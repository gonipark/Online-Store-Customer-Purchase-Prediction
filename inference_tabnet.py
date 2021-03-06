# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

import os, sys, gc, warnings, random

import datetime
import dateutil.relativedelta

# Data manipulation
import pandas as pd 
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GroupKFold
from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb

from tqdm.notebook import trange, tqdm

pd.options.display.max_rows = 10000
pd.options.display.max_columns = 1000
pd.options.display.max_colwidth = 1000
# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import os, sys, gc, random
import datetime
import dateutil.relativedelta
import argparse

# Machine learning
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
import xgboost as xg

# Custom library
from utils import seed_everything, print_score
from features import generate_label, feature_engineering2

from pytorch_tabnet.multitask import TabNetMultiTaskClassifier
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

from pytorch_tabnet.pretraining import TabNetPretrainer


import torch
# from sklearn.linear_model import LinearRegression
# from sklearn.neural_network import MLPRegressor
# from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
# from sklearn.metrics import mean_absolute_error

# models = [['Linear Regression', LinearRegression(n_jobs=-1)], 
#           ['MultiLayerPerceptron', MLPRegressor(random_state=2020)],
#           ['RandomForest', RandomForestRegressor(random_state=2020,
#                                                   n_jobs=-1)], 
#           ['XGBoost', XGBRegressor(random_state=2020,
#                                     n_jobs=7)], 
#           ['LightGBM', LGBMRegressor(random_state=2020,
#                                       n_jobs=7)]]


TOTAL_THRES = 300 # ????????? ?????????
SEED = 42 # ?????? ??????
seed_everything(SEED) # ?????? ??????


data_dir = '../input' # os.environ['SM_CHANNEL_TRAIN']
model_dir = '../model' # os.environ['SM_MODEL_DIR']
output_dir = '../output' # os.environ['SM_OUTPUT_DATA_DIR']


def make_lgb_oof_prediction(train, y, test, features, categorical_features='auto', model_params=None, folds=10):
    ####################MLFLOW###########################
    import mlflow
    HOST = "http://localhost"
    mlflow.set_tracking_uri(HOST+":6006/")
    mlflow.start_run()
    ####################MLFLOW###########################

    x_train = train[features]
    x_test = test[features]
    
    # ????????? ????????? ???????????? ????????? ??????
    test_preds = np.zeros(x_test.shape[0])
    
    # Out Of Fold Validation ?????? ???????????? ????????? ??????
    y_oof = np.zeros(x_train.shape[0])
    
    # ????????? ?????? Validation ???????????? ????????? ??????
    score = 0
    
    # ?????? ???????????? ????????? ????????? ????????? ??????
    fi = pd.DataFrame()
    fi['feature'] = features
    
    # Stratified K Fold ??????
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(x_train, y)):
        # train index, validation index??? train ???????????? ??????
        x_tr, x_val = x_train.loc[tr_idx, features], x_train.loc[val_idx, features]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        print(f'fold: {fold+1}, x_tr.shape: {x_tr.shape}, x_val.shape: {x_val.shape}')

        # # LightGBM ???????????? ??????
        # dtrain = clf.Dataset(x_tr, label=y_tr)
        # dvalid = clf.Dataset(x_val, label=y_val)

        unsupervised_model = TabNetPretrainer(
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            mask_type='sparsemax' # "sparsemax"
        )

        unsupervised_model.fit(
            X_train=x_tr.to_numpy(),
             eval_set=[x_val.to_numpy()],
            pretraining_ratio=0.8,
        )


        clf = TabNetClassifier(
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size":10, # how to use learning rate scheduler
                      "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='sparsemax' # This will be overwritten if using pretrain model
        )

        clf.fit(
            x_tr.to_numpy() , y_tr.to_numpy() ,
            eval_set=[(x_val.to_numpy() , y_val.to_numpy() )],
            eval_metric=['auc'],
            from_unsupervised=unsupervised_model
        )
                
        # Validation ????????? ??????
        val_preds = clf.predict(x_val.values)
        y_oof[val_idx] = val_preds
        
        # ????????? Validation ????????? ??????
        print(f"Fold {fold + 1} | AUC: {roc_auc_score(y_val, val_preds)}")
        print('-'*80)

        # score ????????? ????????? ?????? Validation ????????? ??????
        score += roc_auc_score(y_val, val_preds) / folds
        
        # ????????? ????????? ???????????? ???????????? ??????
        test_preds += clf.predict(x_test.to_numpy()) / folds
        
        # ????????? ?????? ????????? ??????
        #fi[f'fold_{fold+1}'] = clf.feature_importance()

        del x_tr, x_val, y_tr, y_val
        gc.collect()
        
    print(f"\nMean AUC = {score}") # ????????? Validation ????????? ??????
    print(f"OOF AUC = {roc_auc_score(y, y_oof)}") # Out Of Fold Validation ????????? ??????

    ####################MLFLOW###########################
    mlflow.log_param("folds", folds)
    for k,v in model_params.items():
        mlflow.log_param(k, v)

    mlflow.log_metric("Mean AUC", score)
    mlflow.log_metric("OOF AUC", roc_auc_score(y, y_oof))
    mlflow.end_run()
    ####################MLFLOW###########################

        
    # # ????????? ?????? ????????? ????????? ???????????? ?????? 
    # fi_cols = [col for col in fi.columns if 'fold_' in col]
    # fi['importance'] = fi[fi_cols].mean(axis=1)
    
    return y_oof, test_preds, fi


if __name__ == '__main__':

    # ?????? ?????? ??????
    parser = argparse.ArgumentParser()
    
    # baseline ?????? ?????? ????????? ????????? model ????????? ??????
    parser.add_argument('model', type=str, default='baseline1', help="set baseline model name among baselin1,basline2,baseline3")
    args = parser.parse_args()
    model = args.model
    print('baseline model:', model)
    
    # ????????? ?????? ??????
    data = pd.read_csv(data_dir + '/train.csv', parse_dates=['order_date'])

    # ????????? ?????? ??????
    year_month = '2011-12'
    
    if model == 'baseline3': # baseline ?????? 3
        model_params = {
            'objective': 'binary', # ?????? ??????
            'boosting_type': 'gbdt',
            'metric': 'auc', # ?????? ?????? ??????
            'feature_fraction': 0.8, # ?????? ????????? ??????
            'bagging_fraction': 0.8, # ????????? ????????? ??????
            'bagging_freq': 1,
            'n_estimators': 10000, # ?????? ??????
            'early_stopping_rounds': 100,
            'seed': SEED,
            'verbose': -1,
            'n_jobs': -1
        }
        
        # ?????? ??????????????? ??????
        train, test, y, features = feature_engineering2(data, year_month)
        
        # Cross Validation Out Of Fold??? LightGBM ?????? ?????? ??? ??????
        y_oof, test_preds, fi = make_lgb_oof_prediction(train, y, test, features, model_params=model_params)
    else:
        test_preds = baseline_no_ml(data, year_month)
    
    # ????????? ?????? ?????? ?????? ??????
    sub = pd.read_csv(data_dir + '/sample_submission.csv')
    
    # ????????? ?????? ?????? ??????
    sub['probability'] = test_preds
    
    
    os.makedirs(output_dir, exist_ok=True)
    # ?????? ?????? ??????
    sub.to_csv(os.path.join(output_dir , 'output.csv'), index=False)