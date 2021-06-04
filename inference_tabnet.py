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


TOTAL_THRES = 300 # 구매액 임계값
SEED = 42 # 랜덤 시드
seed_everything(SEED) # 시드 고정


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
    
    # 테스트 데이터 예측값을 저장할 변수
    test_preds = np.zeros(x_test.shape[0])
    
    # Out Of Fold Validation 예측 데이터를 저장할 변수
    y_oof = np.zeros(x_train.shape[0])
    
    # 폴드별 평균 Validation 스코어를 저장할 변수
    score = 0
    
    # 피처 중요도를 저장할 데이터 프레임 선언
    fi = pd.DataFrame()
    fi['feature'] = features
    
    # Stratified K Fold 선언
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(x_train, y)):
        # train index, validation index로 train 데이터를 나눔
        x_tr, x_val = x_train.loc[tr_idx, features], x_train.loc[val_idx, features]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        print(f'fold: {fold+1}, x_tr.shape: {x_tr.shape}, x_val.shape: {x_val.shape}')

        # # LightGBM 데이터셋 선언
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
                
        # Validation 데이터 예측
        val_preds = clf.predict(x_val.values)
        y_oof[val_idx] = val_preds
        
        # 폴드별 Validation 스코어 측정
        print(f"Fold {fold + 1} | AUC: {roc_auc_score(y_val, val_preds)}")
        print('-'*80)

        # score 변수에 폴드별 평균 Validation 스코어 저장
        score += roc_auc_score(y_val, val_preds) / folds
        
        # 테스트 데이터 예측하고 평균해서 저장
        test_preds += clf.predict(x_test.to_numpy()) / folds
        
        # 폴드별 피처 중요도 저장
        #fi[f'fold_{fold+1}'] = clf.feature_importance()

        del x_tr, x_val, y_tr, y_val
        gc.collect()
        
    print(f"\nMean AUC = {score}") # 폴드별 Validation 스코어 출력
    print(f"OOF AUC = {roc_auc_score(y, y_oof)}") # Out Of Fold Validation 스코어 출력

    ####################MLFLOW###########################
    mlflow.log_param("folds", folds)
    for k,v in model_params.items():
        mlflow.log_param(k, v)

    mlflow.log_metric("Mean AUC", score)
    mlflow.log_metric("OOF AUC", roc_auc_score(y, y_oof))
    mlflow.end_run()
    ####################MLFLOW###########################

        
    # # 폴드별 피처 중요도 평균값 계산해서 저장 
    # fi_cols = [col for col in fi.columns if 'fold_' in col]
    # fi['importance'] = fi[fi_cols].mean(axis=1)
    
    return y_oof, test_preds, fi


if __name__ == '__main__':

    # 인자 파서 선언
    parser = argparse.ArgumentParser()
    
    # baseline 모델 이름 인자로 받아서 model 변수에 저장
    parser.add_argument('model', type=str, default='baseline1', help="set baseline model name among baselin1,basline2,baseline3")
    args = parser.parse_args()
    model = args.model
    print('baseline model:', model)
    
    # 데이터 파일 읽기
    data = pd.read_csv(data_dir + '/train.csv', parse_dates=['order_date'])

    # 예측할 연월 설정
    year_month = '2011-12'
    
    if model == 'baseline3': # baseline 모델 3
        model_params = {
            'objective': 'binary', # 이진 분류
            'boosting_type': 'gbdt',
            'metric': 'auc', # 평가 지표 설정
            'feature_fraction': 0.8, # 피처 샘플링 비율
            'bagging_fraction': 0.8, # 데이터 샘플링 비율
            'bagging_freq': 1,
            'n_estimators': 10000, # 트리 개수
            'early_stopping_rounds': 100,
            'seed': SEED,
            'verbose': -1,
            'n_jobs': -1
        }
        
        # 피처 엔지니어링 실행
        train, test, y, features = feature_engineering2(data, year_month)
        
        # Cross Validation Out Of Fold로 LightGBM 모델 훈련 및 예측
        y_oof, test_preds, fi = make_lgb_oof_prediction(train, y, test, features, model_params=model_params)
    else:
        test_preds = baseline_no_ml(data, year_month)
    
    # 테스트 결과 제출 파일 읽기
    sub = pd.read_csv(data_dir + '/sample_submission.csv')
    
    # 테스트 예측 결과 저장
    sub['probability'] = test_preds
    
    
    os.makedirs(output_dir, exist_ok=True)
    # 제출 파일 쓰기
    sub.to_csv(os.path.join(output_dir , 'output.csv'), index=False)