import pandas as pd
import numpy as np
import os, sys, gc, random
import datetime
import dateutil.relativedelta

# Machine learning
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Custom library
from utils import seed_everything, print_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
TOTAL_THRES = 300 # 구매액 임계값
SEED = 1118 # 랜덤 시드
seed_everything(SEED) # 시드 고정

data_dir = '../input/train.csv' # os.environ['SM_CHANNEL_TRAIN']
model_dir = '../model' # os.environ['SM_MODEL_DIR']


'''
    입력인자로 받는 year_month에 대해 고객 ID별로 총 구매액이
    구매액 임계값을 넘는지 여부의 binary label을 생성하는 함수
'''
def generate_label(df, year_month, total_thres=TOTAL_THRES, print_log=False):
    df = df.copy()
    
    # year_month에 해당하는 label 데이터 생성
    df['year_month'] = df['order_date'].dt.strftime('%Y-%m')
    df.reset_index(drop=True, inplace=True)

    # year_month 이전 월의 고객 ID 추출
    cust = df[df['year_month']<year_month]['customer_id'].unique()
    # year_month에 해당하는 데이터 선택
    df = df[df['year_month']==year_month]
    
    # label 데이터프레임 생성
    label = pd.DataFrame({'customer_id':cust})
    label['year_month'] = year_month
    
    # year_month에 해당하는 고객 ID의 구매액의 합 계산
    grped = df.groupby(['customer_id','year_month'], as_index=False)[['total']].sum()
    
    # label 데이터프레임과 merge하고 구매액 임계값을 넘었는지 여부로 label 생성
    label = label.merge(grped, on=['customer_id','year_month'], how='left')
    label['total'].fillna(0.0, inplace=True)
    label['label'] = (label['total'] > total_thres).astype(int)

    # 고객 ID로 정렬
    label = label.sort_values('customer_id').reset_index(drop=True)
    if print_log: print(f'{year_month} - final label shape: {label.shape}')
    
    return label



def feature_preprocessing(train, test, features, do_imputing=True):
    x_tr = train.copy()
    x_te = test.copy()
    
    # 범주형 피처 이름을 저장할 변수
    cate_cols = []

    # 레이블 인코딩
    for f in features:
        if x_tr[f].dtype.name == 'object': # 데이터 타입이 object(str)이면 레이블 인코딩
            cate_cols.append(f)
            le = LabelEncoder()
            # train + test 데이터를 합쳐서 레이블 인코딩 함수에 fit
            le.fit(list(x_tr[f].values) + list(x_te[f].values))
            
            # train 데이터 레이블 인코딩 변환 수행
            x_tr[f] = le.transform(list(x_tr[f].values))
            
            # test 데이터 레이블 인코딩 변환 수행
            x_te[f] = le.transform(list(x_te[f].values))

    print('categorical feature:', cate_cols)

    if do_imputing:
        # 중위값으로 결측치 채우기
        #imputer = SimpleImputer(strategy='median')
        imputer = SimpleImputer(strategy='constant', fill_value=0)
        x_tr[features] = imputer.fit_transform(x_tr[features])
        x_te[features] = imputer.transform(x_te[features])

    # scaler = StandardScaler()

    # x_tr[features] = scaler.fit_transform(x_tr[features]) 
    # x_te[features] = scaler.fit_transform(x_te[features]) 

    # for f in features:
    #     x_tr[f] = pd.to_numeric(x_tr[f]) 
    #     x_te[f] = pd.to_numeric(x_te[f])

    return x_tr, x_te




def feature_engineering2(df, year_month):
    df = df.copy()
    # customer_id 기준으로 pandas group by 후 total, quantity, price 누적합 계산
    df['cumsum_total_by_cust_id'] = df.groupby(['customer_id'])['total'].cumsum()
    df['cumsum_quantity_by_cust_id'] = df.groupby(['customer_id'])['quantity'].cumsum()
    df['cumsum_price_by_cust_id'] = df.groupby(['customer_id'])['price'].cumsum()

    # product_id 기준으로 pandas group by 후 total, quantity, price 누적합 계산
    df['cumsum_total_by_prod_id'] = df.groupby(['product_id'])['total'].cumsum()
    df['cumsum_quantity_by_prod_id'] = df.groupby(['product_id'])['quantity'].cumsum()
    df['cumsum_price_by_prod_id'] = df.groupby(['product_id'])['price'].cumsum()
    
    # order_id 기준으로 pandas group by 후 total, quantity, price 누적합 계산
    df['cumsum_total_by_order_id'] = df.groupby(['order_id'])['total'].cumsum()
    df['cumsum_quantity_by_order_id'] = df.groupby(['order_id'])['quantity'].cumsum()
    df['cumsum_price_by_order_id'] = df.groupby(['order_id'])['price'].cumsum()    

    #timeseries
    df['order_ts']=df['order_date'].astype(np.int64)//1e9
    df['order_ts_diff']=df.groupby(['customer_id'])['order_ts'].diff()
    df['quantity_diff']=df.groupby(['customer_id'])['quantity'].diff()
    df['price_diff']=df.groupby(['customer_id'])['price'].diff()
    df['total_diff']=df.groupby(['customer_id'])['total'].diff()
    
    df['order_ts_custom1']=df.groupby(['order_ts'])['customer_id'].diff()
    #df['order_ts_custom2']=df.groupby(['order_ts'])['customer_id'].cumsum()
    #df['order_ts_custom3']=df.groupby(['order_id'])['order_ts'].cumsum()#에라이 제출하면 떨어짐
    #df['order_ts_custom4']=df.groupby(['order_id'])['order_ts'].diff() #성능떨어짐
    #df['order_ts_custom5']=df.groupby(['order_id'])['total'].diff() #떨어짐
    df['order_ts_custom2']=df.groupby(['customer_id'])['total'].cummax()
    df['order_ts_custom3']=df.groupby(['customer_id'])['total'].cummin()
    # f['order_ts_custom3']=df.groupby(['order_ts'])['customer_id'].diff()

    df['order_ts_month_only']=df['order_date'].astype(int).apply(lambda x: (int(str(x)[:5])-12596)/630)
    df['order_ts_diff_month_only']=df.groupby(['order_ts_month_only'])['customer_id'].diff()
    # df['order_ts_diff_2']=df.groupby(['customer_id'])['order_ts_month_only'].diff()
 

    def maxtomin(arr):
        if arr.min()>0:
            return arr.max()-arr.min()
        else:
            return arr.max()+arr.min()

    def categorize(x):
        m = x.mean()
        return 1 if m > 0.5 else 0 if m < 0.5 else np.nan

    

    # year_month 이전 월 계산
    d = datetime.datetime.strptime(year_month, "%Y-%m")
    prev_ym = d - dateutil.relativedelta.relativedelta(months=1)
    prev_ym = prev_ym.strftime('%Y-%m')
    
    # train, test 데이터 선택
    train = df[df['order_date'] < prev_ym]
    test = df[df['order_date'] < year_month]
    
    # train, test 레이블 데이터 생성
    train_label = generate_label(df, prev_ym)[['customer_id','year_month','label']]
    test_label = generate_label(df, year_month)[['customer_id','year_month','label']]
    
    # group by aggregation 함수 선언
    agg_func = ['mean','max','min','sum','count','std','skew',maxtomin,'median']
    agg_dict = {
        'quantity': agg_func,
        'price': agg_func,
        'total': agg_func,
        'cumsum_total_by_cust_id': agg_func,
        'cumsum_quantity_by_cust_id': agg_func,
        'cumsum_price_by_cust_id': agg_func,
         'cumsum_total_by_prod_id': agg_func,
        # 'cumsum_quantity_by_prod_id': agg_func,
        # 'cumsum_price_by_prod_id': agg_func,
        'cumsum_total_by_order_id': agg_func,
        'cumsum_quantity_by_order_id': agg_func,
        'cumsum_price_by_order_id': agg_func,
        'order_id': ['nunique'],
        'product_id': ['nunique'],
        'order_ts':['first','last'],
        'order_ts_diff': agg_func,
        'quantity_diff': agg_func,
        #'price_diff': agg_func, #빼니까 오름 굿
        'total_diff': agg_func,
        'order_ts_custom1': agg_func,

         #'order_ts_month_only':['first','last'],
         'order_ts_diff_month_only': agg_func,
        # 'order_ts_diff_2': agg_func,
        'order_ts_custom2': agg_func,
        'order_ts_custom3': agg_func,
        #'order_ts_custom5': agg_func,

    }
    all_train_data = pd.DataFrame()
    
    for i, tr_ym in enumerate(train_label['year_month'].unique()):
        # group by aggretation 함수로 train 데이터 피처 생성
        train_agg = train.loc[train['order_date'] < tr_ym].groupby(['customer_id']).agg(agg_dict)

        new_cols = []
        for col in agg_dict.keys():
            for stat in agg_dict[col]:
                if type(stat) is str:
                    new_cols.append(f'{col}-{stat}')
                else:
                    new_cols.append(f'{col}-mode')

        train_agg.columns = new_cols
        train_agg.reset_index(inplace = True)
        
        train_agg['year_month'] = tr_ym
        
        
        all_train_data = all_train_data.append(train_agg)
  

    all_train_data = train_label.merge(all_train_data, on=['customer_id', 'year_month'], how='left')
  
    features = all_train_data.drop(columns=['customer_id', 'label', 'year_month']).columns
    
    # group by aggretation 함수로 test 데이터 피처 생성
    test_agg = test.groupby(['customer_id']).agg(agg_dict)
    test_agg.columns = new_cols
    
    test_data = test_label.merge(test_agg, on=['customer_id'], how='left')
     # train, test 데이터 전처리
    x_tr, x_te = feature_preprocessing(all_train_data, test_data, features)
    
    print('x_tr.shape', x_tr.shape, ', x_te.shape', x_te.shape)
    
    return x_tr, x_te, all_train_data['label'], features

if __name__ == '__main__':
    
    print('data_dir', data_dir)
