import pandas as pd
import numpy as np
import random, os  
import math
import copy
import ast
import calendar
from datetime import datetime, timedelta
from collections import OrderedDict, defaultdict
from functools import partial
import operator
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet,HuberRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor, VotingRegressor, AdaBoostRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, BaggingRegressor
from sklearn.preprocessing import StandardScaler,RobustScaler,OneHotEncoder
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV,TimeSeriesSplit,cross_val_score,KFold
from sklearn.metrics import  make_scorer
from xgboost.sklearn import XGBRegressor
import xgboost as xgb
import lightgbm as ltb
import catboost as cb
import statsmodels.api as sm
from fbprophet import Prophet
import shap
shap.initjs()
from dateutil.relativedelta import relativedelta
from scipy.stats import spearmanr
from scipy.stats import linregress
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')


##Function to extract cumulative week and month feature
def get_week_all(df1):
    curr_yr = pd.to_datetime('now').isocalendar()[0] 
    df1['week_all'] = df1.week
    df1['year'] = np.where(df1.week==53,df1['year']-1,df1['year'])
    ##Need to add week_all and month_all lines everytime we move to new year
    for i in range(2019,curr_yr+2):
        df1['week_all'] = np.where(df1.year==(i),df1[df1['year']==(i-1)]['week_all'].max()+df1.week,df1.week_all)
    df1['month'] = np.where(df1.week==1,1,df1['month'])
    df1['month'] = np.where(df1.week==53,12,df1['month'])
    df1['month_all'] = df1.month
    ##Need to add week_all and month_all lines everytime we move to new year
    for i in range(2019,curr_yr+2):
        df1['month_all'] = np.where(df1.year==(i),df1[df1['year']==(i-1)]['month_all'].max()+df1.month,df1.month_all)
    df1['week_all'] = df1['week_all'].astype(int)
    df1['month_all'] = df1['month_all'].astype(int)
    return df1
 
def data_wrangling_df_final(df, max_date, min_date):
    print("\nLatest Date :",df['Date'].max())

    df['week_start'] = df['Date'].dt.to_period('W-SUN').apply(lambda r: r.start_time)
    df = df.groupby(['Stock','week_start'])[['Volume', 'Open', 'Adj Close', 'Low', 'High', 'Close']].sum().reset_index()
    df.rename(columns={'week_start':'Date'},inplace=True)

#     #Changing the latest Date as per requirement 
#     df = df[df['Date']<=pd.to_datetime(max_date)]
#     df = df[df['Date']>=pd.to_datetime(min_date)]
#     print("\nLast date after the change as per requirement :",df['Date'].min())
#     print("\nLatest date after the change as per requirement :",df['Date'].max())

    df_new = pd.DataFrame(columns=df.columns)
    for stocks in df.Stock.unique():
        df_iter = df[df['Stock']==stocks].copy()
        df_iter = pd.DataFrame(pd.date_range(start='2018-01-01', end=df.Date.max(), freq='W-MON'),columns=['Date']).merge(df_iter,on='Date',how='left')
        df_iter['Stock'] = df_iter['Stock'].ffill()
        df_iter['Stock'] = df_iter['Stock'].bfill()
        df_iter[['Volume','Open','Adj Close','Low','High','Close']] = df_iter[['Volume','Open','Adj Close','Low','High','Close']].fillna(0)
        df_new = df_new.append(df_iter)
    df = df_new.copy()
    del df_new
    
    ##Extract week, Month and Year from Date column
    df['week'] = df['Date'].dt.week
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    
    ##Extracting cumulative week and month feature
    df = get_week_all(df)
    return df

def data_wrangling_df(df, max_date, min_date):
    print("\nLatest Date :",df['Date'].max())

    df['week_start'] = df['Date'].dt.to_period('W-SUN').apply(lambda r: r.start_time)
    df = df.groupby(['Stock','week_start'])[['Volume', 'Open', 'Adj Close', 'Low', 'High', 'Close']].mean().reset_index()
    df.rename(columns={'week_start':'Date'},inplace=True)

    #Changing the latest Date as per requirement 
    df = df[df['Date']<=max_date]
    df = df[df['Date']>=min_date]
    print("\nLast date after the change as per requirement :",df['Date'].min())
    print("\nLatest date after the change as per requirement :",df['Date'].max())

    df_new = pd.DataFrame(columns=df.columns)
    for stocks in df.Stock.unique():
        df_iter = df[df['Stock']==stocks].copy()
        df_iter = pd.DataFrame(pd.date_range(start='2018-01-01', end=df.Date.max(), freq='W-MON'),columns=['Date']).merge(df_iter,on='Date',how='left')
        df_iter['Stock'] = df_iter['Stock'].ffill()
        df_iter['Stock'] = df_iter['Stock'].bfill()
        df_iter[['Volume','Open','Adj Close','Low','High','Close']] = df_iter[['Volume','Open','Adj Close','Low','High','Close']].fillna(0)
        df_new = df_new.append(df_iter)
    df = df_new.copy()
    del df_new
    
    ##Extract week, Month and Year from Date column
    df['week'] = df['Date'].dt.week
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    
    ##Extracting cumulative week and month feature
    df = get_week_all(df)
    return df


def day_contribution(df):
    current_week = df.week_all.max()
    last_5_week = [i for i in range(current_week-4,current_week+1)]
    print("\nCurrent week is: {} (i.e. {})".format(current_week,pd.to_datetime(df['Date'].max(), format="%Y-%m-%d", errors='ignore')))
    current_week_dict = {}
    for idx, dates in enumerate(df['Date'].unique()):
        current_week_dict[idx] = dates
    print("\nLast 5 weeks are: {} (i.e. from {} to {})".format(last_5_week,pd.to_datetime(current_week_dict[last_5_week[0]-1], format="%Y-%m-%d", errors='ignore'),pd.to_datetime(current_week_dict[last_5_week[4]-1], format="%Y-%m-%d", errors='ignore')))
    last_5_week = df[df['week_all'].isin(last_5_week)] 
    del current_week_dict
    return current_week, last_5_week

def get_train_val_params(data1_in,last_forecast_week_index,N_RUNS=5,train_decrease_window = 1):
    train_val_list = []
    for runid in range(0,N_RUNS):
        train_start = data1_in.index[0]
        train_end = data1_in.index[last_forecast_week_index - 6 - runid*train_decrease_window]
        val_start = data1_in.index[last_forecast_week_index - 5 - runid*train_decrease_window]
        val_end = data1_in.index[last_forecast_week_index - 1 - runid*train_decrease_window]
        test_start = data1_in.index[last_forecast_week_index - runid*train_decrease_window]
        test_end = data1_in.index[last_forecast_week_index + 4 - runid*train_decrease_window]
        train_val_params = {'train_start' : train_start,
                         'train_end'   : train_end,
                         'val_start'   : val_start,
                         'val_end'     : val_end,
                         'test_start'  : test_start,
                         'test_end'    : test_end
                         }
        train_val_list.append(train_val_params)
    return train_val_list

def get_train_val_params_final_iter(data_check,train_val_params_list,N_RUNS=5,train_decrease_window = 1):
    train_val_list = []
    for runid in range(0,N_RUNS):
        train_start = list(data_check['date'].sort_values())[0]
        train_end = list(data_check['date'].sort_values())[-6-runid*train_decrease_window] 
        val_start = list(data_check['date'].sort_values())[-5-runid*train_decrease_window]
        val_end = list(data_check['date'].sort_values())[-1-runid*train_decrease_window]
        test_start = train_val_params_list[runid]['test_start']
        test_end = train_val_params_list[runid]['test_end']
        train_val_params = {'train_start' : train_start,
                         'train_end'   : train_end,
                         'val_start'   : val_start,
                         'val_end'     : val_end,
                         'test_start'  : test_start,
                         'test_end'    : test_end
                         }
        train_val_list.append(train_val_params)
    return train_val_list

def get_train_val_params_forecast_run(data_check,train_val_params_list):
    train_start = list(data_check['date'].sort_values())[0]
    train_end = list(data_check['date'].sort_values())[-6] 
    val_start = list(data_check['date'].sort_values())[-5]
    val_end = list(data_check['date'].sort_values())[-1]
    test_start = train_val_params_list[0]['test_start']
    test_end = train_val_params_list[0]['test_end']
    train_val_params = {'train_start' : train_start,
                     'train_end'   : train_end,
                     'val_start'   : val_start,
                     'val_end'     : val_end,
                     'test_start'  : test_start,
                     'test_end'    : test_end
                     }
    train_val_params_list = [train_val_params]
    return train_val_params_list

def last_week_shift(df,current_week):
    last_1w_df = df[df['week_all']==current_week]
    ##Add the missing dates and replace with 0
    data_period = pd.DataFrame(pd.date_range(start=last_1w_df['Date'].max() + timedelta(7), periods=9, freq='W-MON'),columns=['Date'])
    plant_grp_comb = df[['Stock']].drop_duplicates()
    data_period['key'] = 1
    plant_grp_comb['key'] = 1
    plant_grp_comb_data = pd.merge(plant_grp_comb, data_period, on ='key').drop("key", 1) 
    plant_grp_comb_data['year'] = plant_grp_comb_data['Date'].dt.year
    plant_grp_comb_data['week'] = plant_grp_comb_data['Date'].dt.week
    plant_grp_comb_data['month']= plant_grp_comb_data['Date'].dt.month
    df = df.append(plant_grp_comb_data)
    df = get_week_all(df)
    
    new_current_week = df.week_all.max()
    new_last_5_week = [i for i in range(new_current_week-4,new_current_week+1)]
    new_current_week_dict = {}
    for idx, dates in enumerate(df['Date'].unique()):
        new_current_week_dict[idx] = dates
    print("\n5 weeks to be forecasted in the final iteration are: {} (i.e. from {} to {})".format(new_last_5_week,pd.to_datetime(new_current_week_dict[new_last_5_week[0]-1], format="%Y-%m-%d", errors='ignore'),pd.to_datetime(new_current_week_dict[new_last_5_week[4]-1], format="%Y-%m-%d", errors='ignore')))
    del new_current_week_dict
    return df, new_last_5_week

def last_week_shift_final_iter(df,current_week):
    last_1w_df = df[df['week_all']==current_week]
    ##Add the missing dates and replace with 0
    data_period = pd.DataFrame(pd.date_range(start=last_1w_df['Date'].max() + timedelta(7), periods=5, freq='W-MON'),columns=['Date'])
    plant_grp_comb = df[['Stock']].drop_duplicates()
    data_period['key'] = 1
    plant_grp_comb['key'] = 1
    plant_grp_comb_data = pd.merge(plant_grp_comb, data_period, on ='key').drop("key", 1) 
    plant_grp_comb_data['year'] = plant_grp_comb_data['Date'].dt.year
    plant_grp_comb_data['week'] = plant_grp_comb_data['Date'].dt.week
    plant_grp_comb_data['month']= plant_grp_comb_data['Date'].dt.month
    df = df.append(plant_grp_comb_data)
    df = get_week_all(df)
    return df

#Function to get the Week Number of each month
def get_week_of_month(year, month, day):
    x = np.array(calendar.monthcalendar(year, month))
    week_of_month = np.where(x==day)[0][0] + 1
    return week_of_month

#Function to get the Week Number of each month on the dataframe
def get_week_of_month_df(df, get_week_of_month):
    df['day'] = df['Date'].dt.day
    df['Week_of_month']=list(map(get_week_of_month,df['year'],df['month'],df['day']))
    df.drop('day', axis=1, inplace=True)
    return df

##Feature to capture the peak stock across week of the month
def peak_week_of_month(df):
    df['max_Open'] = df.groupby(['Stock','year','month'])['Open'].transform('max')
    df['peak_week_flag'] = np.where((df['max_Open']==df['Open']) & (df['max_Open']>0),df['Week_of_month'],None)
    df['peak_week_flag'] = np.where(df['Week_of_month']==df['peak_week_flag'].value_counts().keys().tolist()[0],1,0)
    df.drop('max_Open', axis=1, inplace=True)
    return df

def get_forecast_dates_df(df):
    ##Create empty dataset with all weeks combinations
    week_dict = {}
    for idx, dates in enumerate(df['Date'].unique()):
        week_dict[idx+1] = pd.to_datetime(dates, format="%Y-%m-%d", errors='ignore')
    df_dates = pd.DataFrame(columns = {},index=list(week_dict.values()))
    return df_dates

def create_derived_features(target,data_c0,lag_vaiables,rolling_mean,trend):
    
    data = data_c0.loc[:,[target]]
    for lag in lag_vaiables:
            data[target+'_lag_'+str(lag)+'_Week']=data[target].shift(lag).fillna(0)

    for lag in rolling_mean:
        data[target+'_Rolling_'+str(lag)+'_week'+'_mean']=data[target].rolling(window=lag).mean().shift(1)
        data[target+'_Rolling_'+str(lag)+'_week'+'_mean']=data[target].rolling(window=lag).mean().shift(1)

    a,b = data.iloc[0][target],np.mean(data.iloc[0:2][target])
    data.iloc[1:3][target+"_Rolling_3_week_mean"] = a,b
    data.iloc[1:2][target+"_Rolling_2_week_mean"] = a

    for lag in trend:
        data=pd.merge(data.reset_index(drop=True),
               data[[target]].shift(1).rolling(window=lag).apply(lambda x:find_normalized_slope(x)).reset_index(drop=True).add_suffix('_normalized_slope_'+str(lag)),
               left_index=True, right_index=True).round(2)
    return data

def find_normalized_slope(y):
    y=pd.Series(y)
    y=(y/max(abs(y))).fillna(0)
    x=pd.Series(range(0,len(y)))/(len(y)-1)
    slope=linregress(x, y).slope
    return(slope)

def mape(y_act,y_pred):
    return np.round(np.mean(np.abs((y_act-y_pred)/(y_act+0.000001)))*100,2)

def wape(actual,pred):
    return np.mean(np.abs((actual - pred)))/ (np.mean(np.abs(actual))+0.000000001) * 100

def scale_fit(df,cols):
    scalers={}
    for col in cols:
        sc=StandardScaler()
        sc.fit(df[col].values.reshape(-1,1))
        scalers[col]=sc
    return scalers

def scale_transform(df,cols,scalers):
    df1=df.copy()
    for col in cols:
        sc=scalers[col]
        df1[col]=sc.transform(df1[col].values.reshape(-1,1))
    return df1

def create_ohe_features(df,cat_cols,cat_col_labels):
    ohes={}
    cat_to_ohe_map = {}
    for col in cat_cols:
        print(col)
        if col in ohes:
            ohe = ohes[col]
            X=ohe.transform(df[col].values.reshape(-1,1)).toarray()
        else:
            ohe=OneHotEncoder([np.array(cat_col_labels[col])],handle_unknown='error')
            X=ohe.fit_transform(df[col].values.reshape(-1,1)).toarray()
            ohes[col] = ohe
        dfOneHot=pd.DataFrame(X,columns=[col+'_'+str(i) for i in range(X.shape[1])])
        df=pd.merge(df,dfOneHot,left_index=True,right_index=True)
        cat_to_ohe_map[col] = list(dfOneHot.columns)
        
    return df,ohes,cat_to_ohe_map


def forecast_target(model,df,y_train,target,feature_cols):
    lag_numbers,med_numbers,trend_numbers = [],[],[]
    
    lag_cols = [col for col in feature_cols if 'target_lag_' in col]
    if len(lag_cols)>0:
        lag_numbers = [-1*int(col.split('_')[2]) for col in lag_cols]
    
    med_cols = [col for col in feature_cols if 'target_Rolling_' in col]
    if len(med_cols)>0:
        med_numbers = [-1*int(col.split('_')[2]) for col in med_cols]
    
    trend_cols = [col for col in feature_cols if str(target)+'_normalized_slope_' in col]
    if len(trend_cols)>0:
        trend_numbers = [-1*int(col[-1]) for col in trend_cols]
    
    df.drop(columns=med_cols+lag_cols+trend_cols,inplace=True)
    
    outputs = y_train.copy()
    for row in range(df.shape[0]):
        if len(lag_cols)>0:
            lag_vals = [outputs[i] for i in lag_numbers] 
            for counter, col in enumerate(lag_cols):
                df.loc[row,col]=lag_vals[counter]

        if len(med_cols)>0:
            mean_vals = [np.mean(outputs[idx:],axis=0) for idx in med_numbers]
            for counter, col in enumerate(med_cols):
                df.loc[row,col]=mean_vals[counter]

        if len(trend_cols)>0:
            trend_vals = []
            for lag in trend_numbers:
                slope_i = find_normalized_slope(outputs[-lag:])
                trend_vals.append(slope_i)
            for counter, col in enumerate(trend_cols):
                df.loc[row,col]=trend_vals[counter]

        output_i= model.predict(df.loc[row,feature_cols].values.reshape(1,-1))[0]
        outputs.append(output_i)
    outputs = outputs[len(y_train):]
    return df,outputs,None  
  

def forecast_test(model,X_val,y_train,target,feature_cols,lag_flag=None,ma_flag=None,trend_flag=None):
    df=X_val.copy()
    df=df.reset_index()
    df=df.loc[:,["date"]+feature_cols]
    return forecast_target(model,df,y_train,target,feature_cols)
    
def predict_ewm(data,test_weeks,target,span=None,alpha=None):
    if span==None and alpha==None:
        print("Pass either span or alpha value for ewm")
        return
    df = data.copy()
    if span!=None:
        pred_col_name="ewm_"+str(span)+"m_"+str(target)
    else:
        pred_col_name="ewm_"+str(alpha)+"_"+str(target)
        
    target_ewm="ewm_"+str(target)+"_sim"
    df[target_ewm]=df[target]
    
    random_error = np.random.randint(-10,10,len(test_weeks))*np.random.randn(len(test_weeks))
    count=0
    
    if span:
        for i in range(df.shape[0]):    
            if i ==len(df)-1:
                estimate_i=df[target_ewm].ewm(span=span,adjust=False,min_periods=0).mean().shift(1)[i]
                df.loc[i,pred_col_name]= estimate_i
                df.loc[i,target_ewm] = estimate_i*(1+(random_error[count])/100)
                return df
            if not np.isnan(df.loc[i,target_ewm]):
                estimate_i,estimate_i1=df[target_ewm].ewm(span=span,adjust=False,min_periods=0).mean().shift(1)[i:i+2]
                df.loc[i,pred_col_name]= estimate_i   
            else:
                df.loc[i,target_ewm]=estimate_i1*(1+(random_error[count])/100)
                estimate_i,estimate_i1=df[target_ewm].ewm(span=span,adjust=False,min_periods=0).mean().shift(1)[i:i+2]
                df.loc[i,pred_col_name]=estimate_i  
                count+=1
        return df
    else:
        for i in range(df.shape[0]):    
            if i ==len(df)-1:
                #print(i)
                estimate_i=df[target_ewm].ewm(alpha=alpha,adjust=False,min_periods=3).mean().shift(1)[i]
                df.loc[i,pred_col_name]= estimate_i
                df.loc[i,target_ewm] = estimate_i1*(1+(random_error[count])/100)
                return df
            if not np.isnan(df.loc[i,target_ewm]):
                estimate_i,estimate_i1=df[target_ewm].ewm(alpha=alpha,adjust=False,min_periods=3).mean().shift(1)[i:i+2]
                df.loc[i,pred_col_name]= estimate_i   
            else:
                df.loc[i,target_ewm]=estimate_i1*(1+(random_error[count])/100)
                estimate_i,estimate_i1=df[target_ewm].ewm(alpha=alpha,adjust=False,min_periods=3).mean().shift(1)[i:i+2]
                df.loc[i,pred_col_name]=estimate_i    
                count+=1
        return df
    
def predict_wma(data,test_weeks,target,window=None,weights=None):
    if window==None and alpha==None:
        print("Pass window size for weighted moving averages")
        return
    df = data.copy()
    
    pred_col_name="wma_"+str(window)+"m_"+str(target)
    target_wma="wma_"+str(target)+"_sim"
    df[target_wma]=df[target]
    for i in range(df.shape[0]):    
        if i ==len(df)-1:
            estimate_i=df[target_wma].rolling(window=window).apply(wma(np.array(weights))).shift(1)[i]
            df.loc[i,pred_col_name]= estimate_i
            df.loc[i,target_wma] = estimate_i
            return df
        if not np.isnan(df.loc[i,target_wma]):
            estimate_i,estimate_i1=df[target_wma].rolling(window=window).apply(wma(np.array(weights))).shift(1)[i:i+2]
            df.loc[i,pred_col_name]= estimate_i
        else:
            df.loc[i,target_wma]=estimate_i1
            estimate_i,estimate_i1=df[target_wma].rolling(window=window).apply(wma(np.array(weights))).shift(1)[i:i+2]
            df.loc[i,pred_col_name]=estimate_i
    return df

def wma(weights):
    def formula(x):
        return (weights*x).mean()
    return formula

def recursive_sma(df,test_period_list,window_num=4):
    df['Open_copy'] = df['Open'].copy() 
    for i in range(len(test_period_list)):    
        df['rec_SMA_'+str(window_num)] = df.loc[df['Date']<=pd.DataFrame(test_period_list).sort_values(by=0).iloc[i,0],'Open_copy'].shift(periods=1).rolling(window=window_num).mean()
        df.loc[df['Date']==pd.DataFrame(test_period_list).sort_values(by=0).iloc[i,0],'Open_copy'] = df.loc[df['Date']==pd.DataFrame(test_period_list).sort_values(by=0).iloc[i,0],'rec_SMA_'+str(window_num)]
    return df.drop(columns={'Open_copy'})

def recursive_ema(df,test_period_list,window_num=4):
    df['Open_copy'] = df['Open'].copy() 
    for i in range(len(test_period_list)):    
        df['rec_EMA_'+str(window_num)] = df.loc[df['Date']<=pd.DataFrame(test_period_list).sort_values(by=0).iloc[i,0],'Open_copy'].shift(periods=1).ewm(span=window_num,adjust=False).mean()
        df.loc[df['Date']==pd.DataFrame(test_period_list).sort_values(by=0).iloc[i,0],'Open_copy'] = df.loc[df['Date']==pd.DataFrame(test_period_list).sort_values(by=0).iloc[i,0],'rec_EMA_'+str(window_num)]
    return df.drop(columns={'Open_copy'})

def find_normalized_slope(y):
    y=pd.Series(y)
    y=(y/max(abs(y))).fillna(0)
    x=pd.Series(range(0,len(y)))/(len(y)-1)
    slope=linregress(x, y).slope
    return(slope)

def rfe_scaler(X_train,y_train):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    
    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)

    X_train -= mean
    X_train = X_train/std
    y_train -= y_mean
    y_train = y_train/y_std
    return X_train,y_train

def create_features_target(target,data_c0,lag_vaiables,rolling_mean,trend):
    
    for lag in lag_vaiables:
            data_c0['target_lag_'+str(lag)+'_Week']=data_c0[target].shift(lag).fillna(0)
    
    for lag in rolling_mean:
        data_c0['target_Rolling_'+str(lag)+'_week'+'_mean']=data_c0[target].rolling(window=lag).mean().shift(1)
        data_c0['target_Rolling_'+str(lag)+'_week'+'_mean']=data_c0[target].rolling(window=lag).mean().shift(1)
      
    a,b = data_c0.iloc[0][target],np.mean(data_c0.iloc[0:2][target])
    data_c0.iloc[1:3]["target_Rolling_3_week_mean"] = a,b
    data_c0.iloc[1:2]["target_Rolling_2_week_mean"] = a
    
    for lag in trend:  
        data_c0=pd.merge(data_c0.reset_index(drop=True),
               data_c0[[target]].shift(1).rolling(window=lag).apply(lambda x:find_normalized_slope(x)).reset_index(drop=True).add_suffix('_normalized_slope_'+str(lag)).rename(columns={target+'_normalized_slope_'+str(lag):'target_normalized_slope_'+str(lag)}),
               left_index=True, right_index=True).round(2)
    
    return data_c0

def rfecv(X,y):
    # Create the RFE object and compute a cross-validated score.
    model = RandomForestRegressor(random_state=42,max_depth=7, max_features=0.25, n_estimators=100, n_jobs=-1)
    
    rfecv = RFECV(estimator=model, step=1, min_features_to_select=3, scoring='neg_mean_squared_error', verbose=False)
    rfecv.fit(X.replace([-np.inf,np.inf,np.NaN],0), y.replace([-np.inf,np.inf,np.NaN],0))
    print("Optimal number of features : %d" % rfecv.n_features_, '\n')

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    features = pd.DataFrame(rfecv.support_,index = X.columns, columns=['keep'])
    return(features)

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
def get_pipeline(model):
    pipeline = Pipeline([('standard_scaler', StandardScaler()), ('model', model)])
    return pipeline

def get_models(params):
    param_grid_xgb = {'model__max_depth': [2, 3, 5, 7, 10],'model__n_estimators': [10, 20, 30]}
    tscv = TimeSeriesSplit(n_splits=4)
    
    models=dict()
    models['ridge']=Ridge()
    models['rf']=RandomForestRegressor(**params['rf'])
    if 'gcv_xgb' in params:
        models['gcv_xgb']=xgb.XGBRegressor(random_state=0, **params['gcv_xgb'])
    else:
        models['gcv_xgb']=GridSearchCV(get_pipeline(xgb.XGBRegressor(random_state=0)), param_grid_xgb, cv=4, n_jobs=-1,scoring='neg_mean_absolute_percentage_error')
    models['ada'] = AdaBoostRegressor(**params['ada'])
    return models

def get_dataset(data,train_val_params,test_flag=False):
    ds = {}
    train_start,train_end,val_start,val_end,test_start,test_end = train_val_params.values()
    ds['train']=data[train_start:train_end]
    val_flag = val_start > train_end
    if val_flag:
        ds['val']=data[val_start:val_end]
    else:
        ds['val']=pd.DataFrame(columns=data.columns)

    if test_flag:
        ds['test'] = data[test_start:]
    else:
        ds['test'] =pd.DataFrame(columns=data.columns)

    return ds
 
def cv_job(model_params,ds,features,target):
    pipelines = get_models(model_params)

    results,names = [],[]
    cv_result_dict = {}
    mape_scorer = make_scorer(mape)
    X_train,Y_train = pd.concat((ds['train'],ds['val']),axis=0)[features],pd.concat((ds['train'],ds['val']),axis=0)[target]
    for name, model in pipelines.items():
        kfold = KFold(n_splits=4)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=mape_scorer)
        results.append(cv_results)
        names.append(name)
        cv_result_dict[name]={'mean':np.round(cv_results.mean(),2),'stdev':np.round(cv_results.std(),2)}
    cv_result_dict = dict(sorted(cv_result_dict.items(), key=lambda item: item[1]['mean']))
    print("Cross validation summary:",cv_result_dict,"\n")
    return cv_result_dict

def training_job(model_params,ds,features,target):
    pipelines = get_models(model_params)
    
    train,val=ds['train'],ds['val']
    val_flag = ds['val'].index.min()>ds['train'].index.max()
    models_dict,summary_dict = {},{}
    
    if val_flag:
        output_df=pd.DataFrame({'date':np.concatenate([ds['train'].index,ds['val'].index])})    
    else:
        output_df=pd.DataFrame({'date':np.concatenate([ds['train'].index])})

    for name,model in pipelines.items():
        X_train, X_val, y_train, y_val = train[features],val[features],train[target],val[target]
       
        pipe = get_pipeline(model)
        pipe.fit(X_train,y_train)
 
        train_pred =pipe.predict(X_train) 
       
        if val_flag:
            val_pred = pipe.predict(X_val)
        else:
            val_pred=[]
        
        if 'gcv' in name:
            best_params = {key.split("model__")[1]:value for key,value in pipe['model'].best_params_.items()}
            model_params[name]=best_params
        models_dict[name]=pipe
        summary_dict[name]={"train":mape(y_train,train_pred),'val':mape(y_val,val_pred)}
        output_df[name]=(np.concatenate([train_pred,val_pred],axis=0))
    summary_dict = dict(sorted(summary_dict.items(), key=lambda item: item[1]['val']))
    print("Training with 5 weeks hold out summary:",summary_dict,"\n")
    return output_df,summary_dict,model_params,models_dict

#model, hyperparameters and features

def inference_job(model_params,ds,features,target):  
    pipelines = get_models(model_params)
  
    ds['train'] = pd.concat((ds['train'],ds['val']),axis=0)
    train,test=ds['train'],ds['test']
    models_dict,summary_dict = {},{}
    
    output_df=pd.DataFrame({'date':np.concatenate([ds['train'].index,ds['test'].index])})    
   
    for name,model in pipelines.items():
        X_train, X_test, y_train, y_test = train[features],test[features],train[target],test[target]
       
        pipe = get_pipeline(model)
        pipe.fit(X_train,y_train)
        train_pred =pipe.predict(X_train) 
       
        lag_flag = any([True if "target_lag_" in col else False for col in features])
        ma_flag = any([True if "target_Rolling_" in col else False for col in features])
        trend_flag = any([True if str(target)+"_normalized_slope_" in col else False for col in features])
        
        kwargs = {'model':pipe,'target':target,'feature_cols':features,'lag_flag':lag_flag,'ma_flag':ma_flag,'trend_flag':trend_flag}
        pred_func  = partial(forecast_test,**kwargs)
        if ds['test'].shape[0]>0:
            y_input = y_train.values.tolist()
            if not (lag_flag or ma_flag or trend_flag):
                test_pred = pipe.predict(test[features])
            else:
                _,test_pred,_ = pred_func(X_val=test[features],y_train=y_input)
        else:
            test_pred=[]
        models_dict[name]=pipe
        summary_dict[name]={"train":mape(y_train,train_pred)}
        output_df[name]=(np.concatenate([train_pred,test_pred],axis=0))

    summary_dict = dict(sorted(summary_dict.items(), key=lambda item: item[1]['train']))
    print("Full Training summary:",summary_dict,"\n")
    return output_df,summary_dict

def _softmax(x): 
    e_x = np.exp(x - np.max(x)) 
    return e_x / e_x.sum(axis=0) 

def weighted_error_train(eval_train,train_params_cur):
    train_error_weight = train_params_cur['train_error_weight']
    val_error_weight = train_params_cur['val_error_weight']
    vals = [np.round(eval_train[model]['train']*train_error_weight+eval_train[model]['val']*val_error_weight,2) for model in eval_train.keys()]
    error_dict = dict(zip(list(eval_train.keys()),vals))
    error_dict = dict(sorted(error_dict.items(),key=lambda item:item[1]))
    return error_dict
    

def ensemble_output(output_df,eval_train,train_params_cur):
    
    error_dict = weighted_error_train(eval_train,train_params_cur)
    print("Train summary based on weighted errors",error_dict)
    top_k_models = list(error_dict.keys())[0:3]
    val_errors = [value for key,value in error_dict.items() if key in top_k_models]
    val_accuracy = [100-val for val in val_errors]
    weights = _softmax(val_accuracy)
    output_df['ensemble_avg'] = (output_df[top_k_models]*weights).sum(axis=1)
    return output_df,error_dict

def plot_trend(data):
    fig = plt.figure(figsize=(20, 26))
    i=1
    for account,df in data.items():
        plt.subplot(6,2,i)
        i+=1
        sim_cols = [col for col in df.columns if 'sim' in col]
        if len(sim_cols)>0:
            df.drop(columns=sim_cols,inplace=True)
        plt.plot(df.iloc[:,:3])
        plt.legend(df.columns.tolist()[:3])
        
def save_to_excel(data,path,output_filename):
    with pd.ExcelWriter(path/output_filename) as writer:  
        for key,df in data.items():
            df.to_excel(writer, sheet_name=key)
            
def find_corr(X,y,corr_threshold):
    X[y.name]=y
    correlations=X.corr()
    priority_list=abs(correlations[y.name]).reset_index().rename(columns={'index':'column'}).sort_values(by=y.name,ascending=False).reset_index(drop=True).reset_index().rename(columns={'index':'rank'})
    correlations.drop(index=[y.name],columns=[y.name],inplace=True)
    
    correlated_pairs=pd.DataFrame(index=correlations.index)
    correlated_pairs['correlation_list']=''
    correlated_pairs['correlation_list'] = correlated_pairs.apply(lambda x: [], axis=1)
    
    # Iterate through the columns
    for col in correlations:
        # Find correlations above the threshold
        above_threshold_vars = []
        above_threshold_vars = [x for x in list(correlations.index[abs(correlations[col]) > corr_threshold]) if x != col]
        correlated_pairs.loc[col,'correlation_list']=above_threshold_vars
    del above_threshold_vars, col
    
    correlated_pairs=correlated_pairs[correlated_pairs['correlation_list'].apply(lambda x: len(x))>0]
    
    drop_list=[]
    for col in correlated_pairs.index:
        current_correlated_pair=correlated_pairs.loc[col][0]
        current_correlated_pair.append(col)
        current_correlated_pair=pd.DataFrame(current_correlated_pair,columns=['column'])
        current_correlated_pair=pd.merge(current_correlated_pair,priority_list,on='column')
        current_drop_list=current_correlated_pair[current_correlated_pair['rank']>current_correlated_pair['rank'].min()].column.tolist()
        drop_list=drop_list+current_drop_list
        del current_correlated_pair,current_drop_list
    drop_list=pd.Series(drop_list).drop_duplicates().tolist()
    
    return(drop_list)

def feature_selection(data,target,drop_columns=[],target_base_features=[]):
    
    drop_features = [col for col in data.columns if any(sub in col for sub in drop_columns)] 
    
    X_train = data.replace([-np.inf,np.inf,np.NaN],0).iloc[3:-3,5:].drop(columns=drop_features)
    X_train = X_train.drop(columns=target_base_features)
    y_train = data.replace([-np.inf,np.inf,np.NaN],0).iloc[3:-3,4].copy()
    
    _features=list(X_train.columns) 
    _features_drop=find_corr(X_train,y_train,corr_threshold=0.9)  

    X_train=X_train.drop(columns=_features_drop)
    X_train = X_train.drop(columns=[target])
      
    return X_train,y_train

def shapley_feature_selection(default_model, X_train, y_train, drop_columns=[]):
    
    drop_features = [col for col in X_train.columns if any(sub in col for sub in drop_columns)] 
    
    X_train = X_train.drop(columns=drop_features)
    default_model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(default_model, enable_categorical = True)
    shap_values = explainer.shap_values(X_train,check_additivity=False)
    shap_sum = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame([X_train.columns.tolist(), shap_sum.tolist()]).T
    importance_df.columns = ['column_name', 'shap_importance']
    importance_df = importance_df.sort_values('shap_importance', ascending=False).reset_index(drop=True)
    return importance_df

def others_feature_selection(others_df,cognos_data,target, drop_columns=[]):
    X_train = others_df.loc[others_df['date'].isin(cognos_data['date'].tolist())]
    X_train = X_train[list(set(X_train.columns.tolist()) - set(['country','date']))]
    y_train = cognos_data[target]

    default_model = XGBRegressor(n_estimators=30, max_depth=4, random_state=0)
    
    #Selecting top K features based on SHAP importance
    importance_df = shapley_feature_selection(default_model, X_train, y_train, drop_columns=[])
    
    select_top_k = 3
    filtered_features = importance_df['column_name'][:select_top_k].tolist()
    return filtered_features



def process(rec_data,empties_features_data,features,target,
            drop_rfe_shapley_columns,base_features,
            train_val_params,model_params,error_weights,
            is_rfe_exec=False,shap_feature_selection=True,top_k_features=5,others_data=None):
        
    print("Starting process for ",target,"\n")
    lag_variables = [1,2,3,4,8,12]
    rolling_mean = [4,8]
    trend  = [4,8]
    acc_model_params = model_params.copy()
    
    cognos_data = rec_data.loc[:,['date','year','month','quarter',target]]
    cognos_data = create_features_target(target,cognos_data,lag_variables,rolling_mean,trend)
    
    if not others_data is None:
        other_imp_featuers = others_feature_selection(others_data,cognos_data,target,drop_columns=drop_rfe_shapley_columns)
        empties_features_data = pd.merge(empties_features_data,others_data[other_imp_featuers+['date']],how='left',on='date')
    
    cognos_data = pd.merge(cognos_data,empties_features_data,how='left',on='date')
    
    #Removing zero cases during forecasting 
    cognos_data = cognos_data[(cognos_data[target]>0.0001)].reset_index(drop=True)
    

    #Setting a default model to be used for Shapley feature selection
    default_model = XGBRegressor(n_estimators=1000, max_depth=10, learning_rate=0.001, random_state=0)

    if (shap_feature_selection==True)&(is_rfe_exec==False):
        X_train,y_train = feature_selection(cognos_data,target,drop_columns=drop_rfe_shapley_columns,target_base_features=base_features)
        features = list(X_train)

        #Selecting top K features based on SHAP importance
        importance_df = shapley_feature_selection(default_model, X_train, y_train,drop_columns=drop_rfe_shapley_columns)
        importance_df_copy = importance_df.copy()
        if len((importance_df[importance_df['shap_importance']>=importance_df['shap_importance'].mean()])['column_name'].to_list())<top_k_features:
            print('\nFeatures in ranked order are:', importance_df['column_name'].to_list())
            importance_df = importance_df[importance_df['shap_importance']>=importance_df['shap_importance'].mean()]
            features = importance_df['column_name'].to_list()
            #Force-fitting date features
            if ((not 'peak_week_flag' in features)&(not 'week' in features)&(not 'week_all' in features)&(not 'year' in features)&(not 'month' in features)&(not 'month_all' in features)&(not 'Week_of_month' in features)):
                features.append(importance_df_copy[importance_df_copy['column_name'].isin(['week', 'year', 'month', 'week_all', 'month_all', 'Week_of_month','peak_week_flag'])]['column_name'].to_list()[0])

            print('\nFinal top features remaining after Shapley feature selection are:',features)
            final_selected_features = features.copy()
        else:
            print('\nFeatures in ranked order are:', importance_df['column_name'].to_list())
            importance_df = importance_df[:top_k_features]
            features = importance_df['column_name'].to_list()
            #Force-fitting date features
            if ((not 'peak_week_flag' in features)&(not 'week' in features)&(not 'week_all' in features)&(not 'year' in features)&(not 'month' in features)&(not 'month_all' in features)&(not 'Week_of_month' in features)):
                features.append(importance_df_copy[importance_df_copy['column_name'].isin(['week', 'year', 'month', 'week_all', 'month_all', 'Week_of_month','peak_week_flag'])]['column_name'].to_list()[0])

            print('\nFinal top features remaining after Shapley feature selection are:',features)
            final_selected_features = features.copy()

        features.append(target)
        cognos_data = cognos_data[features+['date']]

    elif (shap_feature_selection==True)&(is_rfe_exec==True):
        X_train,y_train = feature_selection(cognos_data,target,drop_columns=drop_rfe_shapley_columns,target_base_features=base_features)
        X_train,y_train = rfe_scaler(X_train,y_train)
        _rfecv_features=rfecv(X_train,y_train)

        features=_rfecv_features[_rfecv_features.keep==True].index.tolist()
        print('\nFeatures Remaining after RFE:',features)


        importance_df = shapley_feature_selection(default_model, X_train[features], y_train,drop_columns=drop_rfe_shapley_columns)

        if len((importance_df[importance_df['shap_importance']>=importance_df['shap_importance'].mean()])['column_name'].to_list())<top_k_features:
            print('\nFeatures in ranked order are:', importance_df['column_name'].to_list())
            importance_df = importance_df[importance_df['shap_importance']>=importance_df['shap_importance'].mean()]
            features = importance_df['column_name'].to_list()
            #Force-fitting date features
            if ((not 'peak_week_flag' in features)&(not 'week' in features)&(not 'week_all' in features)&(not 'year' in features)&(not 'month' in features)&(not 'month_all' in features)&(not 'Week_of_month' in features)):
                features.append(importance_df_copy[importance_df_copy['column_name'].isin(['week', 'year', 'month', 'week_all', 'month_all', 'Week_of_month','peak_week_flag'])]['column_name'].to_list()[0])

            print('\nFinal top features remaining after Shapley feature selection are:',features)
            final_selected_features = features.copy()
        else:
            print('\nFeatures in ranked order are:', importance_df['column_name'].to_list())
            importance_df = importance_df[:top_k_features]
            features = importance_df['column_name'].to_list()
            #Force-fitting date features
            if ((not 'peak_week_flag' in features)&(not 'week' in features)&(not 'week_all' in features)&(not 'year' in features)&(not 'month' in features)&(not 'month_all' in features)&(not 'Week_of_month' in features)):
                features.append(importance_df_copy[importance_df_copy['column_name'].isin(['week', 'year', 'month', 'week_all', 'month_all', 'Week_of_month','peak_week_flag'])]['column_name'].to_list()[0])

            print('\nFinal top features remaining after Shapley feature selection are:',features)
            final_selected_features = features.copy()

        features.append(target)
        cognos_data = cognos_data[features+['date']]

    elif (shap_feature_selection==False)&(is_rfe_exec==True):
        X_train,y_train = feature_selection(cognos_data,target,drop_columns=drop_rfe_shapley_columns,target_base_features=base_features)
        X_train,y_train = rfe_scaler(X_train,y_train)
        _rfecv_features=rfecv(X_train,y_train)

        features=_rfecv_features[_rfecv_features.keep==True].index.tolist()
        #Force-fitting date features
        if ((not 'peak_week_flag' in features)&(not 'week' in features)&(not 'week_all' in features)&(not 'year' in features)&(not 'month' in features)&(not 'month_all' in features)&(not 'Week_of_month' in features)):
            features.append(importance_df_copy[importance_df_copy['column_name'].isin(['week', 'year', 'month', 'week_all', 'month_all', 'Week_of_month','peak_week_flag'])]['column_name'].to_list()[0])

        print('\nFeatures Remaining after RFE:',features)
        final_selected_features = features.copy()
        features.append(target)
        cognos_data = cognos_data[features+['date']]

    else:
        #Force-fitting date features
        if ((not 'peak_week_flag' in features)&(not 'week' in features)&(not 'week_all' in features)&(not 'year' in features)&(not 'month' in features)&(not 'month_all' in features)&(not 'Week_of_month' in features)):
            features.append(importance_df_copy[importance_df_copy['column_name'].isin(['week', 'year', 'month', 'week_all', 'month_all', 'Week_of_month','peak_week_flag'])]['column_name'].to_list()[0])

        print('\nFeatures used are:',features)
        final_selected_features = features.copy()
        features = features+[target]
        cognos_data = cognos_data[features+['date']]

    empties_features_features = list(set(features).intersection(set(empties_features_data.columns)))
    df1 = pd.concat([cognos_data,empties_features_data.loc[empties_features_data['date']>cognos_data['date'].max(),['date']+empties_features_features]],axis=0,ignore_index=True).set_index('date')
    df1['quarter'] = df1.index.quarter

    features = list(set(features)-set([target]))
    lag_numbers,med_numbers = [],[]
    highest_lag=0
    lag_cols = [col for col in features if 'lag_' in col]
    if len(lag_cols)>0:
        lag_numbers = [-1*int(col.split('_')[-2]) for col in lag_cols]
        highest_lag = min(lag_numbers)

    med_cols = [col for col in features if 'Rolling_' in col]
    if len(med_cols)>0:
        med_numbers = [-1*int(col.split('_')[-3]) for col in med_cols]
        highest_lag = min(highest_lag,min(med_numbers))

    trend_cols = [col for col in features if '_normalized_slope_' in col]
    if len(trend_cols)>0:
        trend_numbers = [-1*int(col[-1]) for col in trend_cols]
        highest_lag = min(highest_lag,min(trend_numbers))

    train_params_cur= train_val_params.copy()
    train_start_ = pd.to_datetime(train_val_params['train_start'])+relativedelta(weeks=-1*highest_lag)
    train_start_ = train_start_.strftime("%Y-%m-%d")
    train_params_cur['train_start']=train_start_

    print("\nTrain Paramters are",train_params_cur)
    ds = get_dataset(df1,train_params_cur,test_flag=True)
    train,val,test=ds['train'],ds['val'],ds['test']
    ds['train'] = ds['train'].replace([-np.inf,np.inf,np.NaN],0)
    ds['val'] = ds['val'].replace([-np.inf,np.inf,np.NaN],0)
    ds['test'] = ds['test'].replace([-np.inf,np.inf,np.NaN],0)

    cv_dict = cv_job(acc_model_params,ds,features,target)
    train_pred_df,eval_train,acc_model_params,models_dict = training_job(acc_model_params,ds,features,target)
    train_pred_df = train_pred_df.set_index('date')
    train_pred_df = train_pred_df.join(df1[[target]],how='left')


    forecast_df,eval_all = inference_job(acc_model_params,ds,features,target)
    train_pred_df,error_dict = ensemble_output(train_pred_df,eval_train,error_weights)
    forecast_df,error_dict = ensemble_output(forecast_df,eval_train,error_weights)
    forecast_df = forecast_df.set_index('date')
    forecast_df = forecast_df.join(df1[[target]],how='left')

    filter_top_k = list(error_dict.keys())[:]+['ensemble_avg']
    forecast_df = forecast_df.loc[:,[target]+filter_top_k]
    train_pred_df = train_pred_df.loc[:,[target]+filter_top_k]

    print("\nTraining and inference process completed for ",target, "\n")
    
    return train_pred_df,forecast_df,train_params_cur,eval_train,final_selected_features
        
def moving_averages(rec_data,test_df,accounts,forecast_all,train_forecasts):
    for account,parameter in accounts.items():
        cognos_data = rec_data[['date',account]]
        cognos_data = pd.concat([cognos_data,test_df],axis=0,ignore_index=True)
        cognos_data['date']=pd.to_datetime(cognos_data['date'])
        if 'alpha' in parameter.keys():
            cognos_data = predict_ewm(cognos_data,test_weeks,account,alpha=parameter['alpha'])
        else:
            cognos_data = predict_ewm(cognos_data,test_weeks,account,span=parameter['span'])
        forecast_df = forecast_all[account]
        forecast_df = pd.concat([forecast_df,cognos_data.drop(columns=[account]).set_index('date')],axis=1)
        
        train_pred_df = train_forecasts[account]
        train_pred_df = pd.concat([train_pred_df,cognos_data.drop(columns=[account]).set_index('date')],axis=1)
        
        forecast_all[account]=forecast_df
        train_forecasts[account]=train_pred_df
    return forecast_all,train_forecasts,train_params_cur

def ensemble_avg_train_val_error_analysis(train_forecasts, train_val_params, error_analysis):
    for account in error_analysis.keys():
        error_analysis[account]['ensemble_avg'] = {'train': 0, 'val': 0}

    for account in error_analysis.keys():
        error_analysis[account]['ensemble_avg']['train'] = np.abs(round((((np.abs(train_forecasts[account][train_val_params['train_start']:train_val_params['train_end']]['ensemble_avg'] - train_forecasts[account][train_val_params['train_start']:train_val_params['train_end']][account]))/train_forecasts[account][train_val_params['train_start']:train_val_params['train_end']][account])*100).mean(),2))
        error_analysis[account]['ensemble_avg']['val'] = np.abs(round((((np.abs(train_forecasts[account][train_val_params['val_start']:train_val_params['val_end']]['ensemble_avg'] - train_forecasts[account][train_val_params['val_start']:train_val_params['val_end']][account]))/train_forecasts[account][train_val_params['val_start']:train_val_params['val_end']][account])*100).mean(),2))
    return error_analysis

def recursive_modeling(rec_main,forecast_tag,output_path,forecast_all_output_filename,initial_features,drop_features,train_val_param_list):
        
    initial_features_accounts = []
    train_val_params_test_start = []
    for idx, train_val_params_id in enumerate(train_val_param_list):
        train_val_params_test_start.append(train_val_params_id['test_start'])
    for keys in initial_features.keys():
        initial_features_accounts.append(keys)
    rec_output_dict = dict.fromkeys(train_val_params_test_start)
    forecast_all = dict.fromkeys(train_val_params_test_start)
    error_analysis_timeframe=dict.fromkeys(train_val_params_test_start)
    error_analysis=dict.fromkeys(train_val_params_test_start)
    select_features = dict.fromkeys(train_val_params_test_start)
    for keys, values in forecast_all.items():
        forecast_all[keys] = dict.fromkeys(initial_features_accounts)
        error_analysis_timeframe[keys]=dict.fromkeys(initial_features_accounts)
        error_analysis[keys]=dict.fromkeys(initial_features_accounts)
        select_features[keys] = dict.fromkeys(initial_features_accounts)
        rec_output_dict[keys] = dict.fromkeys(initial_features_accounts)
    for train_val_params_id in train_val_param_list:       
        for account,feat in initial_features.items():
                
            train_pred_df,forecast_df,train_params_cur,eval_train,final_selected_features = rec_main(features = feat,target = account,drop_rfe_shapley_columns = drop_features.get(account,[]),train_val_params = train_val_params_id)
            select_features[train_val_params_id['test_start']][account] = final_selected_features
            forecast_all[train_val_params_id['test_start']][account] = forecast_df
            error_analysis_timeframe[train_val_params_id['test_start']][account] = train_params_cur
            error_analysis[train_val_params_id['test_start']][account] = eval_train
            rec_output_dict[train_val_params_id['test_start']][account]=train_pred_df
        ## Adding the ensemble model errors to "error analysis" which has the train and val errors for each model of accounts
        error_analysis[train_val_params_id['test_start']] = ensemble_avg_train_val_error_analysis(rec_output_dict[train_val_params_id['test_start']], train_val_params_id, error_analysis[train_val_params_id['test_start']])
      
    return rec_output_dict, select_features, forecast_all, error_analysis_timeframe, error_analysis 

def get_mob_features(google_mob_path,google_mob_filename,select_country,mob_dates_df):
    
    ##Read mobility data
    mob_df = read_mob_data(google_mob_path,google_mob_filename,select_country)

    ##Get mobility features 
    mob_data_agg = mob_feature_set(mob_df,select_country)
    
    #Getting all the dates
    mob_data_agg = pd.merge(mob_dates_df,mob_data_agg,on=['year','month','week'],how='left')
    mob_data_agg.drop(columns={'year','month','week'},inplace=True)

    return mob_data_agg
    
def read_mob_data(google_mob_path,google_mob_filename,select_country):
    mob_data_base = pd.read_csv(google_mob_path / google_mob_filename)
    mob_data_base.rename(columns=str.lower,inplace=True)
    mob_data_base.rename(columns={'country_region':'country'},inplace=True)
    mob_data_base = mob_data_base[pd.isnull(mob_data_base['sub_region_1'])]
    mob_data_base = mob_data_base.loc[mob_data_base['country']==select_country]
    mob_data_base = mob_data_base[['date','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','residential_percent_change_from_baseline','workplaces_percent_change_from_baseline']].reset_index(drop=True).copy() 
    mob_data_base['date'] = pd.to_datetime(mob_data_base['date'])
    mob_data_base['year']=mob_data_base['date'].dt.year
    mob_data_base['month']=mob_data_base['date'].dt.month
    mob_data_base['week']=mob_data_base['date'].dt.week
    mob_data_base = mob_data_base[['year','month','week','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','residential_percent_change_from_baseline','workplaces_percent_change_from_baseline']]
    return mob_data_base

def mob_feature_set(mob_data,select_country):    
    measures = {col:np.mean for col in mob_data.columns if 'baseline' in col}
    mob_data_agg = mob_data.groupby(['year','month','week'],as_index=False).agg(measures)
    mob_data_agg[list(measures.keys())] = mob_data_agg[list(measures.keys())]+100 
    return mob_data_agg
    
def create_null_flag(df):
    #Setting a flag for zero values
    df['null_flag'] = np.NaN
    df.loc[((df['Open']<=0.0001)|(df['Open']==np.NaN)),'null_flag'] = 'null'
    df['null_flag'].fillna('not_null',inplace=True)
    return df

def create_vol_tag(df_non_zero):
    ##Group the stock by cumulative volume into 10 groups
    regression_vol = df_non_zero.groupby(['Stock'])['Volume'].sum().reset_index().sort_values(['Volume'], ascending=False)
    regression_vol['Vol_cont'] = (regression_vol['Volume']/regression_vol['Volume'].sum())*100
    regression_vol['Vol_Prop_cumsum']=regression_vol['Vol_cont'].cumsum()
    regression_vol['Volume Group'] = np.where(regression_vol['Vol_Prop_cumsum']<10, '10% Vol',np.where(regression_vol['Vol_Prop_cumsum']<20, '20% Vol', np.where(regression_vol['Vol_Prop_cumsum']<30, '30% Vol', np.where(regression_vol['Vol_Prop_cumsum']<40, '40% Vol', np.where(regression_vol['Vol_Prop_cumsum']<50,'50% Vol', np.where(regression_vol['Vol_Prop_cumsum']<60, '60% Vol', np.where(regression_vol['Vol_Prop_cumsum']<70, '70% Vol', np.where(regression_vol['Vol_Prop_cumsum']<80, '80% Vol', np.where(regression_vol['Vol_Prop_cumsum']<90, '90% Vol','Complete 100% Vol')))))))))
    return regression_vol

def daily_percentage(df_daily):
    df_daily['year'] = df_daily['Date'].dt.year
    df_daily['dayofweek'] = df_daily['Date'].dt.dayofweek
    df_daily_dayofweek = df_daily.groupby(["year","dayofweek"])['Open'].sum().reset_index().rename(columns={"Open":"Open_per_day_of_week"})
    df_daily_year=df_daily.groupby(["year"])['Open'].sum().reset_index()
    df_daily = df_daily_dayofweek.merge(df_daily_year,
                                        on='year',
                                        how='left')
    df_daily['percentage'] = df_daily['Open_per_day_of_week']/df_daily['Open']
    return df_daily