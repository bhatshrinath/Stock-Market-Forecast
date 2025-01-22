import numpy as np
import pandas as pd
import math
from statsmodels.tsa import holtwinters
import warnings
warnings.filterwarnings('ignore')

from func_util_stock_market_forecasting import get_forecast_dates_df

def mape(y_act,y_pred):
    return np.round(np.mean(np.abs((y_act-y_pred)/(y_act+0.000001)))*100,2)

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

def holtwinter_esm(train_data,test_data,forecast_duration):
    '''
    Generate forecasts using Holt-Winter smoothing
    inputs:
    train_data and test_data - arrays
    forecast_duration - duration of expected forecast
    
    Open:
    mape_dict - dictionary with model name as key and mape as values
    forecast_dict - dictionary with model name as key and forecasted array as values
    '''
    
    mape_dict = {}
    forecast_dict = {}    
    
    train_arr = train_data.copy()
    test_arr = test_data.copy()
    model = holtwinters.SimpleExpSmoothing(train_arr).fit(smoothing_level=0.9,optimized=False,use_brute=True)
    predictions = model.forecast(len(test_arr))
    mape_dict['HW_simexpsm'] = mape(np.array(test_arr),np.array(predictions))
    
    train_arr.extend(test_arr)
    model = holtwinters.SimpleExpSmoothing(train_arr).fit(smoothing_level=0.9,optimized=False,use_brute=True)
    predictions = model.forecast(forecast_duration)
    forecast_dict['HW_simexpsm'] = predictions
    train_arr = train_data.copy()
    test_arr = test_data.copy()
    model = holtwinters.ExponentialSmoothing(train_arr,trend='add',seasonal='add',seasonal_periods=2).fit()
    predictions = model.forecast(len(test_arr))
    mape_dict['HW_expsm'] = mape(np.array(test_arr),np.array(predictions))
    
    train_arr.extend(test_arr)
    model = holtwinters.ExponentialSmoothing(train_arr,trend='add',seasonal='add',seasonal_periods=2).fit()
    predictions = model.forecast(forecast_duration)
    forecast_dict['HW_expsm'] = predictions
    
    return mape_dict,forecast_dict

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

# def recursive_sma(df1,test_period_list,window_num=48):
#     df1['Open_copy'] = df1['Open'].copy()
#     for i in range(len(test_period_list)):
#         print(pd.DataFrame(test_period_list).sort_values(by=0).iloc[i,0])
#         temp = df1.loc[df1['Date']<=pd.DataFrame(test_period_list).sort_values(by=0).iloc[i,0],'Open_copy'].shift(periods=1).rolling(window=window_num).mean()
   
#         df1.loc[temp[~temp.isnull()].index[0],'Open_copy'] = temp[~temp.isnull()].values[0]
 
#     df1['rec_sma_'+str(window_num)] = temp
#     return df1.drop(columns={'Open_copy'})

# def recursive_ema(df1,test_period_list,window_num=48):
#     df1['Open_copy'] = df1['Open'].copy()
#     for i in range(len(test_period_list)):
#         print(pd.DataFrame(test_period_list).sort_values(by=0).iloc[i,0])
#         temp = df1.loc[df1['Date']<=pd.DataFrame(test_period_list).sort_values(by=0).iloc[i,0],'Open_copy'].shift(periods=1).ewm(span=window_num,adjust=False).mean()
   
#         df1.loc[temp[~temp.isnull()].index[0],'Open_copy'] = temp[~temp.isnull()].values[0]
 
#     df1['rec_ema_'+str(window_num)] = temp
#     return df1.drop(columns={'Open_copy'})

def ewm_alpha_opt(train,test,forecast_duration,arr_dates):
    
    # create dataframe with nan in place of test values
    arr_test = np.array([np.NaN]*len(test))
    arr_train_test = np.append(train,arr_test,axis=0)
    
    arr_dates_train = arr_dates[:len(train)]
    arr_dates_train_test = arr_dates[:len(arr_train_test)]

    test_weeks = list(set(arr_dates_train_test)-set(arr_dates_train))
    df_train_test = pd.DataFrame({'date':arr_dates_train_test,'var':arr_train_test})
    
    # create a dataframe with nan in place of forecast values
    arr_train_test = np.append(train,test,axis=0)
    arr_forecast = np.array([np.NaN]*forecast_duration)
    arr_forecast = np.append(arr_train_test,arr_forecast,axis=0)

    forecast_weeks = list(set(arr_dates)-set(arr_dates_train_test))
    if len(forecast_weeks)!=forecast_duration:
        print("Something is wrong. Please check input for forecast duration")
    df_forecast = pd.DataFrame({"date":arr_dates,'var':arr_forecast})
    
    mape_dict = {}
    forecast_dict = {}
    for alpha in np.arange(0.01,1.01,0.1):
        df_ewm = predict_ewm(df_train_test,test_weeks,'var',alpha=alpha)
        df_ewm.columns = ['date','var','var_sim','var_pred']
        predictions = df_ewm['var_pred'].values[-len(test_weeks):]
        mape_dict['ewm_'+str(alpha)] = mape(np.array(test),np.array(predictions))
        
        df_ewm = predict_ewm(df_forecast,forecast_weeks,'var',alpha=alpha)
        df_ewm.columns = ['date','var','var_sim','var_pred']
        predictions = df_ewm['var_pred'].values[-len(forecast_weeks):]
        forecast_dict['ewm_'+str(alpha)] = predictions

    return mape_dict, forecast_dict

def generate_forecasts(train,test,forecast_duration,arr_dates):
    '''
    Generate forecasts from each model available
    '''
    mape_dict_master, forecast_dict_master = holtwinter_esm(train,test,forecast_duration)
   
    mape_dict, forecast_dict = ewm_alpha_opt(train,test,forecast_duration,arr_dates)
    mape_dict_master.update(mape_dict)
    forecast_dict_master.update(forecast_dict)
     
    return mape_dict_master, forecast_dict_master

def bestmodel_univar(mape_dict, forecast_dict):
    '''
    Select the best model based on MAPE value
    '''
    best_model = min(mape_dict, key=mape_dict.get)
    
    return forecast_dict[best_model]

def univar_forecast(arr_ts,forecast_duration,arr_dates):
    '''
    Generate forecasts using Univariate methods
    input:
    arr_ts - array of time series
    forecast_duration - duration of forecast
    
    return: arr_ts - array of time series along with forecast
    '''
    
    train_test_split = -forecast_duration
    train = arr_ts[:train_test_split]
    test = arr_ts[train_test_split:]
    
    # function to generate forecasts
    mape_dict, forecast_dict = generate_forecasts(train,test,forecast_duration,arr_dates)

    print("MAPE of all univariate models on test period are",mape_dict,"\n")
    print("\n--------------------------------------------------------------------------------------------------------\n")
    
    # function to select best forecast
    best_forecast = bestmodel_univar(mape_dict, forecast_dict)
    
    arr_ts = np.append(arr_ts,best_forecast,axis=0)
    return arr_ts

