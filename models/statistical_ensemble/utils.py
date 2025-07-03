from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, AutoCES, DynamicOptimizedTheta
import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm import tqdm
import math
import torch


def rolling_window(data, available_points, context_length, prediction_length, season_length, freq):

    sf = StatsForecast(models = [AutoETS(season_length=season_length),
                             AutoCES(season_length=season_length),
                             DynamicOptimizedTheta(season_length=season_length)], freq = freq, n_jobs=-1)
    
    n_channel = data['unique_id'].nunique()
    train_length=context_length
    n = available_points-train_length
    q = n//prediction_length
    
    np.random.seed(0)

    predictions = [[] for i in range(q+1)]


    for i in tqdm(range(q+1)):
        df = data.loc[(data.d >= i*prediction_length) & (data.d < train_length+i*prediction_length)].drop(columns = 'd', axis=1).reset_index(drop=True)
        df_pred = sf.forecast(h=prediction_length, df = df)
        df_pred = df_pred.reset_index()
        df_pred['d'] = (train_length+i*prediction_length)+(df_pred.index%prediction_length)
        predictions[i] = df_pred
  
    return pd.concat(predictions).sort_values(by=['unique_id', 'd']).reset_index(drop=True)


def general_pipeline(data, available_points, prediction_length, season_length, freq):
    n_channel = data['unique_id'].nunique()
    metrics={}
    context_length = int(0.8*available_points)

    sf = StatsForecast(models = [AutoETS(season_length=season_length),
                                AutoCES(season_length=season_length),
                                DynamicOptimizedTheta(season_length=season_length)], freq = freq, n_jobs=-1)


    models = ['AutoETS', 'CES', 'DynamicOptimizedTheta']


    calibration_results = rolling_window(data, available_points, context_length, prediction_length, season_length, freq)
    calibration_results['y_pred'] = np.mean(calibration_results[['AutoETS', 'CES', 'DynamicOptimizedTheta']], axis=1)

    n_cal = available_points-context_length

    local_quantiles = np.quantile(np.abs(data[(data.d >=context_length) & (data.d < available_points)]['y'].reset_index(drop=True) - calibration_results[calibration_results.d < available_points]['y_pred'].reset_index(drop=True)).values.reshape(n_channel, n_cal), math.ceil((n_cal+1)*0.9)/n_cal, axis=1)

    global_quantiles = np.quantile(np.abs(data[(data.d >=context_length) & (data.d < available_points)]['y'].reset_index(drop=True) - calibration_results[calibration_results.d < available_points]['y_pred'].reset_index(drop=True)).values.ravel(), math.ceil(((n_cal*n_channel)+1)*0.9)/(n_cal*n_channel))

    df = data.loc[(data.d >=available_points-context_length) & (data.d <available_points)].drop(columns = 'd', axis=1).reset_index(drop=True)
    test_results = sf.forecast(h=prediction_length, df=df).reset_index(drop=True)
    test_results['y_pred'] = np.mean(test_results[['AutoETS', 'CES', 'DynamicOptimizedTheta']], axis=1)

    mae = np.mean(np.abs(test_results['y_pred'].reset_index(drop=True) - data[data.d >=available_points]['y'].reset_index(drop=True)))

    metrics['local_quantiles'] = np.mean(local_quantiles)
    metrics['global_quantiles'] = global_quantiles
    metrics['mae'] = mae


    coverage_global = np.abs(data[data.d>=available_points]['y'].reset_index(drop=True) - test_results['y_pred'])  < global_quantiles
    metrics['global_coverage'] = coverage_global.mean() * 100

    coverage_local = np.abs(data[data.d >= available_points]['y'].reset_index(drop=True) - test_results['y_pred']).values.reshape(n_channel, prediction_length)  < local_quantiles.reshape(-1, 1)
    metrics['local_coverage'] = coverage_local.mean()



    return metrics, calibration_results, test_results, local_quantiles, global_quantiles