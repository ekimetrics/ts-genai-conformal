import numpy as np
import torch
import math

def rolling_window_naive(df, available_points, prediction_length):
    n = available_points-1
    q = n//prediction_length

    predictions = []
    np.random.seed(0)

    for i in range(q+1):
        next_predictions = np.tile(df.iloc[1+i*prediction_length-1].values, (prediction_length, 1))
        predictions.append(next_predictions) 
        
    return np.concatenate(predictions, axis=0)[1:available_points]




def general_pipeline_naive(df, available_points, prediction_length):
    torch.manual_seed(0)
    np.random.seed(0)   

    n_channel = df.shape[1]
    
    metrics = {}

    calibration_results_naive = rolling_window_naive(df, available_points, prediction_length)

    n_cal = available_points-1
    local_quantiles = np.quantile(np.abs(df.iloc[1:available_points]-calibration_results_naive), math.ceil((n_cal+1)*0.9)/n_cal, axis=0)
    global_quantiles = np.quantile(np.abs(df.iloc[1:available_points]-calibration_results_naive).values.ravel(), math.ceil(((n_cal*n_channel)+1)*0.9)/(n_cal*n_channel))
    metrics['local_quantiles'] = local_quantiles.mean()

    test_results_naive = np.tile(df.iloc[available_points-1].values, (prediction_length, 1))
    
    mae = np.abs(test_results_naive-df.iloc[available_points:available_points+prediction_length, :]).mean(axis=0)
    metrics['mae'] = mae.mean()



    coverage_local=  100*((df.iloc[available_points:available_points+prediction_length]-test_results_naive).abs() < local_quantiles).mean(axis = 1)
    coverage_global = 100*((df.iloc[available_points:available_points+prediction_length]-test_results_naive).abs() < global_quantiles).mean(axis = 1)
    metrics['local_coverage'] = coverage_local.mean()
    metrics['global_coverage'] = coverage_global.mean()
    metrics['global_quantiles'] = global_quantiles

    return metrics, calibration_results_naive, test_results_naive, local_quantiles, global_quantiles