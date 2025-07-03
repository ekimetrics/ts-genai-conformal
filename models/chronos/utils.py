import pandas as pd
import numpy as np
import torch
import math
from tqdm import tqdm

from chronos import BaseChronosPipeline



def rolling_window(calibration_tensor, available_points, context_length, prediction_length, num_samples, batch_size, n_channel, pipeline):
    n = calibration_tensor.shape[1]-context_length
    q = n//prediction_length
    r = n % prediction_length

    torch.manual_seed(0)
    np.random.seed(0)
    
    q_batch = n_channel//batch_size

    predictions = []
    for i in tqdm(range(q+1)):
        results = []
        for j in range(q_batch+1):
            if (j ==q_batch and n_channel%batch_size==0) : 
                continue
            df = calibration_tensor[batch_size*j:min(n_channel, batch_size*(j+1)), i*prediction_length:context_length+i*prediction_length]
            fcsts_df_cal = pipeline.predict(df, prediction_length, limit_prediction_length=False)
            next_predictions = torch.median(fcsts_df_cal, dim=1).values
            results.append(next_predictions)
        predictions.append(torch.cat(results, dim = 0))

        
    return torch.cat(predictions, dim=1)[:n_channel, :available_points-context_length]



def general_pipeline(tensor_df, available_points, prediction_length, num_samples, batch_size, n_channel, context_length, chronos_version):
    torch.manual_seed(0)
    np.random.seed(0)
    
    metrics = {}
    calibration_tensor = tensor_df[:n_channel, :available_points]
    test_tensor = tensor_df[:n_channel, available_points:available_points+prediction_length]

    if chronos_version == 'bolt':
        pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-small",
        device_map="cuda", 
        torch_dtype=torch.bfloat16,
        )
    else:
        pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-t5-small",
            device_map="cuda", 
            torch_dtype=torch.bfloat16,
        )


    calibration_results = rolling_window(calibration_tensor, available_points, context_length, prediction_length, num_samples, batch_size, n_channel, pipeline)

    

    n_cal = available_points-context_length  

    local_quantiles=torch.quantile(torch.abs(calibration_tensor[:n_channel, context_length:available_points]-calibration_results), math.ceil(0.9*(n_cal+1))/n_cal, dim=1)
    global_quantile = torch.quantile(torch.abs(calibration_tensor[:n_channel, context_length:available_points]-calibration_results).reshape(-1),  math.ceil(0.9*((n_cal*n_channel)+1))/(n_cal*n_channel))



    q_cal = n_channel//batch_size

    results = []
    for i in tqdm(range(q_cal+1)):
        if i==q_cal and n_channel%batch_size==0:
            continue
        df = calibration_tensor[batch_size*i:min(n_channel, batch_size*(i+1)), -context_length:]
        res = pipeline.predict(df, prediction_length, limit_prediction_length=False)
        results.append(torch.median(res, dim=1).values)

    test_results = torch.cat(results)


    mae = (test_results-test_tensor[:n_channel]).abs().mean()
    metrics['mae'] = mae.mean().item()


    local_coverage = torch.mean(100*((test_tensor[:n_channel]-test_results).abs() < local_quantiles.view(-1, 1)).float(), axis = 1)
    metrics['local_coverage'] = local_coverage.mean().item()

    global_coverage = torch.mean(100*((test_tensor[:n_channel]-test_results).abs() < global_quantile).float(), axis = 1)
    metrics['global_coverage'] = global_coverage.mean().item()

    metrics['local_quantiles'] = local_quantiles.mean().item()
    metrics['global_quantile'] = global_quantile.item()

    return metrics, calibration_results, test_results, local_quantiles, global_quantile

    