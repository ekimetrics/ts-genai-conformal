import timesfm
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import math

def rolling_window(df, available_points, context_length, prediction_length, freq, tfm):
    n = available_points-context_length
    q = n//prediction_length

    freqs = {'W' : 1, 'M' : 1, 'H' : 0, 'D' : 0}

    predictions = []

    torch.manual_seed(0)
    np.random.seed(0)


    for i in tqdm(range(q+1)):
        preds = tfm.forecast(inputs = df.iloc[i*prediction_length:context_length+i*prediction_length].to_numpy().T, 
                     freq = [freqs[freq] for i in range(df.shape[1])])[0]
        predictions.append(preds) 

    return np.concatenate(predictions, axis=1)[:, :available_points-context_length]



def general_pipeline(df, available_points, context_length, prediction_length, freq, timesfm_version):
    freqs = {'W' : 1, 'M' : 1, 'H' : 0, 'D' : 0}
    torch.manual_seed(0)
    np.random.seed(0)   
    metrics = {}

    if timesfm_version == 1:
        tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="gpu",
            per_core_batch_size=32,
            horizon_len=prediction_length,
            context_len=context_length,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
        )
    else:
        tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="gpu",
            per_core_batch_size=32,
            horizon_len=prediction_length,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=50,
            model_dims=1280,
            context_len=context_length,
            use_positional_embedding=False,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
        )

    calibration_results = rolling_window(df, available_points, context_length, prediction_length, freq, tfm)

    n_cal = available_points-context_length
    n_channel=df.shape[1]
    local_quantiles = np.quantile(np.abs(df.iloc[context_length:available_points]-calibration_results.T), math.ceil((n_cal+1)*0.9)/n_cal, axis=0)
    global_quantile = np.quantile(np.abs(df.iloc[context_length:available_points]-calibration_results.T).values.ravel(), math.ceil(((n_cal*n_channel)+1)*0.9)/(n_cal*n_channel))

    metrics['local_quantiles'] = local_quantiles.mean()

    #Prediction sur le set de test

    if timesfm_version == 1:

        tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="gpu",
            per_core_batch_size=32,
            horizon_len=prediction_length,
            context_len=context_length,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
        )
    
    else:

        tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="gpu",
            per_core_batch_size=32,
            horizon_len=prediction_length,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=50,
            model_dims=1280,
            context_len=context_length,
            use_positional_embedding=False,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
        )

    test_results = tfm.forecast(inputs = df.iloc[available_points-context_length:available_points].to_numpy().T, 
                    freq = [freqs[freq] for i in range(df.shape[1])])[0].T
    
    mae = np.abs(test_results-df.iloc[available_points:available_points+prediction_length, :]).mean(axis=1)
    metrics['mae'] = mae.mean()

    local_coverage=  100*((df.iloc[available_points:available_points+prediction_length]-test_results).abs() < local_quantiles).mean(axis = 0)

    global_coverage = 100*((df.iloc[available_points:available_points+prediction_length]-test_results).abs() < global_quantile).mean(axis = 0)
    metrics['local_coverage'] = local_coverage.mean()
    metrics['global_coverage'] = global_coverage.mean()
    metrics['global_quantile'] = global_quantile

    return metrics, calibration_results, test_results, local_quantiles, global_quantile