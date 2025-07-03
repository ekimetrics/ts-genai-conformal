import torch
import numpy as np
import math
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from uni2ts.model.moirai import MoiraiForecast, MoiraiModule





def pipeline(SIZE, PDT, CTX, PSZ, NS, BSZ):
    model = MoiraiForecast(
        module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-{SIZE}", revision='a34614afbe6b16fffbc11c77daba5aab3ed277fb'),
        prediction_length=PDT,
        context_length=CTX,
        patch_size=PSZ,
        num_samples=NS,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0
    )

    predictor = model.create_predictor(batch_size=BSZ)
    return predictor


def rolling_window(df, available_points, context_length, prediction_length, num_samples, SIZE, PSZ, BSZ):
    n = available_points-context_length
    q = n//prediction_length
    torch.manual_seed(0)
    np.random.seed(0)

    predictions = [[] for i in range(q+1)]
    model = pipeline(SIZE, prediction_length, context_length, PSZ, num_samples, BSZ)

    for i in tqdm(range(q+1)):
        preds = model.predict(PandasDataset(dict(df.iloc[:context_length+i*prediction_length])))
        predictions[i] = np.array([b.quantile(.5) for b in preds])
        
    return np.concatenate(predictions, axis=1)[:, :available_points-context_length]


def general_pipeline(df, SIZE, PSZ, BSZ, num_samples, available_points, context_length, prediction_length):
    np.random.seed(0)
    torch.manual_seed(0)
    metrics = {}

    calibration_results = rolling_window(df, available_points, context_length, prediction_length, num_samples, SIZE, PSZ, BSZ)

    n_cal = available_points-context_length
    local_quantiles = np.quantile(np.abs(df.iloc[context_length:available_points]-calibration_results.T), math.ceil(0.9*(n_cal+1))/n_cal, axis=0)
    global_quantile = np.quantile(np.abs(df.iloc[context_length:available_points]-calibration_results.T).values.ravel(), math.ceil(0.9*(n_cal+1))/n_cal)

    #Prediction sur le set de test    
    model = pipeline(SIZE, prediction_length, context_length, PSZ, num_samples, BSZ)
    
    
    fcsts_df = model.predict(PandasDataset(dict(df.iloc[:available_points])))
    test_results = np.stack([b.quantile(.5) for b in fcsts_df]).T
    
    
    metrics['mae'] = np.abs(test_results-df.iloc[available_points:available_points+prediction_length, :]).mean()

    metrics['local_coverage']= 100*((df.iloc[available_points:available_points+prediction_length]-test_results).abs() < local_quantiles).mean()

    metrics['global_coverage']= 100*((df.iloc[available_points:available_points+prediction_length]-test_results).abs() < global_quantile).mean()


    metrics['local_quantiles'] = local_quantiles.mean()

    metrics['global_quantile'] = global_quantile

    return metrics, calibration_results, test_results, local_quantiles, global_quantile



