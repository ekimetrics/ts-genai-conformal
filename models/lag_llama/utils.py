import os 
import sys

import pandas as pd
import numpy as np
import math
from tqdm import tqdm
import torch



from gluonts.dataset.pandas import PandasDataset
from gluonts.evaluation import make_evaluation_predictions, Evaluator

from lag_llama.gluon.estimator import LagLlamaEstimator

torch.manual_seed(0)
np.random.seed(0)



#Paramètres globaux 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ckpt = torch.load("lag-llama.ckpt", map_location=device) # Uses GPU since in this Colab we use a GPU.
estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

def model(prediction_length, context_length):
    estimator = LagLlamaEstimator(
        ckpt_path="lag-llama.ckpt",
        prediction_length=prediction_length,
        context_length=context_length, # Lag-Llama was trained with a context length of 32, but can work with any context length

        # estimator args
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],
        rope_scaling=None,
        batch_size=32,
        num_parallel_samples=20,
        device=device,
    )

    lightning_module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)
    return predictor



def get_lag_llama_predictions(predictor, dataset, num_samples):
    forecast_it, _ = make_evaluation_predictions(
        dataset=dataset,
        predictor=predictor,
        num_samples=num_samples
    )
    return forecast_it



def rolling_window(df, available_points, context_length, prediction_length, num_samples):
    n = available_points-context_length
    q = n//prediction_length

    predictions = []

    torch.manual_seed(0)
    np.random.seed(0)

    predictor = model(prediction_length, 32)
    for i in tqdm(range(q+1)):
        dataset_df = PandasDataset(dict(df.iloc[:context_length+i*prediction_length]))
        preds = get_lag_llama_predictions(predictor = predictor,
                                           dataset = dataset_df,
                                           num_samples = num_samples)
        next_predictions = np.array([b.quantile(.5) for b in preds])
        predictions.append(next_predictions) 

    return np.concatenate(predictions, axis=1)[:, :available_points-context_length]



def general_pipeline(df, num_samples, calibration_length, context_length, prediction_length):
    n_channel = df.shape[1]
    torch.manual_seed(0)
    np.random.seed(0)   
    metrics = {}

    calibration_results = rolling_window(df, calibration_length, context_length, prediction_length, num_samples)

    n_cal = calibration_length-context_length
    local_quantiles = np.quantile(np.abs(df.iloc[context_length:calibration_length]-calibration_results.T), math.ceil((n_cal+1)*0.9)/n_cal, axis=0)
    global_quantile = np.quantile(np.abs(df.iloc[context_length:calibration_length]-calibration_results.T).values.ravel(), math.ceil(((n_cal*n_channel)+1)*0.9)/(n_cal*n_channel))

    metrics['local_quantiles'] = local_quantiles.mean()

    #Prediction sur le set de test
    fcsts_df = get_lag_llama_predictions(predictor = model(prediction_length, 32),
                                         dataset = PandasDataset(dict(df.iloc[:calibration_length])),
                                         num_samples=num_samples)
    test_results = np.array([b.quantile(.5) for b in fcsts_df]).T

    
    mae = np.abs(test_results-df.iloc[calibration_length:calibration_length+prediction_length, :]).mean(axis=1)
    metrics['mae'] = mae.mean()


    local_coverage=  100*((df.iloc[calibration_length:calibration_length+prediction_length]-test_results).abs() < local_quantiles).mean(axis = 1)

    global_coverage = 100*((df.iloc[calibration_length:calibration_length+prediction_length]-test_results).abs() < global_quantile).mean(axis = 1)
    metrics['local_coverage'] = local_coverage.mean()
    metrics['global_coverage'] = global_coverage.mean()
    metrics['global_quantile'] = global_quantile

    return metrics, calibration_results, test_results, local_quantiles, global_quantile
