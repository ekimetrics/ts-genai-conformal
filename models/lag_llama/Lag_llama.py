# Databricks notebook source
# MAGIC %md
# MAGIC #Packages

# COMMAND ----------

# MAGIC %sh
# MAGIC pip install --upgrade pip
# MAGIC pip install xlrd
# MAGIC pip install -r packages_llama.txt --quiet

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import math
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
import torch

from gluonts.dataset.pandas import PandasDataset
from gluonts.evaluation import make_evaluation_predictions, Evaluator
import sys
import os

sys.path.append(os.path.abspath('..'))
sys.path.insert(0, os.path.join(os.getcwd(), 'lag_llama'))

from utils import rolling_window, general_pipeline, get_lag_llama_predictions, model
from dataset.ercot import get_ercot
from dataset.nn5_daily import get_nn5_daily
from dataset.nn5_weekly import get_nn5_weekly
from dataset.m3_monthly import get_m3_monthly


from naive.utils import rolling_window_naive, general_pipeline_naive

torch.manual_seed(0)
np.random.seed(0)

%load_ext autoreload
%autoreload 2

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Ercot

# COMMAND ----------

ercot_df_f = get_ercot().set_index('ds').astype('float32')

print(ercot_df_f.shape)
print(ercot_df_f.reset_index().ds.min(), '---', ercot_df_f.reset_index().ds.max())

fig = px.line(ercot_df_f.reset_index(), 
              x='ds', 
              y='target', 
              title='ercot data')
fig.show()

# COMMAND ----------

n = len(ercot_df_f)
available_points = 2232
context_length = 512
prediction_length = 168
num_samples = 20
batch_size = 32

all_metrics = []

for i in np.linspace(0, n-available_points-prediction_length-200, 20, dtype=int):

    metrics, calibration_results, _, local_quantiles, _ = general_pipeline(ercot_df_f[i:], num_samples, available_points, context_length, prediction_length)

    metrics_naive, _, _, local_quantiles_naive, _ = general_pipeline_naive(ercot_df_f[i:], available_points, prediction_length)

    metrics['local_quantiles_normalized'] = local_quantiles/local_quantiles_naive
    metrics['mase'] = metrics['mae']/metrics_naive['mae']

    all_metrics.append(metrics)

# COMMAND ----------

{ 'mase' : np.stack([c['mase'] for c in all_metrics]).mean(),
 'quantile' : np.array([c['local_quantiles'] for c in all_metrics]).mean(),
 'coverage' : np.array([c['local_coverage'] for c in all_metrics]).mean(),
 'quantile_normalized' : np.array([c['local_quantiles_normalized'] for c in all_metrics]).mean()
}

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # NN5 Daily

# COMMAND ----------

df_nn5 = get_nn5_daily().astype('float32')

# COMMAND ----------

prediction_length = 35
available_points = len(df_nn5) - prediction_length
num_samples = 20
context_length = 128

metrics, calibration_results, test_results, local_quantiles, global_quantile = general_pipeline(df_nn5, num_samples, available_points, context_length, prediction_length)

metrics_naive, _, _, local_quantiles_naive, _ = general_pipeline_naive(df_nn5, available_points, prediction_length)

metrics['local_quantiles_normalized'] = (local_quantiles/local_quantiles_naive).mean()
metrics['global_quantiles_normalized'] = global_quantile.item()/metrics_naive['global_quantiles']
metrics['mase'] = metrics['mae']/metrics_naive['mae']

metrics

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # NN5 Weekly

# COMMAND ----------

df = get_nn5_weekly().astype('float32')

# COMMAND ----------

prediction_length = 12
available_points = len(df) - prediction_length
num_samples = 20
context_length = 64

metrics, calibration_results, test_results, local_quantiles, global_quantile = general_pipeline(df, num_samples, available_points, context_length, prediction_length)

metrics_naive, _, _, local_quantiles_naive, _ = general_pipeline_naive(df, available_points, prediction_length)

metrics['local_quantiles_normalized'] = (local_quantiles/local_quantiles_naive).mean()
metrics['global_quantiles_normalized'] = global_quantile.item()/metrics_naive['global_quantiles']
metrics['mase'] = metrics['mae']/metrics_naive['mae']

metrics

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # M3 Monthly

# COMMAND ----------

df_m3 = pd.read_excel('../../dataset/m3_monthly/m3_monthly.xls', sheet_name = 'M3Month')
df_m3['Starting Date'] = pd.to_datetime(df_m3['Starting Year'].astype(str) + '-' + df_m3['Starting Month'].astype(str), errors='coerce') \
    .fillna(pd.Timestamp('1990-01-01'))
df_m3 = df_m3.sort_values(by=['Starting Date', 'Series'])

col_non_null = df_m3.columns[~df_m3.isna().any()]
df_m3_entire = df_m3[col_non_null].T.iloc[6:]

n = len(df_m3_entire) - 1

l_datasets = []

for date in df_m3['Starting Date'].unique():
    tmp_df = df_m3[df_m3['Starting Date'] == date].copy()
    tmp_df = tmp_df.drop('Starting Date', axis=1).T.iloc[6:n+6]
    tmp_df = tmp_df.set_index(pd.date_range(start = date, periods=n, freq = 'M')).astype('float32')
    l_datasets.append(tmp_df)

# COMMAND ----------

prediction_length = 9
available_points = n - prediction_length
num_samples = 20
context_length = 32

l_metrics = []

for i, dataset in enumerate(l_datasets):

    n_channel = df_m3.Series.nunique()
    torch.manual_seed(0)
    np.random.seed(0)   
    metrics = {}

    tmp_calibration_results = rolling_window(dataset, available_points, context_length, prediction_length, num_samples)

    if i == 0:
        calibration_results = tmp_calibration_results
    else:
        calibration_results = np.concatenate((calibration_results, tmp_calibration_results))

n_cal = available_points-context_length
local_quantiles = np.quantile(np.abs(df_m3_entire.iloc[context_length:available_points]-calibration_results.T), math.ceil((n_cal+1)*0.9)/n_cal, axis=0)
global_quantile = np.quantile(np.abs(df_m3_entire.iloc[context_length:available_points]-calibration_results.T).values.ravel(), math.ceil(((n_cal*n_channel)+1)*0.9)/(n_cal*n_channel))

metrics['local_quantiles'] = local_quantiles.mean()

for i, dataset in enumerate(l_datasets):

    #Prediction sur le set de test
    fcsts_df = get_lag_llama_predictions(predictor = model(prediction_length, 32),
                                         dataset = PandasDataset(dict(dataset.iloc[:available_points])),
                                         num_samples=num_samples)
    tmp_test_results = np.array([b.quantile(.5) for b in fcsts_df])

    if i == 0:
        test_results = tmp_test_results
    else:
        test_results = np.concatenate((test_results, tmp_test_results))

    
mae = np.abs(test_results.T-df_m3_entire.iloc[available_points:available_points+prediction_length, :]).mean(axis=1)
metrics['mae'] = mae.mean()

local_coverage=  100*((df_m3_entire.iloc[available_points:available_points+prediction_length]-test_results.T).abs() < local_quantiles).mean(axis = 1)
global_coverage = 100*((df_m3_entire.iloc[available_points:available_points+prediction_length]-test_results.T).abs() < global_quantile).mean(axis = 1)

metrics['local_coverage'] = local_coverage.mean()
metrics['global_coverage'] = global_coverage.mean()
metrics['global_quantile'] = global_quantile


df_m3_f = get_m3_monthly().astype('float32')

metrics_naive, _, _, local_quantiles_naive, _ = general_pipeline_naive(df_m3_f, available_points, prediction_length)

metrics['local_quantiles_normalized'] = (local_quantiles/local_quantiles_naive).mean()
metrics['global_quantiles_normalized'] = global_quantile.item()/metrics_naive['global_quantiles']
metrics['mase'] = metrics['mae']/metrics_naive['mae']

metrics

# COMMAND ----------


