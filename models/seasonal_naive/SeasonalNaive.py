# Databricks notebook source
# MAGIC %md
# MAGIC #Packages

# COMMAND ----------

# MAGIC %sh
# MAGIC pip install --upgrade pip
# MAGIC pip install xlrd

# COMMAND ----------

import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
import matplotlib.pyplot as plt
import math
import torch

import sys
import os

from dataset.ercot import get_ercot
from dataset.nn5_daily import get_nn5_daily
from dataset.nn5_weekly import get_nn5_weekly
from dataset.m3_monthly import get_m3_monthly


sys.path.append(os.path.abspath('..'))

from naive.utils import rolling_window_naive, general_pipeline_naive

# COMMAND ----------

# MAGIC %md
# MAGIC # Ercot

# COMMAND ----------

ercot_df = get_ercot(sample=True).set_index('ds')
print(ercot_df.shape)
print(ercot_df.reset_index().ds.min(), '---', ercot_df.reset_index().ds.max())

fig = px.line(ercot_df.reset_index(), 
              x='ds', 
              y='target', 
              title='ercot data')
fig.show()

# COMMAND ----------

n = len(ercot_df)
available_points = 2232
prediction_length = 168

res=[]
for i in np.linspace(0, n-available_points-prediction_length-200, 20, dtype=int):
    n_cal = available_points-prediction_length
    local_quantiles = np.quantile(ercot_df.diff(prediction_length)[i+prediction_length:i+available_points].abs(), math.ceil((n_cal+1)*0.9)/n_cal)
    mae_sn = ercot_df.diff(prediction_length)[i+available_points:i+available_points+prediction_length].abs().mean()

    metrics_naive,_, _, local_quantiles_naive, _ = general_pipeline_naive(ercot_df[i:], available_points, prediction_length)
    mase = mae_sn/metrics_naive['mae']

    local_coverage = 100*(ercot_df.diff(prediction_length)[i+available_points:i+available_points+prediction_length].abs() < local_quantiles).mean().item()

    local_quantiles_normalized = local_quantiles/local_quantiles_naive
    res.append([local_quantiles, local_coverage, mase.item(), local_quantiles_normalized.item()])
    break

# COMMAND ----------

dict(zip(['quantile', 'coverage', 'mase', 'quantile_normalized'], np.stack(res).mean(axis=0)))

# COMMAND ----------

# MAGIC %md
# MAGIC # NN5 Daily

# COMMAND ----------

df_nn5 = get_nn5_daily()

# COMMAND ----------

prediction_length=35
available_points=len(df_nn5)-prediction_length

n_cal = available_points-prediction_length
n_channel = df_nn5.shape[1]
local_quantiles = np.quantile(df_nn5.diff(prediction_length)[prediction_length:available_points].abs().to_numpy(), math.ceil((n_cal+1)*0.9)/n_cal, axis=0)
global_quantile = np.quantile(df_nn5.diff(prediction_length)[prediction_length:available_points].abs().to_numpy().ravel(), math.ceil(((n_cal*n_channel)+1)*0.9)/(n_cal*n_channel))

local_coverage = 100*(df_nn5.diff(prediction_length).abs()[available_points:available_points+prediction_length] < local_quantiles).mean()
global_coverage = 100*(df_nn5.diff(prediction_length).abs()[available_points:available_points+prediction_length] < global_quantile).mean()

mae = df_nn5.diff(prediction_length).abs()[available_points:available_points+prediction_length].mean()

metrics_naive, _, _, local_quantiles_naive, global_quantile_naive = general_pipeline_naive(df_nn5, available_points, prediction_length)

local_quantiles_normalized = local_quantiles/local_quantiles_naive

{
    'mase' : mae.mean()/metrics_naive['mae'],
    'mae' : mae.mean(),
    'local_quantiles' : local_quantiles.mean(),
    'global_quantile' : global_quantile,
    'local_coverage' : local_coverage.mean(),
    'global_coverage' : global_coverage.mean(),
    'local_quantiles_normalized' : local_quantiles_normalized.mean(),
    'global_quantiles_normalized' : global_quantile/global_quantile_naive
}

# COMMAND ----------

# MAGIC %md
# MAGIC # NN5 Weekly

# COMMAND ----------

df = get_nn5_weekly()

# COMMAND ----------

prediction_length=12
available_points=len(df)-prediction_length

n_cal = available_points-prediction_length
n_channel = df.shape[1]
local_quantiles = np.quantile(df.diff(prediction_length)[prediction_length:available_points].abs().to_numpy(), math.ceil((n_cal+1)*0.9)/n_cal, axis=0)
global_quantile = np.quantile(df.diff(prediction_length)[prediction_length:available_points].abs().to_numpy().ravel(), math.ceil(((n_cal*n_channel)+1)*0.9)/(n_cal*n_channel))

local_coverage = 100*(df.diff(prediction_length).abs()[available_points:available_points+prediction_length] < local_quantiles).mean()
global_coverage = 100*(df.diff(prediction_length).abs()[available_points:available_points+prediction_length] < global_quantile).mean()

mae = df.diff(prediction_length).abs()[available_points:available_points+prediction_length].mean()

metrics_naive, _, _, local_quantiles_naive, global_quantile_naive = general_pipeline_naive(df, available_points, prediction_length)

local_quantiles_normalized = local_quantiles/local_quantiles_naive

{   
    'mase' : mae.mean()/metrics_naive['mae'],
    'mae' : mae.mean(),
    'local_quantiles' : local_quantiles.mean(),
    'global_quatile' : global_quantile,
    'local_coverage' : local_coverage.mean(),
    'global_coverage' : global_coverage.mean(),
    'local_quantiles_normalized' : local_quantiles_normalized.mean(),
    'global_quantiles_normalized' : global_quantile/global_quantile_naive
}

# COMMAND ----------

# MAGIC %md
# MAGIC # M3M

# COMMAND ----------

df_m3_f = get_m3_monthly()

# COMMAND ----------

prediction_length=9
available_points=len(df_m3_f)-prediction_length

n_cal = available_points-prediction_length
n_channel = df_m3_f.shape[1]
local_quantiles = np.quantile(df_m3_f.diff(prediction_length)[prediction_length:available_points].abs().to_numpy(), math.ceil((n_cal+1)*0.9)/n_cal, axis=0)
global_quantile = np.quantile(df_m3_f.diff(prediction_length)[prediction_length:available_points].abs().to_numpy().ravel(), math.ceil(((n_cal*n_channel)+1)*0.9)/(n_cal*n_channel))

local_coverage = 100*(df_m3_f.diff(prediction_length).abs()[available_points:available_points+prediction_length] < local_quantiles).mean()
global_coverage = 100*(df_m3_f.diff(prediction_length).abs()[available_points:available_points+prediction_length] < global_quantile).mean()

mae = df_m3_f.diff(prediction_length).abs()[available_points:available_points+prediction_length].mean()

metrics_naive, _, _, local_quantiles_naive, global_quantile_naive = general_pipeline_naive(df_m3_f, available_points, prediction_length)

local_quantiles_normalized = local_quantiles/local_quantiles_naive

{
    'mase' : mae.mean()/metrics_naive['mae'],
    'mae' : mae.mean(),
    'local_quantiles' : local_quantiles.mean(),
    'global_quatile' : global_quantile,
    'local_coverage' : local_coverage.mean(),
    'global_coverage' : global_coverage.mean(),
    'local_quantiles_normalized' : local_quantiles_normalized.mean(),
    'global_quantiles_normalized' : global_quantile/global_quantile_naive
}

# COMMAND ----------


