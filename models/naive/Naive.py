# Databricks notebook source
# MAGIC %md
# MAGIC # Packages

# COMMAND ----------

# MAGIC %sh
# MAGIC pip install --upgrade pip
# MAGIC pip install xlrd

# COMMAND ----------

import pandas as pd
import plotly.express as px
import numpy as np
import math
import torch

from utils import rolling_window_naive, general_pipeline_naive

from dataset.ercot import get_ercot
from dataset.nn5_daily import get_nn5_daily
from dataset.nn5_weekly import get_nn5_weekly
from dataset.m3_monthly import get_m3_monthly

%load_ext autoreload
%autoreload 2

# COMMAND ----------

# MAGIC %md
# MAGIC #Ercot

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

all_metrics = []

for i in np.linspace(0, n-available_points-prediction_length-200, 20, dtype=int):
    metrics,_, _, _, _ = general_pipeline_naive(ercot_df[i:], available_points, prediction_length)
    all_metrics.append(metrics)

# COMMAND ----------

{ 'coverage' : np.array([c['local_coverage'] for c in all_metrics]).mean(), 
 'quantile' : np.array([c['local_quantiles'] for c in all_metrics]).mean()
}

# COMMAND ----------

# MAGIC %md
# MAGIC # NN5 Daily

# COMMAND ----------

df_nn5 = get_nn5_daily()
print(df_nn5.shape)
print(df_nn5.reset_index().ds.min(), '---', df_nn5.reset_index().ds.max())

fig = px.line(df_nn5.reset_index(), 
              x='ds', 
              y=0, 
              title='nn5 daily data')
fig.show()

# COMMAND ----------

n = len(df_nn5)
prediction_length = 35
available_points = n - prediction_length

metrics, _, test_results_naive, local_quantiles, _ = general_pipeline_naive(df_nn5, available_points, prediction_length)
metrics['mase'] = 1
metrics

# COMMAND ----------

# MAGIC %md
# MAGIC # NN5 Weekly

# COMMAND ----------

df = get_nn5_weekly()

print(df.shape)
print(df.reset_index().index.min(), '---', df.reset_index().index.max())

fig = px.line(df.reset_index(), 
              x='index', 
              y=0, 
              title='nn5 weekly data')
fig.show()

# COMMAND ----------

n = len(df)
prediction_length = 12
available_points = n - prediction_length

metrics, _, _, _, _ = general_pipeline_naive(df, available_points, prediction_length)
metrics['mase'] = 1
metrics

# COMMAND ----------

# MAGIC %md
# MAGIC # M3 Monthly

# COMMAND ----------

df_m3 = get_m3_monthly()

print(df_m3.shape)
print(df_m3.reset_index().index.min(), '---', df_m3.reset_index().index.max())

fig = px.line(df_m3.reset_index(), 
              x='index', 
              y=0, 
              title='m3 monthly data')
fig.show()


# COMMAND ----------

n = len(df_m3)
prediction_length = 9
available_points = n - prediction_length

metrics, _, _, _, _ = general_pipeline_naive(df_m3, available_points, prediction_length)
metrics['mase'] = 1
metrics

# COMMAND ----------


