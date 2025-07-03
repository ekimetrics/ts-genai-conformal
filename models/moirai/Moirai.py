# Databricks notebook source
# MAGIC %md
# MAGIC #Packages

# COMMAND ----------

# MAGIC %sh
# MAGIC pip install --upgrade pip
# MAGIC pip install -r packages_moirai.txt

# COMMAND ----------

import torch
import math
import pandas as pd
import numpy as np

torch.manual_seed(0)
np.random.seed(0)

import sys
import os

sys.path.append(os.path.abspath('..'))
from utils import rolling_window, general_pipeline
from naive.utils import rolling_window_naive, general_pipeline_naive
from dataset.ercot import get_ercot
from dataset.nn5_daily import get_nn5_daily
from dataset.nn5_weekly import get_nn5_weekly
from dataset.m3_monthly import get_m3_monthly


# COMMAND ----------

# MAGIC %md
# MAGIC # Ercot

# COMMAND ----------

ercot_df_f = get_ercot(with_features=False).set_index('ds')

# COMMAND ----------

n = len(ercot_df_f)
available_points = 744
prediction_length = 24 
num_samples = 20
context_length = 256
SIZE = "small"
PSZ = "auto"
BSZ = 32  

all_metrics = []

for i in np.linspace(0, n-available_points-prediction_length-200, 20, dtype=int):
    metrics, _, _, local_quantiles, _ = general_pipeline(ercot_df_f[i:], SIZE, PSZ, BSZ, num_samples, available_points, context_length, prediction_length)
    all_metrics.append(metrics)

    metrics_naive, _, _, local_quantiles_naive, _ = general_pipeline_naive(ercot_df_f[['target']][i:], available_points, prediction_length)

    metrics['local_quantiles_normalized'] = local_quantiles/local_quantiles_naive
    metrics['mase'] = metrics['mae']/metrics_naive['mae']


# COMMAND ----------

{'mase' : np.array([c['mase'] for c in all_metrics]).mean(),
 'coverage' : np.array([c['local_coverage'] for c in all_metrics]).mean(),
 'quantile' : np.array([c['local_quantiles'] for c in all_metrics]).mean(),
 'quantile_normalized' : np.array([c['local_quantiles_normalized'] for c in all_metrics]).mean()
}

# COMMAND ----------

n = len(ercot_df_f)
available_points = 1488
prediction_length = 24 
num_samples = 20
context_length = 512
SIZE = "small"
PSZ = "auto"
BSZ = 32  

all_metrics = []

for i in np.linspace(0, n-available_points-prediction_length-200, 20, dtype=int):
    metrics, _, _, local_quantiles, _ = general_pipeline(ercot_df_f[i:], SIZE, PSZ, BSZ, num_samples, available_points, context_length, prediction_length)
    all_metrics.append(metrics)

    metrics_naive, _, _, local_quantiles_naive, _ = general_pipeline_naive(ercot_df_f[['target']][i:], available_points, prediction_length)

    metrics['local_quantiles_normalized'] = local_quantiles/local_quantiles_naive
    metrics['mase'] = metrics['mae']/metrics_naive['mae']


# COMMAND ----------

{'mase' : np.array([c['mase'] for c in all_metrics]).mean(),
 'coverage' : np.array([c['local_coverage'] for c in all_metrics]).mean(),
 'quantile' : np.array([c['local_quantiles'] for c in all_metrics]).mean(),
 'quantile_normalized' : np.array([c['local_quantiles_normalized'] for c in all_metrics]).mean()
}

# COMMAND ----------


