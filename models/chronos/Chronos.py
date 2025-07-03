# Databricks notebook source
# MAGIC %md
# MAGIC # Packages

# COMMAND ----------

# MAGIC %pip install chronos-forecasting
# MAGIC %pip install xlrd

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm

import sys
import os

import plotly.express as px
import plotly.graph_objects as go
import torch
import time
import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

from utils import rolling_window, general_pipeline
from dataset.ercot import get_ercot
from dataset.nn5_daily import get_nn5_daily
from dataset.nn5_weekly import get_nn5_weekly
from dataset.m3_monthly import get_m3_monthly


sys.path.append(os.path.abspath('..'))

from naive.utils import rolling_window_naive, general_pipeline_naive

torch.manual_seed(0)
np.random.seed(0)

CHRONOS_VERSION = 'bolt' # bolt or 1

%load_ext autoreload
%autoreload 2

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Ercot

# COMMAND ----------

ercot_df = get_ercot(sample=True)
print(ercot_df.shape)
print(ercot_df.ds.min(), '---', ercot_df.ds.max())

fig = px.line(ercot_df, 
              x='ds', 
              y='target', 
              title='ercot data')
fig.show()

ercot_tensor = torch.from_numpy(ercot_df['target'].values).unsqueeze(dim=0)

ercot_df_naive = ercot_df[['target']]

# COMMAND ----------

n = ercot_tensor.shape[1]
n_channel = 1
available_points = 2232
prediction_length = 24
num_samples = 20
batch_size = 32
context_length = 512

all_metrics = []
for i in np.linspace(0, n-available_points-prediction_length-200, 20, dtype=int)[2:3]:

    metrics, calibration_results, test_results, local_quantiles, global_quantile = general_pipeline(ercot_tensor[:, i:], available_points, prediction_length, num_samples, batch_size, n_channel, context_length, CHRONOS_VERSION)
    all_metrics.append(metrics)

    metrics_naive, _, _, local_quantiles_naive, _ = general_pipeline_naive(ercot_df_naive[i:], available_points, prediction_length)

    metrics['local_quantiles_normalized'] = local_quantiles/local_quantiles_naive
    metrics['mase'] = metrics['mae']/metrics_naive['mae']

    start_index = i

# COMMAND ----------

{ 'mase' : np.stack([c['mase'] for c in all_metrics]).mean(),
 'quantile' : np.stack([c['local_quantiles'] for c in all_metrics]).mean(),
 'quantile_normalized' : np.stack([c['local_quantiles_normalized'] for c in all_metrics]).mean(),
 'coverage' : np.stack([c['local_coverage'] for c in all_metrics]).mean()
}

# COMMAND ----------

from plotly.subplots import make_subplots
x_ref = np.arange(2232 + 24)
# y_ref = ercot_tensor[0, :2232 + 24]
y_ref = ercot_tensor[0, start_index:start_index+2232+24]

x_cal = np.arange(512, 2232)
y_cal = calibration_results[0]

x_test = np.arange(2232, 2232 + 24)
y_test = test_results[0, :24]

lower_bound = y_test - local_quantiles
upper_bound = y_test + local_quantiles

fig = make_subplots(rows=1, cols=2)

# First subplot
fig.add_trace(go.Scatter(x=x_ref, y=y_ref, mode='lines', name='Truth', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=x_cal, y=y_cal, mode='lines', name='Calibration predictions', line=dict(color='green')), row=1, col=1)
fig.add_trace(go.Scatter(x=x_test, y=y_test, mode='lines', name='Test predictions', line=dict(color='red')), row=1, col=1)
fig.add_trace(go.Scatter(x=[2232, 2232], y=[20000, 70000], mode='lines', line=dict(color='black', dash='dot', width=1), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=[512, 512], y=[20000, 70000], mode='lines', line=dict(color='black', dash='dot', width=1), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(
    x=np.concatenate([x_test, x_test[::-1]]), 
    y=np.concatenate([upper_bound.numpy(), lower_bound.numpy()[::-1]]),
    fill='toself',
    fillcolor='rgba(255, 0, 0, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    name='Prediction Interval',
    showlegend=True
), row=1, col=1)

zone_labels = ['Context', 'Calibration', 'Test']
zone_positions = [512/2, (512+2232)/2, (2232+2232+24)/2]
for i, label in enumerate(zone_labels):
    fig.add_annotation(
        x=zone_positions[i],
        y=20000,
        text=label,
        showarrow=False,
        yshift=10,
        row=1,
        col=1
    )

# Second subplot (Clone of the first)
fig.add_trace(go.Scatter(x=x_ref, y=y_ref, mode='lines', name='Truth', line=dict(color='blue'), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=x_cal, y=y_cal, mode='lines', name='Calibration predictions', line=dict(color='green'), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=x_test, y=y_test, mode='lines', name='Test predictions', line=dict(color='red'), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=[2232, 2232], y=[20000, 70000], mode='lines', line=dict(color='black', dash='dot', width=1), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=[512, 512], y=[20000, 70000], mode='lines', line=dict(color='black', dash='dot', width=1), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(
    x=np.concatenate([x_test, x_test[::-1]]), 
    y=np.concatenate([upper_bound.numpy(), lower_bound.numpy()[::-1]]),
    fill='toself',
    fillcolor='rgba(255, 0, 0, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    name='Prediction Interval',
    showlegend=False 
), row=1, col=2)

for i, label in enumerate(zone_labels):
    fig.add_annotation(
        x=zone_positions[i],
        y=20000,
        text=label,
        showarrow=False,
        yshift=10,
        row=1,
        col=2
    )

# Update layout for subplot size
fig.update_layout(
    title='Split conformal prediction : ChronosBolt',
    xaxis_title='Time index',
    # height=500,  
    # width=1500,    
    # yaxis_range=[20000, 40000]
)

# fig['layout']['xaxis2']['title']='Time index'
# fig['layout']['xaxis2']['range']=[600, 2232+24]
# fig['layout']['yaxis2']['range']=[20000, 40000]

fig.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # NN5 Daily

# COMMAND ----------

df_nn5 = get_nn5_daily()
df_nn5_tensor = torch.tensor(df_nn5.astype('float64').T.to_numpy())

# COMMAND ----------

n_channel = df_nn5_tensor.shape[0]
prediction_length = 7
available_points = df_nn5_tensor.shape[1] - prediction_length
num_samples = 20
batch_size = 32
context_length = 128


metrics, calibration_results, test_results, local_quantiles, global_quantile = general_pipeline(df_nn5_tensor, available_points, prediction_length, num_samples, batch_size, n_channel, context_length, CHRONOS_VERSION)

metrics_naive, _, _, local_quantiles_naive, _ = general_pipeline_naive(df_nn5, available_points, prediction_length)
metrics['local_quantiles_normalized'] = (local_quantiles/local_quantiles_naive).mean().item()
metrics['global_quantiles_normalized'] = global_quantile.item()/metrics_naive['global_quantiles']
metrics['mase'] = metrics['mae']/metrics_naive['mae']

metrics

# COMMAND ----------

from plotly.subplots import make_subplots
x_ref = np.arange(len(df_nn5))
y_ref = df_nn5[0]

x_cal = np.arange(context_length, len(df_nn5) - prediction_length)
y_cal = calibration_results[0]

x_test = np.arange(len(df_nn5) - prediction_length, len(df_nn5))
y_test = test_results[0]

lower_bound = y_test - local_quantiles[0]
upper_bound = y_test + local_quantiles[0]

fig = make_subplots(rows=1, cols=2)

# First subplot
fig.add_trace(go.Scatter(x=x_ref, y=y_ref, mode='lines', name='Truth', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=x_cal, y=y_cal, mode='lines', name='Calibration predictions', line=dict(color='green')), row=1, col=1)
fig.add_trace(go.Scatter(x=x_test, y=y_test, mode='lines', name='Test predictions', line=dict(color='red')), row=1, col=1)
fig.add_trace(go.Scatter(x=[len(x_ref) - prediction_length, len(x_ref) - prediction_length], y=[0, 100], mode='lines', line=dict(color='black', dash='dot', width=1), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=[x_cal[0], x_cal[0]], y=[0, 100], mode='lines', line=dict(color='black', dash='dot', width=1), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(
    x=np.concatenate([x_test, x_test[::-1]]), 
    y=np.concatenate([upper_bound.numpy(), lower_bound.numpy()[::-1]]),
    fill='toself',
    fillcolor='rgba(255, 0, 0, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    name='Prediction Interval',
    showlegend=True
), row=1, col=1)

zone_labels = ['Context', 'Calibration', 'Test']
zone_positions = [x_cal[0]/2, (x_cal[0]+len(x_ref))/2, (len(df_nn5)+len(x_ref))/2-3]
for i, label in enumerate(zone_labels):
    fig.add_annotation(
        x=zone_positions[i],
        y=0,
        text=label,
        showarrow=False,
        yshift=10,
        row=1,
        col=1
    )

# Second subplot (Clone of the first)
fig.add_trace(go.Scatter(x=x_ref, y=y_ref, mode='lines', name='Truth', line=dict(color='blue'), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=x_cal, y=y_cal, mode='lines', name='Calibration predictions', line=dict(color='green'), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=x_test, y=y_test, mode='lines', name='Test predictions', line=dict(color='red'), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=[len(x_ref) - prediction_length, len(x_ref) - prediction_length], y=[0, 100], mode='lines', line=dict(color='black', dash='dot', width=1), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=[x_cal[0], x_cal[0]], y=[0, 100], mode='lines', line=dict(color='black', dash='dot', width=1), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(
    x=np.concatenate([x_test, x_test[::-1]]), 
    y=np.concatenate([upper_bound.numpy(), lower_bound.numpy()[::-1]]),
    fill='toself',
    fillcolor='rgba(255, 0, 0, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    name='Prediction Interval',
    showlegend=False 
), row=1, col=2)

for i, label in enumerate(zone_labels):
    fig.add_annotation(
        x=zone_positions[i],
        y=0,
        text=label,
        showarrow=False,
        yshift=10,
        row=1,
        col=2
    )

# Update layout for subplot size
fig.update_layout(
    title='Split conformal prediction : ChronosBolt',
    xaxis_title='Time index',
    # height=500,  
    # width=1500,    
    # yaxis_range=[20000, 40000]
)

# fig['layout']['xaxis2']['title']='Time index'
# fig['layout']['xaxis2']['range']=[600, 2232+24]
# fig['layout']['yaxis2']['range']=[20000, 40000]

fig.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # NN5 Weekly

# COMMAND ----------

n_channel = nn5_tensor_weekly.shape[0]
prediction_length = 4
available_points = nn5_tensor_weekly.shape[1] - prediction_length
num_samples = 20
batch_size = 32
context_length = 64


metrics, calibration_results, test_results, local_quantiles, global_quantile = general_pipeline(nn5_tensor_weekly, available_points, prediction_length, num_samples, batch_size, n_channel, context_length, CHRONOS_VERSION)

metrics_naive, _, _, local_quantiles_naive, _ = general_pipeline_naive(df_nn5_weekly, available_points, prediction_length)
metrics['local_quantiles_normalized'] = (local_quantiles/local_quantiles_naive).mean().item()
metrics['global_quantiles_normalized'] = global_quantile.item()/metrics_naive['global_quantiles']
metrics['mase'] = metrics['mae']/metrics_naive['mae']

metrics

# COMMAND ----------

from plotly.subplots import make_subplots
x_ref = np.arange(len(df_nn5_weekly))
y_ref = df_nn5_weekly[0]

x_cal = np.arange(context_length, len(df_nn5_weekly) - prediction_length)
y_cal = calibration_results[0]

x_test = np.arange(len(df_nn5_weekly) - prediction_length, len(df_nn5_weekly))
y_test = test_results[0]

lower_bound = y_test - local_quantiles[0]
upper_bound = y_test + local_quantiles[0]

fig = make_subplots(rows=1, cols=2)

# First subplot
fig.add_trace(go.Scatter(x=x_ref, y=y_ref, mode='lines', name='Truth', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=x_cal, y=y_cal, mode='lines', name='Calibration predictions', line=dict(color='green')), row=1, col=1)
fig.add_trace(go.Scatter(x=x_test, y=y_test, mode='lines', name='Test predictions', line=dict(color='red')), row=1, col=1)
fig.add_trace(go.Scatter(x=[len(x_ref) - prediction_length, len(x_ref) - prediction_length], y=[0, 350], mode='lines', line=dict(color='black', dash='dot', width=1), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=[x_cal[0], x_cal[0]], y=[0, 350], mode='lines', line=dict(color='black', dash='dot', width=1), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(
    x=np.concatenate([x_test, x_test[::-1]]), 
    y=np.concatenate([upper_bound.numpy(), lower_bound.numpy()[::-1]]),
    fill='toself',
    fillcolor='rgba(255, 0, 0, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    name='Prediction Interval',
    showlegend=True
), row=1, col=1)

zone_labels = ['Context', 'Calibration', 'Test']
zone_positions = [x_cal[0]/2, (x_cal[0]+len(x_ref))/2, (len(df_nn5_weekly)+len(x_ref))/2-3]
for i, label in enumerate(zone_labels):
    fig.add_annotation(
        x=zone_positions[i],
        y=0,
        text=label,
        showarrow=False,
        yshift=10,
        row=1,
        col=1
    )

# Second subplot (Clone of the first)
fig.add_trace(go.Scatter(x=x_ref, y=y_ref, mode='lines', name='Truth', line=dict(color='blue'), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=x_cal, y=y_cal, mode='lines', name='Calibration predictions', line=dict(color='green'), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=x_test, y=y_test, mode='lines', name='Test predictions', line=dict(color='red'), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=[len(x_ref) - prediction_length, len(x_ref) - prediction_length], y=[0, 350], mode='lines', line=dict(color='black', dash='dot', width=1), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=[x_cal[0], x_cal[0]], y=[0, 350], mode='lines', line=dict(color='black', dash='dot', width=1), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(
    x=np.concatenate([x_test, x_test[::-1]]), 
    y=np.concatenate([upper_bound.numpy(), lower_bound.numpy()[::-1]]),
    fill='toself',
    fillcolor='rgba(255, 0, 0, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    name='Prediction Interval',
    showlegend=False 
), row=1, col=2)

for i, label in enumerate(zone_labels):
    fig.add_annotation(
        x=zone_positions[i],
        y=0,
        text=label,
        showarrow=False,
        yshift=10,
        row=1,
        col=2
    )

# Update layout for subplot size
fig.update_layout(
    title='Split conformal prediction : ChronosBolt',
    xaxis_title='Time index',
    # height=500,  
    # width=1500,    
    # yaxis_range=[20000, 40000]
)

# fig['layout']['xaxis2']['title']='Time index'
# fig['layout']['xaxis2']['range']=[600, 2232+24]
# fig['layout']['yaxis2']['range']=[20000, 40000]

fig.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # M3 Monthly

# COMMAND ----------

df_m3_f = get_m3_monthly()
m3_tensor_monthly = torch.from_numpy(df_m3_f.to_numpy().T)

# COMMAND ----------

n_channel = m3_tensor_monthly.shape[0]
prediction_length = 3
available_points = m3_tensor_monthly.shape[1] - prediction_length
context_length = 32
num_samples = 20
batch_size = 32

metrics, calibration_results, test_results, local_quantiles, global_quantile = general_pipeline(m3_tensor_monthly, available_points, prediction_length, num_samples, batch_size, n_channel, context_length, CHRONOS_VERSION)

metrics_naive, _, _, local_quantiles_naive, _ = general_pipeline_naive(df_m3_f, available_points, prediction_length)
metrics['local_quantiles_normalized'] = (local_quantiles/local_quantiles_naive).mean().item()
metrics['global_quantiles_normalized'] = global_quantile.item()/metrics_naive['global_quantiles']
metrics['mase'] = metrics['mae']/metrics_naive['mae']

metrics

# COMMAND ----------

from plotly.subplots import make_subplots
x_ref = np.arange(len(df_m3_f))
y_ref = df_m3_f[1]

x_cal = np.arange(context_length, len(df_m3_f) - prediction_length)
y_cal = calibration_results[1]

x_test = np.arange(len(df_m3_f) - prediction_length, len(df_m3_f))
y_test = test_results[1]

lower_bound = y_test - local_quantiles[1]
lower_bound = torch.tensor([0, 0, 0])
upper_bound = y_test + local_quantiles[1]

fig = make_subplots(rows=1, cols=2)

# First subplot
fig.add_trace(go.Scatter(x=x_ref, y=y_ref, mode='lines', name='Truth', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=x_cal, y=y_cal, mode='lines', name='Calibration predictions', line=dict(color='green')), row=1, col=1)
fig.add_trace(go.Scatter(x=x_test, y=y_test, mode='lines', name='Test predictions', line=dict(color='red')), row=1, col=1)
fig.add_trace(go.Scatter(x=[len(x_ref) - prediction_length, len(x_ref) - prediction_length], y=[0, 8000], mode='lines', line=dict(color='black', dash='dot', width=1), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=[x_cal[0], x_cal[0]], y=[0, 8000], mode='lines', line=dict(color='black', dash='dot', width=1), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(
    x=np.concatenate([x_test, x_test[::-1]]), 
    y=np.concatenate([upper_bound.numpy(), lower_bound.numpy()[::-1]]),
    fill='toself',
    fillcolor='rgba(255, 0, 0, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    name='Prediction Interval',
    showlegend=True
), row=1, col=1)

zone_labels = ['Context', 'Calibration', 'Test']
zone_positions = [x_cal[0]/2, (x_cal[0]+len(x_ref))/2, (len(df_m3_f)+len(x_ref))/2-3]
for i, label in enumerate(zone_labels):
    fig.add_annotation(
        x=zone_positions[i],
        y=0,
        text=label,
        showarrow=False,
        yshift=10,
        row=1,
        col=1
    )

# Second subplot (Clone of the first)
fig.add_trace(go.Scatter(x=x_ref, y=y_ref, mode='lines', name='Truth', line=dict(color='blue'), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=x_cal, y=y_cal, mode='lines', name='Calibration predictions', line=dict(color='green'), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=x_test, y=y_test, mode='lines', name='Test predictions', line=dict(color='red'), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=[len(x_ref) - prediction_length, len(x_ref) - prediction_length], y=[0, 8000], mode='lines', line=dict(color='black', dash='dot', width=1), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=[x_cal[0], x_cal[0]], y=[0, 8000], mode='lines', line=dict(color='black', dash='dot', width=1), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(
    x=np.concatenate([x_test, x_test[::-1]]), 
    y=np.concatenate([upper_bound.numpy(), lower_bound.numpy()[::-1]]),
    fill='toself',
    fillcolor='rgba(255, 0, 0, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    name='Prediction Interval',
    showlegend=False 
), row=1, col=2)

for i, label in enumerate(zone_labels):
    fig.add_annotation(
        x=zone_positions[i],
        y=0,
        text=label,
        showarrow=False,
        yshift=10,
        row=1,
        col=2
    )

# Update layout for subplot size
fig.update_layout(
    title='Split conformal prediction : ChronosBolt',
    xaxis_title='Time index',
    # height=500,  
    # width=1500,    
    # yaxis_range=[20000, 40000]
)

# fig['layout']['xaxis2']['title']='Time index'
# fig['layout']['xaxis2']['range']=[600, 2232+24]
# fig['layout']['yaxis2']['range']=[20000, 40000]

fig.show()

