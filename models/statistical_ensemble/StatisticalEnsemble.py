# Databricks notebook source
# MAGIC %md
# MAGIC #Packages

# COMMAND ----------

# MAGIC %sh
# MAGIC pip install --upgrade pip
# MAGIC pip install statsforecast --quiet
# MAGIC pip install xlrd

# COMMAND ----------

import sys
import os
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, AutoCES, DynamicOptimizedTheta
import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm import tqdm
import math
import torch
import plotly.express as px

from utils import rolling_window, general_pipeline

from dataset.ercot import get_ercot
from dataset.nn5_daily import get_nn5_daily
from dataset.nn5_weekly import get_nn5_weekly
from dataset.m3_monthly import get_m3_monthly


sys.path.append(os.path.abspath('..'))

from naive.utils import rolling_window_naive, general_pipeline_naive

%load_ext autoreload
%autoreload 2

# COMMAND ----------

# MAGIC %md
# MAGIC # Ercot

# COMMAND ----------

ercot_df = get_ercot()
ercot_df = ercot_df.rename(columns={'target':'y'})
ercot_df['unique_id'] = 1
ercot_df_f = ercot_df.reset_index(drop=True)
ercot_df_f['d'] = ercot_df_f.index

ercot_df_naive = get_ercot().set_index('ds')
ercot_df_naive = ercot_df_naive.rename(columns={'target':'y'})

# COMMAND ----------

n = len(ercot_df_f)
available_points = 2232
prediction_length = 24


all_metrics = []

for i in tqdm(np.linspace(0, n-available_points-prediction_length-200, 20, dtype=int)[2:3]):

    copy = ercot_df_f[i:i+available_points+prediction_length]
    copy = copy.reset_index(drop=True)
    copy['d'] = copy['d']-i


    metrics, calibration_results, test_results, local_quantiles, global_quantile  = general_pipeline(copy, available_points=available_points, prediction_length=prediction_length, season_length=24, freq = 'H')

    metrics_naive, _, _, local_quantiles_naive, _ = general_pipeline_naive(ercot_df_naive, available_points, prediction_length)

    metrics['local_quantiles_normalized'] = local_quantiles/local_quantiles_naive
    metrics['mase'] = metrics['mae']/metrics_naive['mae']

    all_metrics.append(metrics)

    start_index = i

# COMMAND ----------

{ 'mase' : np.stack([c['mase'] for c in all_metrics]).mean(),
 'quantile' : np.array([c['local_quantiles'] for c in all_metrics]).mean(),
 'coverage' : np.array([c['local_coverage'] for c in all_metrics]).mean(),
 'quantile_normalized' : np.array([c['local_quantiles_normalized'] for c in all_metrics]).mean()
}

# COMMAND ----------

import plotly.graph_objects as go
from plotly.subplots import make_subplots

x_ref = np.arange(2232 + 24)
y_ref = ercot_df_f.y.values[start_index:start_index+2232+24]

# Données de calibration (points aux indices 128:744 de cal_res)
x_cal = np.arange(1785, 2232)
y_cal = calibration_results['y_pred'].values[:447]

# Données de test (points aux indices 744:744+24 de test_res)
x_test = np.arange(2232, 2232 + 24)
y_test = test_results['y_pred'].values

lower_bound = y_test - local_quantiles
upper_bound = y_test + local_quantiles

fig = make_subplots(rows=1, cols=2)

# First subplot
fig.add_trace(go.Scatter(x=x_ref, y=y_ref, mode='lines', name='Truth', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=x_cal, y=y_cal, mode='lines', name='Calibration predictions', line=dict(color='green')), row=1, col=1)
fig.add_trace(go.Scatter(x=x_test, y=y_test, mode='lines', name='Test predictions', line=dict(color='red')), row=1, col=1)
fig.add_trace(go.Scatter(x=[2232, 2232], y=[20000, 70000], mode='lines', line=dict(color='black', dash='dot', width=1), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=[1785, 1785], y=[20000, 70000], mode='lines', line=dict(color='black', dash='dot', width=1), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(
    x=np.concatenate([x_test, x_test[::-1]]), 
    y=np.concatenate([upper_bound, lower_bound[::-1]]),
    fill='toself',
    fillcolor='rgba(255, 0, 0, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    name='Prediction Interval',
    showlegend=True
), row=1, col=1)

zone_labels = ['Train', 'Calibration', 'Test']
zone_positions = [1785/2, (1785+2232)/2, (2232+2232+24)/2]
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
fig.add_trace(go.Scatter(x=[1785, 1785], y=[20000, 70000], mode='lines', line=dict(color='black', dash='dot', width=1), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(
    x=np.concatenate([x_test, x_test[::-1]]), 
    y=np.concatenate([upper_bound, lower_bound[::-1]]),
    fill='toself',
    fillcolor='rgba(255, 0, 0, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    name='Prediction Interval',
    showlegend=False  # Disable legend for the second plot
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
    title='Split conformal prediction : StatisticalEnsemble_light',
    xaxis_title='Time index',
    # height=500,  # Height of the entire figure
    # width=1500,    # Width of the entire figure
    # yaxis_range=[20000, 40000]
)

# fig['layout']['xaxis2']['title']='Time index'
# fig['layout']['xaxis2']['range']=[600, 744+24]
# fig['layout']['yaxis2']['range']=[20000, 40000]

fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # NN5 Daily

# COMMAND ----------

df_nn5=get_nn5_daily()
df_nn5_naive = df_nn5.copy()
df_nn5 = df_nn5.melt()
df_nn5 = df_nn5.rename(columns = {'variable' : 'unique_id', 'value' : 'y'})

n = len(df_nn5_naive)

for i in range(100):
    df_nn5.loc[df_nn5.unique_id ==i, 'ds'] = pd.date_range(start='2010-01-01', periods = n, freq='D')

df_nn5['d'] = df_nn5.index%n
df_nn5['y'] = df_nn5['y'].astype('float64')

# COMMAND ----------

prediction_length=7
available_points = n-prediction_length

metrics, calibration_results, test_results, local_quantiles, global_quantile = general_pipeline(df_nn5, available_points=available_points, prediction_length=prediction_length, season_length=7, freq='D')

metrics_naive, _, _, local_quantiles_naive, _ = general_pipeline_naive(df_nn5_naive, available_points=available_points, prediction_length=prediction_length)

metrics['local_quantiles_normalized'] = np.mean(local_quantiles/local_quantiles_naive)
metrics['global_quantiles_normalized'] = global_quantile.item()/metrics_naive['global_quantiles']
metrics['mase'] = metrics['mae']/metrics_naive['mae']

metrics

# COMMAND ----------

import plotly.graph_objects as go
from plotly.subplots import make_subplots

ts = 0

x_ref = np.arange(len(df_nn5_naive))
y_ref = df_nn5_naive[ts].values

x_cal = np.arange(0.8*(len(df_nn5_naive) - prediction_length), len(df_nn5_naive) - prediction_length)
y_cal = calibration_results[calibration_results.unique_id == ts]['y_pred']

x_test = np.arange(len(df_nn5_naive) - prediction_length,  len(df_nn5_naive))
y_test = test_results[test_results.unique_id == ts]['y_pred']

lower_bound = y_test - local_quantiles[ts]
upper_bound = y_test + local_quantiles[ts]

fig = make_subplots(rows=1, cols=2)

fig.add_trace(go.Scatter(x=x_ref, y=y_ref, mode='lines', name='Truth', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=x_cal, y=y_cal, mode='lines', name='Calibration predictions', line=dict(color='green')), row=1, col=1)
fig.add_trace(go.Scatter(x=x_test, y=y_test, mode='lines', name='Test predictions', line=dict(color='red')), row=1, col=1)
fig.add_trace(go.Scatter(x=[len(x_ref) - prediction_length, len(x_ref) - prediction_length], y=[0, 100], mode='lines', line=dict(color='black', dash='dot', width=1), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=[x_cal[0], x_cal[0]], y=[0, 100], mode='lines', line=dict(color='black', dash='dot', width=1), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(
    x=np.concatenate([x_test, x_test[::-1]]), 
    y=np.concatenate([upper_bound, lower_bound[::-1]]),
    fill='toself',
    fillcolor='rgba(255, 0, 0, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    name='Prediction Interval',
    showlegend=True
), row=1, col=1)

zone_labels = ['Train', 'Calibration', 'Test']
zone_positions = [x_cal[0]/2, (x_cal[0]+len(x_ref))/2, (len(df_nn5_naive)+len(x_ref))/2-2]
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

fig.add_trace(go.Scatter(x=x_ref, y=y_ref, mode='lines', name='Truth', line=dict(color='blue'), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=x_cal, y=y_cal, mode='lines', name='Calibration predictions', line=dict(color='green'), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=x_test, y=y_test, mode='lines', name='Test predictions', line=dict(color='red'), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=[len(x_ref) - prediction_length, len(x_ref) - prediction_length], y=[0, 100], mode='lines', line=dict(color='black', dash='dot', width=1), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=[x_cal[0], x_cal[0]], y=[0, 100], mode='lines', line=dict(color='black', dash='dot', width=1), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(
    x=np.concatenate([x_test, x_test[::-1]]), 
    y=np.concatenate([upper_bound, lower_bound[::-1]]),
    fill='toself',
    fillcolor='rgba(255, 0, 0, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    name='Prediction Interval',
    showlegend=False  # Disable legend for the second plot
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

fig.update_layout(
    title='Split conformal prediction : StatisticalEnsemble_light',
    xaxis_title='Time index',
    # height=500,  # Height of the entire figure
    # width=1500,    # Width of the entire figure
    # yaxis_range=[20000, 40000]
)

# fig['layout']['xaxis2']['title']='Time index'
# fig['layout']['xaxis2']['range']=[600, 744+24]
# fig['layout']['yaxis2']['range']=[20000, 40000]

fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # NN5 Weekly

# COMMAND ----------

df = get_nn5_weekly()
df_naive = df.copy()
df = df.melt()
df = df.rename(columns = {'variable' : 'unique_id', 'value' : 'y'})

n = len(df_naive)

for i in range(100):
    df.loc[df.unique_id ==i, 'ds'] = pd.date_range(start='2010-01-01', periods = n, freq='W')

df['d'] = df.index%n
df['y'] = df['y'].astype('float64')

# COMMAND ----------

prediction_length=4
available_points = n-prediction_length

metrics, calibration_results, test_results, local_quantiles, global_quantile = general_pipeline(df, available_points=available_points, prediction_length=prediction_length, season_length=4, freq='W')

metrics_naive, _, _, local_quantiles_naive, _ = general_pipeline_naive(df_naive, available_points=available_points, prediction_length=prediction_length)

metrics['local_quantiles_normalized'] = np.mean(local_quantiles/local_quantiles_naive)
metrics['global_quantiles_normalized'] = global_quantile.item()/metrics_naive['global_quantiles']
metrics['mase'] = metrics['mae']/metrics_naive['mae']

metrics

# COMMAND ----------

import plotly.graph_objects as go
from plotly.subplots import make_subplots

ts = 0

x_ref = np.arange(len(df_naive))
y_ref = df_naive[ts].values

x_cal = np.arange(0.8*(len(df_naive) - prediction_length), len(df_naive) - prediction_length)
y_cal = calibration_results[calibration_results.unique_id == ts]['y_pred']

x_test = np.arange(len(df_naive) - prediction_length,  len(df_naive))
y_test = test_results[test_results.unique_id == ts]['y_pred']

lower_bound = y_test - local_quantiles[ts]
upper_bound = y_test + local_quantiles[ts]

fig = make_subplots(rows=1, cols=2)

fig.add_trace(go.Scatter(x=x_ref, y=y_ref, mode='lines', name='Truth', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=x_cal, y=y_cal, mode='lines', name='Calibration predictions', line=dict(color='green')), row=1, col=1)
fig.add_trace(go.Scatter(x=x_test, y=y_test, mode='lines', name='Test predictions', line=dict(color='red')), row=1, col=1)
fig.add_trace(go.Scatter(x=[len(x_ref) - prediction_length, len(x_ref) - prediction_length], y=[0, 350], mode='lines', line=dict(color='black', dash='dot', width=1), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=[x_cal[0], x_cal[0]], y=[0, 350], mode='lines', line=dict(color='black', dash='dot', width=1), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(
    x=np.concatenate([x_test, x_test[::-1]]), 
    y=np.concatenate([upper_bound, lower_bound[::-1]]),
    fill='toself',
    fillcolor='rgba(255, 0, 0, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    name='Prediction Interval',
    showlegend=True
), row=1, col=1)

zone_labels = ['Train', 'Calibration', 'Test']
zone_positions = [x_cal[0]/2, (x_cal[0]+len(x_ref))/2, (len(df_naive)+len(x_ref))/2-2]
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

fig.add_trace(go.Scatter(x=x_ref, y=y_ref, mode='lines', name='Truth', line=dict(color='blue'), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=x_cal, y=y_cal, mode='lines', name='Calibration predictions', line=dict(color='green'), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=x_test, y=y_test, mode='lines', name='Test predictions', line=dict(color='red'), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=[len(x_ref) - prediction_length, len(x_ref) - prediction_length], y=[0, 350], mode='lines', line=dict(color='black', dash='dot', width=1), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=[x_cal[0], x_cal[0]], y=[0, 350], mode='lines', line=dict(color='black', dash='dot', width=1), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(
    x=np.concatenate([x_test, x_test[::-1]]), 
    y=np.concatenate([upper_bound, lower_bound[::-1]]),
    fill='toself',
    fillcolor='rgba(255, 0, 0, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    name='Prediction Interval',
    showlegend=False  # Disable legend for the second plot
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

fig.update_layout(
    title='Split conformal prediction : StatisticalEnsemble_light',
    xaxis_title='Time index',
    # height=500,  # Height of the entire figure
    # width=1500,    # Width of the entire figure
    # yaxis_range=[20000, 40000]
)

# fig['layout']['xaxis2']['title']='Time index'
# fig['layout']['xaxis2']['range']=[600, 744+24]
# fig['layout']['yaxis2']['range']=[20000, 40000]

fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # M3M

# COMMAND ----------

df_m3_f = get_m3_monthly()
df_m3_f_naive = df_m3_f.copy()
df_m3_f = df_m3_f.melt()
df_m3_f = df_m3_f.rename(columns = {'variable' : 'unique_id', 'value' : 'y'})

n = len(df_m3_f_naive)

for i in range(df_m3_f.unique_id.nunique()):
    df_m3_f.loc[df_m3_f.unique_id ==i, 'ds'] = pd.date_range(start = '1990-01-01', periods=n, freq = 'M')

df_m3_f['d'] = df_m3_f.index%n
df_m3_f['y'] = df_m3_f['y'].astype('float64')

# COMMAND ----------

prediction_length=3
available_points = n-prediction_length

metrics, calibration_results, test_results, local_quantiles, global_quantile = general_pipeline(df_m3_f, available_points=available_points, prediction_length=prediction_length, season_length=12, freq='M')

metrics_naive, _, _, local_quantiles_naive, _ = general_pipeline_naive(df_m3_f_naive, available_points=available_points, prediction_length=prediction_length)

metrics['local_quantiles_normalized'] = np.mean(local_quantiles/local_quantiles_naive)
metrics['global_quantiles_normalized'] = global_quantile.item()/metrics_naive['global_quantiles']
metrics['mase'] = metrics['mae']/metrics_naive['mae']

metrics

# COMMAND ----------

import plotly.graph_objects as go
from plotly.subplots import make_subplots

ts = 1

x_ref = np.arange(len(df_m3_f_naive))
y_ref = df_m3_f_naive[ts].values

x_cal = np.arange(0.8*(len(df_m3_f_naive) - prediction_length), len(df_m3_f_naive) - prediction_length)
y_cal = calibration_results[calibration_results.unique_id == ts]['y_pred']

x_test = np.arange(len(df_m3_f_naive) - prediction_length,  len(df_m3_f_naive))
y_test = test_results[test_results.unique_id == ts]['y_pred']

lower_bound = y_test - local_quantiles[ts]
upper_bound = y_test + local_quantiles[ts]

fig = make_subplots(rows=1, cols=2)

fig.add_trace(go.Scatter(x=x_ref, y=y_ref, mode='lines', name='Truth', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=x_cal, y=y_cal, mode='lines', name='Calibration predictions', line=dict(color='green')), row=1, col=1)
fig.add_trace(go.Scatter(x=x_test, y=y_test, mode='lines', name='Test predictions', line=dict(color='red')), row=1, col=1)
fig.add_trace(go.Scatter(x=[len(x_ref) - prediction_length, len(x_ref) - prediction_length], y=[0, 8000], mode='lines', line=dict(color='black', dash='dot', width=1), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=[x_cal[0], x_cal[0]], y=[0, 8000], mode='lines', line=dict(color='black', dash='dot', width=1), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(
    x=np.concatenate([x_test, x_test[::-1]]), 
    y=np.concatenate([upper_bound, lower_bound[::-1]]),
    fill='toself',
    fillcolor='rgba(255, 0, 0, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    name='Prediction Interval',
    showlegend=True
), row=1, col=1)

zone_labels = ['Train', 'Calibration', 'Test']
zone_positions = [x_cal[0]/2, (x_cal[0]+len(x_ref))/2, (len(df_m3_f_naive)+len(x_ref))/2-2]
for i, label in enumerate(zone_labels):
    fig.add_annotation(
        x=zone_positions[i],
        y=50,
        text=label,
        showarrow=False,
        yshift=10,
        row=1,
        col=1
    )

fig.add_trace(go.Scatter(x=x_ref, y=y_ref, mode='lines', name='Truth', line=dict(color='blue'), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=x_cal, y=y_cal, mode='lines', name='Calibration predictions', line=dict(color='green'), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=x_test, y=y_test, mode='lines', name='Test predictions', line=dict(color='red'), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=[len(x_ref) - prediction_length, len(x_ref) - prediction_length], y=[0, 8000], mode='lines', line=dict(color='black', dash='dot', width=1), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=[x_cal[0], x_cal[0]], y=[0, 8000], mode='lines', line=dict(color='black', dash='dot', width=1), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(
    x=np.concatenate([x_test, x_test[::-1]]), 
    y=np.concatenate([upper_bound, lower_bound[::-1]]),
    fill='toself',
    fillcolor='rgba(255, 0, 0, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    name='Prediction Interval',
    showlegend=False  # Disable legend for the second plot
), row=1, col=2)

for i, label in enumerate(zone_labels):
    fig.add_annotation(
        x=zone_positions[i],
        y=50,
        text=label,
        showarrow=False,
        yshift=10,
        row=1,
        col=2
    )

fig.update_layout(
    title='Split conformal prediction : StatisticalEnsemble_light',
    xaxis_title='Time index',
    # height=500,  # Height of the entire figure
    # width=1500,    # Width of the entire figure
    # yaxis_range=[20000, 40000]
)

# fig['layout']['xaxis2']['title']='Time index'
# fig['layout']['xaxis2']['range']=[600, 744+24]
# fig['layout']['yaxis2']['range']=[20000, 40000]

fig.show()

# COMMAND ----------


