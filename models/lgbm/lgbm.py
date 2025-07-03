# Databricks notebook source
# MAGIC %sh
# MAGIC pip install --upgrade pip
# MAGIC pip install xlrd

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder
import math
import torch
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objs as go

import logging
logging.getLogger('py4j').setLevel(logging.ERROR)
np.random.seed(0)

import sys
import os

sys.path.append(os.path.abspath('..'))

from naive.utils import rolling_window_naive, general_pipeline_naive
from dataset.ercot import get_ercot
from dataset.nn5_daily import get_nn5_daily
from dataset.nn5_weekly import get_nn5_weekly
from dataset.m3_monthly import get_m3_monthly

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

ercot_df['hour'] = ercot_df['ds'].dt.hour
ercot_df['dow'] = ercot_df['ds'].dt.dayofweek
enc = OneHotEncoder(sparse=False)
ercot_df = ercot_df.join(pd.DataFrame(index = ercot_df.index, data = enc.fit_transform(ercot_df.loc[:, ['dow']]))).drop(columns = 'dow')
ercot_df = ercot_df.reset_index().drop(columns ='index')

h = 24
max_lag = h + (24*7)

for i in range(h, h+24):
    ercot_df[f'lag_{i}'] = ercot_df.target.shift(i)

for i in range(max_lag - 24, max_lag):
    ercot_df[f'lag_{i}'] = ercot_df.target.shift(i)

n = len(ercot_df)

# COMMAND ----------

prediction_length = h
available_points = 2232
available_points_lgb = available_points - max_lag

for rate in [0.8]:

    train_rate = rate
    train_length = int(train_rate*(available_points_lgb))
    calibration_length = available_points-train_length
    train_length = train_length + max_lag

    y = ercot_df['target']
    X = ercot_df.drop(columns = ['target', 'ds'])

    all_metrics = []

    for i in np.linspace(0, n-available_points-prediction_length-200, 20, dtype=int)[2:3]:

        y_train = y[i:i+train_length][max_lag:]
        X_train = X[i:i+train_length][max_lag:]
        y_cal = y[i+train_length:i+available_points]
        X_cal = X[i+train_length:i+available_points]
        X_test = X[i+available_points:i+available_points+prediction_length]
        y_test = y[i+available_points:i+available_points+prediction_length]
        lgbm  = LGBMRegressor(verbose=-1)
        lgbm.fit(X_train, y_train)
        y_cal_pred = lgbm.predict(X_cal)
        
        residus = np.abs(y_cal - y_cal_pred)
        y_test_pred = lgbm.predict(X_test)

        quantile = np.quantile(residus, math.ceil((calibration_length+1)*0.9)/calibration_length)

        
        metrics_naive, _, _, local_quantiles_naive, _ = general_pipeline_naive(ercot_df[['target']][i:], available_points, prediction_length)

        mase = np.mean(np.abs(y_test_pred-y_test))/metrics_naive['mae']
        local_quantiles_normalized = (quantile/local_quantiles_naive)[0]
        coverage = 100*np.mean(y_test-y_test_pred < quantile)
        all_metrics.append([mase, coverage, quantile, local_quantiles_normalized])

        start_index = i


    print(y_train.shape, y_cal.shape, y_test.shape)
    print(rate, dict(zip(['mase', 'coverage', 'quantile', 'quantile_normalized'], np.stack(all_metrics).mean(axis=0))))


# COMMAND ----------

from plotly.subplots import make_subplots

x_ref = np.arange(2232 + 24)
# y_ref = y.values[:2232 + 24]
y_ref = y.values[start_index:start_index+2232+24]

# Données de calibration (points aux indices 128:744 de cal_res)
x_cal = np.arange(2232-y_cal_pred.shape[0], 2232)
y_cal = y_cal_pred

# Données de test (points aux indices 744:744+24 de test_res)
x_test = np.arange(2232, 2232 + 24)
y_test = y_test_pred

lower_bound = y_test - quantile
upper_bound = y_test + quantile


fig = make_subplots(rows=1, cols=2)

# First subplot
fig.add_trace(go.Scatter(x=x_ref, y=y_ref, mode='lines', name='Truth', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=x_cal, y=y_cal, mode='lines', name='Calibration predictions', line=dict(color='green')), row=1, col=1)
fig.add_trace(go.Scatter(x=x_test, y=y_test, mode='lines', name='Test predictions', line=dict(color='red')), row=1, col=1)
fig.add_trace(go.Scatter(x=[2232, 2232], y=[20000, 70000], mode='lines', line=dict(color='black', dash='dot', width=1), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=[1824, 1824], y=[20000, 70000], mode='lines', line=dict(color='black', dash='dot', width=1), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(
    x=np.concatenate([x_test, x_test[::-1]]), 
    y=np.concatenate([upper_bound, lower_bound[::-1]]),
    fill='toself',
    fillcolor='rgba(255, 0, 0, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    name='Prediction Interval',
    showlegend=True
), row=1, col=1)

zone_labels = ['Context', 'Calibration', 'Test']
zone_positions = [1824/2, (1824+2232)/2, (2232+2232+24)/2]
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
fig.add_trace(go.Scatter(x=[1824, 1824], y=[20000, 70000], mode='lines', line=dict(color='black', dash='dot', width=1), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(
    x=np.concatenate([x_test, x_test[::-1]]), 
    y=np.concatenate([upper_bound, lower_bound[::-1]]),
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
    title='Split conformal prediction : LGBM_80_20',
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
df_nn5_naive = df_nn5.copy()
df_nn5 = df_nn5.melt()

print(df_nn5_naive.shape)
print(df_nn5_naive.reset_index().ds.min(), '---', df_nn5_naive.reset_index().ds.max())

fig = px.line(df_nn5_naive.reset_index(), 
              x='ds', 
              y=0, 
              title='nn5 daily data')
fig.show()

h = 7
max_lag = (h + (4*7))

for i in range(h, h+7):
    df_nn5[f'lag_{i}'] = df_nn5.groupby('variable')['value'].transform(lambda x: x.shift(i))

for i in range(max_lag-7, max_lag):
    df_nn5[f'lag_{i}'] = df_nn5.groupby('variable')['value'].transform(lambda x: x.shift(i))

n = len(df_nn5_naive)
n_item = df_nn5_naive.shape[1]
df_nn5['d'] = df_nn5.index%n
df_nn5 = df_nn5.dropna()

# COMMAND ----------

prediction_length = h
available_points = n - prediction_length
available_points_lgb = available_points - max_lag

# for rate in [0.2, 0.5, 0.8]:
for rate in [0.2]:
    
    train_rate = rate
    train_length = int(train_rate*available_points_lgb)
    calibration_length = available_points_lgb-train_length
    train_length = train_length + max_lag

    df_train = df_nn5[df_nn5.d < train_length]
    df_calib = df_nn5[(df_nn5.d >= train_length) & (df_nn5.d < train_length + calibration_length)]
    df_test = df_nn5[(df_nn5.d >= train_length + calibration_length)]

    y_train = df_train['value']
    X_train = df_train.drop(columns = ['value'], axis=1)
    y_cal = df_calib['value']
    X_cal = df_calib.drop(columns = ['value'], axis=1)
    y_test = df_test['value']
    X_test = df_test.drop(columns = ['value'], axis=1) 
    lgbm  = LGBMRegressor(verbose=-1)
    lgbm.fit(X_train, y_train)
    y_cal_pred = lgbm.predict(X_cal)
    print(y_train.shape, y_cal.shape, y_test.shape)

    residus = np.abs(y_cal-y_cal_pred).values.reshape((n_item, -1))
    y_test_pred = lgbm.predict(X_test)

    n_cal=calibration_length
    local_quantiles = np.quantile(residus, math.ceil((n_cal+1)*0.9)/n_cal, axis=1).reshape((-1, 1))

    mae = np.mean(np.abs(y_test_pred-y_test))
    local_coverage = 100*((y_test-y_test_pred).abs().values.reshape((n_item, -1)) < local_quantiles)

    metrics_naive, _, _, local_quantiles_naive, _ = general_pipeline_naive(df_nn5_naive, available_points, prediction_length)
    local_quantiles_normalized = (local_quantiles/local_quantiles_naive).mean()

    print(rate, dict(zip(['mase', 'mae', 'local_coverage', 'local_quantiles', 'local_quantiles_normalized'], [mae.mean()/metrics_naive['mae'].mean(), mae.mean(), local_coverage.mean(), local_quantiles.mean(), local_quantiles_normalized])))


    residus = np.abs(y_cal-y_cal_pred)
    y_test_pred = lgbm.predict(X_test)

    n_channel = df_train.variable.nunique()

    global_quantile = np.quantile(residus, math.ceil(((n_cal*n_channel)+1)*0.9)/(n_cal*n_channel))

    mae = np.mean(np.abs(y_test_pred-y_test))
    global_coverage = 100*((y_test-y_test_pred).abs() < global_quantile)
    print(rate, dict(zip(['mase', 'mae', 'global_coverage', 'global_quantiles', 'global_quantiles_normalized'], [mae.mean()/metrics_naive['mae'].mean(), mae.mean(), global_coverage.mean(), global_quantile, global_quantile.item()/metrics_naive['global_quantiles']])))

# COMMAND ----------

import plotly.graph_objects as go
from plotly.subplots import make_subplots

ts = 0

x_ref = np.arange(len(df_nn5_naive))
y_ref = df_nn5_naive[0].values

x_cal = np.arange(X_train.d.nunique() + max_lag-1, len(df_nn5_naive) - prediction_length + max_lag-1)
y_cal = y_cal_pred.reshape(df_nn5_naive.shape[1], -1)[ts, :]

x_test = np.arange(len(df_nn5_naive) - prediction_length,  len(df_nn5_naive))
y_test = y_test_pred.reshape(df_nn5_naive.shape[1], -1)[ts, :]

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
    title='Split conformal prediction : LGBM_80_20',
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
# MAGIC
# MAGIC # NN5 Weekly

# COMMAND ----------

df = get_nn5_weekly()
df_naive = df.copy()
df = df.melt()

print(df_naive.shape)
print(df_naive.reset_index().index.min(), '---', df_naive.reset_index().index.max())

fig = px.line(df_naive.reset_index(), 
              x='index', 
              y=0, 
              title='nn5 weekly data')
fig.show()

h = 4
max_lag = h + (2*4)

for i in range(h, h+4):
    df[f'lag_{i}'] = df.groupby('variable')['value'].transform(lambda x: x.shift(i))

for i in range(max_lag-4, max_lag):
    df[f'lag_{i}'] = df.groupby('variable')['value'].transform(lambda x: x.shift(i))

n = len(df_naive)
n_item = df_naive.shape[1]
df['d'] = df.index%n
df = df.dropna()

# COMMAND ----------

prediction_length = h
available_points = n - prediction_length
available_points_lgb = available_points - max_lag


# for rate in [0.2, 0.5, 0.8]:
for rate in [0.8]:

    train_rate = rate
    train_length = int(train_rate*available_points_lgb)
    calibration_length = available_points_lgb-train_length
    train_length = train_length + max_lag

    df_train = df[df.d < train_length]
    df_calib = df[(df.d >= train_length) & (df.d < train_length + calibration_length)]
    df_test = df[df.d >= train_length + calibration_length]

    y_train = df_train['value']
    X_train = df_train.drop(columns = ['value'], axis=1)
    y_cal = df_calib['value']
    X_cal = df_calib.drop(columns = ['value'], axis=1)
    y_test = df_test['value']
    X_test = df_test.drop(columns = ['value'], axis=1) 
    lgbm  = LGBMRegressor(verbose=-1)
    lgbm.fit(X_train, y_train)
    y_cal_pred = lgbm.predict(X_cal)
    residus = np.abs(y_cal-y_cal_pred)
    y_test_pred = lgbm.predict(X_test)
    print(y_train.shape, y_cal.shape, y_test.shape)


    n_cal=calibration_length
    local_quantiles = np.quantile(residus.values.reshape(-1, n_item), math.ceil((n_cal+1)*0.9)/n_cal, axis=0)

    metrics_naive, _, _, local_quantiles_naive, _ = general_pipeline_naive(df_naive, available_points, prediction_length)
    local_quantiles_normalized = (local_quantiles/local_quantiles_naive).mean()


    mae = np.mean(np.abs(y_test_pred-y_test))
    coverage = 100*np.mean((y_test-y_test_pred).values.reshape((-1, n_item)) < local_quantiles)
    print(train_rate, dict(zip(['mase', 'mae', 'local_coverage', 'local_quantiles', 'local_quantiles_normalized'], [mae.mean()/metrics_naive['mae'].mean(), mae.mean(), coverage.mean(), local_quantiles.mean(), local_quantiles_normalized])))

    n_channel = df_train.variable.nunique()
    global_quantile = np.quantile(residus.values.ravel(), math.ceil(((n_cal*n_channel)+1)*0.9)/(n_cal*n_channel))

    global_coverage = 100*np.mean((y_test-y_test_pred) < global_quantile)
    print(rate, dict(zip(['mase', 'mae', 'global_coverage', 'global_quantiles', 'global_quantiles_normalized'], [mae.mean()/metrics_naive['mae'].mean(), mae.mean(), global_coverage.mean(), global_quantile, global_quantile.item()/metrics_naive['global_quantiles']])))

# COMMAND ----------

import plotly.graph_objects as go
from plotly.subplots import make_subplots

ts = 0

x_ref = np.arange(len(df_naive))
y_ref = df_naive[0].values

x_cal = np.arange(X_train.d.nunique() + max_lag-1, len(df_naive) - prediction_length + max_lag-1)
y_cal = y_cal_pred.reshape(df_naive.shape[1], -1)[ts, :]

x_test = np.arange(len(df_naive) - prediction_length,  len(df_naive))
y_test = y_test_pred.reshape(df_naive.shape[1], -1)[ts, :]

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
    title='Split conformal prediction : LGBM_80_20',
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
# MAGIC
# MAGIC # M3M

# COMMAND ----------

df_m3_f = get_m3_monthly()
df_m3_f_naive = df_m3_f.copy()
df_m3_f = df_m3_f.melt()

print(df_m3_f_naive.shape)
print(df_m3_f_naive.reset_index().index.min(), '---', df_m3_f_naive.reset_index().index.max())

fig = px.line(df_m3_f_naive.reset_index(), 
              x='index', 
              y=0, 
              title='m3 monthly data')
fig.show()

h = 3
max_lag = h+6

for i in range(max_lag-6, max_lag):
    df_m3_f[f'lag_{i}'] = df_m3_f.groupby('variable')['value'].transform(lambda x: x.shift(i))

n = len(df_m3_f_naive)
n_item = df_m3_f_naive.shape[1]
df_m3_f['d'] = df_m3_f.index%n
df_m3_f = df_m3_f.dropna()

# COMMAND ----------

prediction_length = h
available_points = n - prediction_length
available_points_lgb = available_points - max_lag

# for rate in [0.2, 0.5, 0.8]:
for rate in [0.8]:

    train_rate = rate
    train_length = int(train_rate*available_points_lgb)
    calibration_length = available_points_lgb-train_length
    train_length = train_length + max_lag

    df_train = df_m3_f[df_m3_f.d < train_length]
    df_calib = df_m3_f[(df_m3_f.d >= train_length) & (df_m3_f.d < train_length + calibration_length)]
    df_test = df_m3_f[df_m3_f.d >= train_length + calibration_length]

    n_cal=calibration_length
    y_train = df_train['value']
    X_train = df_train.drop(columns = ['value'], axis=1)
    y_cal = df_calib['value']
    X_cal = df_calib.drop(columns = ['value'], axis=1)
    y_test = df_test['value']
    X_test = df_test.drop(columns = ['value'], axis=1) 
    lgbm  = LGBMRegressor(verbose=-1)
    lgbm.fit(X_train, y_train)
    y_cal_pred = lgbm.predict(X_cal)
    print(y_train.shape, y_cal.shape, y_test.shape)

    residus = np.abs(y_cal-y_cal_pred)
    y_test_pred = lgbm.predict(X_test)

    local_quantiles = np.quantile(residus.values.reshape((n_item, -1)), math.ceil((n_cal+1)*0.9)/n_cal, axis=1).reshape((-1, 1))

    metrics_naive, _, _, local_quantiles_naive, _ = general_pipeline_naive(df_m3_f_naive, available_points, prediction_length)
    local_quantiles_normalized = (local_quantiles/local_quantiles_naive).mean()

    mae = np.mean(np.abs(y_test_pred-y_test))
    coverage = 100*np.mean(np.abs(y_test-y_test_pred).values.reshape((n_item, -1)) < local_quantiles)
    print(rate, dict(zip(['mase', 'mae', 'local_coverage', 'local_quantiles', 'local_quantiles_normalized'], [mae.mean()/metrics_naive['mae'].mean(), mae.mean(), coverage.mean(), local_quantiles.mean(), local_quantiles_normalized])))

    residus = np.abs(y_cal-y_cal_pred)
    y_test_pred = lgbm.predict(X_test)

    n_channel = df_train.variable.nunique()

    global_quantile = np.quantile(residus, math.ceil(((n_cal*n_channel)+1)*0.9)/(n_cal*n_channel))

    mae = np.mean(np.abs(y_test_pred-y_test))
    global_coverage = 100*np.mean(np.abs(y_test-y_test_pred) < global_quantile)
    print(rate, dict(zip(['mase', 'mae', 'global_coverage', 'global_quantiles', 'global_quantiles_normalized'], [mae.mean()/metrics_naive['mae'].mean(), mae.mean(), global_coverage.mean(), global_quantile, global_quantile.item()/metrics_naive['global_quantiles']])))

# COMMAND ----------

import plotly.graph_objects as go
from plotly.subplots import make_subplots

ts = 1

x_ref = np.arange(len(df_m3_f_naive))
y_ref = df_m3_f_naive[1].values

# Données de calibration (points aux indices 128:744 de cal_res)
x_cal = np.arange(X_train.d.nunique() + max_lag-1, len(df_m3_f_naive) - prediction_length + max_lag-1)
y_cal = y_cal_pred.reshape(df_m3_f_naive.shape[1], -1)[ts, :]

# Données de test (points aux indices 744:744+24 de test_res)
x_test = np.arange(len(df_m3_f_naive) - prediction_length,  len(df_m3_f_naive))
y_test = y_test_pred.reshape(df_m3_f_naive.shape[1], -1)[ts, :]

lower_bound = y_test - local_quantiles[ts]
upper_bound = y_test + local_quantiles[ts]

fig = make_subplots(rows=1, cols=2)

# First subplot
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

# Second subplot (Clone of the first)
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

# Update layout for subplot size
fig.update_layout(
    title='Split conformal prediction : LGBM_80_20',
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


