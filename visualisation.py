# Databricks notebook source
import plotly.express as px
import plotly.graph_objects as go

# COMMAND ----------

colors = [
    "#e6194b",  # rouge vif
    "#fabebe",  # rose clair
    "#42d4f4",  # cyan clair
    "#2a9ec4",  # cyan vif
    "#f58231",  # orange vif
    "#d1a3ff",  # violet clair
    "#911eb4",  # violet foncé
    "#bfef45",  # vert clair
    "#3cb44b",  # vert vif
    "#469990",  # vert sarcelle
    "#8b4513"   # rose vif
]

colors = [
    "#fabebe",  # rose clair
    "#e6194b",  # rouge vif
    "#8b4513",  # rose vif
    "#bfef45",  # vert clair
    "#3cb44b",  # vert vif
    "#469990",  # vert sarcelle
    "#f58231",  # orange vif
    "#42d4f4",  # cyan clair
    "#2a9ec4",  # cyan vif
    "#d1a3ff",  # violet clair
    "#911eb4",  # violet foncé
]

# COMMAND ----------

import itertools
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import numpy as np

import pandas as pd

# Define the data as a list of lists
data = [
    ["Naive", "S", 1.000, 88.1, 16.59, 1.000, 1.000, 88.0, 32.13, 1.000, 1.000, 86.4, 1228, 1.000],
    ["Naive", "M", 1.000, 84.9, 16.45, 1.000, 1.000, 94.6, 39.37, 1.000, 1.000, 86.0, 1512, 1.000],
    ["Naive", "L", 1.000, 86.0, 17.90, 1.000, 1.000, 90.2, 42.82, 1.000, 1.000, 87.4, 1678, 1.000],
    ["SeasonalNaive", "S", 0.455, 96.6, 11.20, 0.696, 0.913, 96.3, 46.22, 1.457, 1.132, 91.7, 1560, 1.436],
    ["SeasonalNaive", "M", 0.605, 90.2, 12.26, 0.766, 1.660, 82.4, 43.86, 1.097, 1.159, 91.0, 1839, 1.396],
    ["SeasonalNaive", "L", 0.525, 92.7, 12.65, 0.725, 1.122, 88.4, 46.39, 1.109, 1.007, 91.4, 1941, 1.324],
    ["Stat.Ensemble_light", "S", 0.365, 98.6, 10.21, 0.636, 0.847, 97.3, 49.30, 1.573, 0.767, 93.3, 1215, 1.168],
    ["Stat.Ensemble_light", "M", 0.508, 94.1, 10.39, 0.669, 0.891, 96.9, 47.14, 1.164, 0.740, 91.1, 1266, 0.963],
    ["Stat.Ensemble_light", "L", 0.430, 93.4, 10.71, 0.611, 1.059, 88.4, 45.01, 1.078, 0.715, 84.5, 1236, 0.804],
    ["LGBM_20_80", "S", 0.440, 96.1, 9.54, 0.660, 1.101, 95.8, 42.16, 1.471, 1.057, 90.9, 1430, 5.202],
    ["LGBM_20_80", "M", 0.505, 93.6, 10.53, 0.728, 1.221, 97.8, 44.73, 1.239, 1.097, 90.8, 1539, 3.448],
    ["LGBM_20_80", "L", 0.506, 92.2, 10.66, 0.680, 0.851, 98.7, 45.35, 1.198, 0.906, 89.9, 1523, 2.607],
    ["LGBM_50_50", "S", 0.440, 96.9, 9.66, 0.668, 0.973, 97.0, 42.90, 1.499, 1.189, 92.0, 1536, 5.588],
    ["LGBM_50_50", "M", 0.496, 94.4, 10.52, 0.727, 1.243, 97.8, 43.17, 1.190, 1.016, 89.6, 1509, 3.379],
    ["LGBM_50_50", "L", 0.537, 91.5, 11.18, 0.713, 0.957, 98.3, 46.70, 1.253, 0.860, 86.1, 1378, 2.359],
    ["LGBM_80_20", "S", 0.420, 98.3, 11.52, 0.797, 1.334, 96.7, 76.94, 2.754, 1.009, 92.0, 1397, 5.080],
    ["LGBM_80_20", "M", 0.466, 98.0, 14.65, 1.012, 1.407, 99.9, 71.50, 1.987, 0.951, 87.3, 1460, 3.270],
    ["LGBM_80_20", "L", 0.478, 94.8, 12.36, 0.789, 0.948, 99.7, 69.47, 1.808, 0.831, 83.3, 1322, 2.263],
    ["Lag-Llama", "S", 0.749, 93.7, 14.55, 0.890, 1.055, 94.8, 41.78, 1.451, 1.500, 90.2, 1901, 6.990],
    ["Lag-Llama", "M", 0.785, 91.8, 15.05, 0.926, 1.119, 95.5, 44.63, 1.110, 1.453, 90.4, 2142, 4.819],
    ["Lag-Llama", "L", 0.725, 92.4, 15.56, 0.880, 0.911, 92.8, 49.65, 1.116, 1.329, 89.1, 2160, 3.696],
    ["Chronos", "S", 0.336, 98.4, 9.61, 0.598, 0.868, 95.5, 41.45, 1.271, 0.800, 92.7, 1210, 1.171],
    ["Chronos", "M", 0.503, 91.5, 10.25, 0.639, 0.410, 84.5, 44.66, 1.206, 0.780, 91.9, 1467, 1.152],
    ["Chronos", "L", 0.491, 91.0, 10.69, 0.611, 0.931, 82.4, 37.96, 0.919, 0.749, 89.3, 1495, 1.003],
    ["ChronosBolt", "S", 0.370, 97.7, 8.68, 0.543, 0.851, 97.3, 43.50, 1.343, 0.808, 92.5, 1162, 1.146],
    ["ChronosBolt", "M", 0.506, 90.2, 9.02, 0.566, 0.445, 79.1, 44.65, 1.193, 0.789, 91.4, 1404, 1.124],
    ["ChronosBolt", "L", 0.422, 92.6, 9.12, 0.527, 0.932, 83.3, 37.40, 0.903, 0.772, 88.8, 1425, 0.973],
    ["TimesFM", "S", 0.340, 97.7, 8.41, 0.525, 0.826, 96.5, 37.81, 1.207, 0.785, 92.4, 1074, 1.056],
    ["TimesFM", "M", 0.519, 88.5, 8.83, 0.553, 0.852, 93.8, 35.73, 0.897, 0.728, 92.2, 1310, 1.035],
    ["TimesFM", "L", 0.413, 92.8, 9.10, 0.525, 0.687, 93.6, 36.26, 0.887, 0.710, 87.8, 1302, 0.876],
    ["TimesFM2", "S", 0.340, 97.7, 8.36, 0.522, 0.826, 95.8, 39.23, 1.259, 0.749, 92.8, 1080, 1.067],
    ["TimesFM2", "M", 0.522, 88.2, 8.74, 0.547, 0.860, 92.9, 35.37, 0.897, 0.720, 92.3, 1312, 1.041],
    ["TimesFM2", "L", 0.419, 92.1, 9.06, 0.521, 0.679, 92.5, 35.51, 0.868, 0.703, 88.2, 1307, 0.886],
]

# Define column names
columns = [
    "Model", "Horizon",
    "NN5 Daily MASE", "NN5 Daily Coverage", "NN5 Daily Interval Width", "NN5 Daily Normalized Interval Width",
    "NN5 Weekly MASE", "NN5 Weekly Coverage", "NN5 Weekly Interval Width", "NN5 Weekly Normalized Interval Width",
    "M3 Monthly MASE", "M3 Monthly Coverage", "M3 Monthly Interval Width", "M3 Monthly Normalized Interval Width"
]

# Create DataFrame
data = pd.DataFrame(data, columns=columns)


l_dataset = ['NN5 Daily', 'NN5 Weekly', 'M3 Monthly']
l_horizon = ['S', 'M', 'L']


for element in itertools.product(*[l_dataset, l_horizon]):

    tmp_data = data.loc[data['Horizon'] == element[1], ['Model'] + [col for col in data.columns if element[0] in col]]

    d_data = {
    "Modèle": tmp_data.Model,
    "MASE": tmp_data[f'{element[0]} MASE'],
    "Coverage quantile local": tmp_data[f'{element[0]} Coverage'] / 100,
    "Moyenne quantile locaux" : tmp_data[f'{element[0]} Normalized Interval Width']
    }

    df = pd.DataFrame(d_data)

    df['Modèle'] = df['Modèle'] + ' (MASE: ' + df['MASE'].astype(str) + ')'

    # Créer un graphique à bulles avec Plotly
    fig = px.scatter(df, x="Coverage quantile local", y="Moyenne quantile locaux",
                    size="MASE", 
                    color="Modèle",
                    color_discrete_sequence=colors,
                    size_max=60,
                    labels={"Coverage quantile local": "MCR",
                            "Moyenne quantile locaux": "MSIW",
                            "MASE": "MASE"})


    fig.add_trace(go.Scatter(
        x=df["Coverage quantile local"],
        y=df["Moyenne quantile locaux"],
        mode='markers',
        marker=dict(
            size=5,
            color=np.array(colors)[:len(df)],  # Use the same color as the bubbles
            line=dict(width=1, color='white')  # Optional: add a black border to the center points
        ),
        showlegend=False  # Do not show in legend
    ))

    y_min = df['Moyenne quantile locaux'].min() - df['Moyenne quantile locaux'].min()*0.2
    y_max = df['Moyenne quantile locaux'].max() + df['Moyenne quantile locaux'].max()*0.2


    fig.add_trace(go.Scatter(x=[0.90, 0.90], 
                             y = [y_min, y_max], 
                             line=dict(color='black', dash='dot', width=.4), 
                             mode='lines',
                             showlegend=False))
    

    fig.update_layout(title = f'{element[0]} - Horizon {element[1]}',
                    showlegend=True)
    
    fig.update_yaxes(range=[y_min, y_max])
    fig.update_xaxes(range=[0.80, 1], tickformat=".0%")
    fig.show()

# COMMAND ----------

import itertools
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import numpy as np

import pandas as pd

data = [
    ["Naive", "S", 1.000, 87.5, 15946, 1.000, 1.000, 87.9, 15160, 1.000],
    ["Naive", "M", 1.000, 86.7, 16610, 1.000, 1.000, 83.7, 16495, 1.000],
    ["Naive", "L", 1.000, 91.8, 17179, 1.000, 1.000, 86.7, 16486, 1.000],
    ["SeasonalNaive", "S", 0.795, 91.7, 5504, 0.461, 1.338, 79.2, 6742, 0.882],
    ["SeasonalNaive", "M", 0.607, 97.2, 9123, 0.698, 0.563, 100.0, 12457, 1.227],
    ["SeasonalNaive", "L", 1.561, 72.0, 9527, 0.652, 0.932, 100.0, 13609, 1.058],
    ["Stat.Ensemble_light", "S", np.nan, np.nan, np.nan, np.nan, 1.018, 87.9, 6119, 0.801],
    ["Stat.Ensemble_light", "M", np.nan, np.nan, np.nan, np.nan, 1.111, 85.3, 8002, 0.788],
    ["Stat.Ensemble_light", "L", np.nan, np.nan, np.nan, np.nan, 1.371, 81.2, 8785, 0.683],
    ["LGBM_20_80", "S", 0.592, 96.7, 10748, 0.708, 1.195, 89.2, 9328, 0.724],
    ["LGBM_20_80", "M", 0.763, 98.0, 18204, 1.129, 1.418, 86.0, 13219, 0.888],
    ["LGBM_20_80", "L", 0.845, 96.7, 17713, 1.034, 1.485, 86.0, 12115, 0.790],
    ["LGBM_50_50", "S", 0.580, 89.8, 7869, 0.511, 0.841, 87.9, 7266, 0.559],
    ["LGBM_50_50", "M", 0.654, 92.7, 13070, 0.808, 1.149, 87.9, 10763, 0.701],
    ["LGBM_50_50", "L", 0.793, 92.3, 12849, 0.774, 1.389, 87.2, 10909, 0.715],
    ["LGBM_80_20", "S", 0.574, 89.0, 5621, 0.369, 0.659, 89.0, 5655, 0.428],
    ["LGBM_80_20", "M", 0.607, 90.3, 8775, 0.553, 0.760, 88.9, 8578, 0.552],
    ["LGBM_80_20", "L", 0.769, 90.2, 9410, 0.557, 0.923, 88.5, 8677, 0.557],
    ["Lag-Llama", "S", 0.646, 89.0, 6976, 0.458, 0.895, 88.7, 7205, 0.555],
    ["Lag-Llama", "M", 0.581, 90.6, 8613, 0.535, 0.618, 96.3, 9140, 0.614],
    ["Lag-Llama", "L", 0.731, 90.2, 9428, 0.562, 0.755, 93.8, 11128, 0.708],
    ["Chronos", "S", 0.522, 82.7, 4396, 0.290, 0.428, 89.0, 4206, 0.334],
    ["Chronos", "M", 0.604, 81.7, 6481, 0.403, 0.475, 86.7, 6266, 0.415],
    ["Chronos", "L", 0.690, 83.9, 7750, 0.462, 0.580, 91.4, 7514, 0.491],
    ["ChronosBolt", "S", 0.430, 86.3, 4113, 0.271, 0.362, 91.3, 3953, 0.312],
    ["ChronosBolt", "M", 0.502, 83.8, 6006, 0.374, 0.417, 88.5, 5954, 0.395],
    ["ChronosBolt", "L", 0.633, 86.3, 7076, 0.423, 0.526, 90.4, 6897, 0.451],
    ["TimesFM", "S", 0.440, 89.8, 4411, 0.291, 0.398, 90.0, 4258, 0.337],
    ["TimesFM", "M", 0.525, 87.4, 6375, 0.397, 0.417, 89.0, 6412, 0.428],
    ["TimesFM", "L", 0.599, 86.8, 7573, 0.452, 0.522, 93.1, 7541, 0.493],
    ["TimesFM2", "S", 0.494, 85.8, 4296, 0.283, 0.416, 89.4, 4108, 0.322],
    ["TimesFM2", "M", 0.500, 83.2, 6110, 0.380, 0.396, 88.5, 6070, 0.400],
    ["TimesFM2", "L", 0.653, 82.8, 7095, 0.424, 0.552, 89.8, 6985, 0.449],
]

columns = [
    "Model", "Horizon",
    "ERCOT 8760 - MASE", "ERCOT 8760 - Coverage", "ERCOT 8760 - Interval Width", "ERCOT 8760 - Normalized Interval Width",
    "ERCOT 2232 - MASE", "ERCOT 2232 - Coverage", "ERCOT 2232 - Interval Width", "ERCOT 2232 - Normalized Interval Width"
]


# Create DataFrame
data = pd.DataFrame(data, columns=columns)


l_dataset = ['ERCOT 8760', 'ERCOT 2232']
l_horizon = ['S', 'M', 'L']


for element in itertools.product(*[l_dataset, l_horizon]):

        tmp_data = data.loc[data['Horizon'] == element[1], ['Model'] + [col for col in data.columns if element[0] in col]]

        if element[0] == 'ERCOT 8760':
            tmp_data = tmp_data[tmp_data.Model != "Stat.Ensemble_light"]
            colors = [
                "#fabebe",  # rose clair
                "#e6194b",  # rouge vif
                "#bfef45",  # vert clair
                "#3cb44b",  # vert vif
                "#469990",  # vert sarcelle
                "#f58231",  # orange vif
                "#42d4f4",  # cyan clair
                "#2a9ec4",  # cyan vif
                "#d1a3ff",  # violet clair
                "#911eb4",  # violet foncé
            ]
        
        else:
            colors = [
                "#fabebe",  # rose clair
                "#e6194b",  # rouge vif
                "#8b4513",  # rose vif
                "#bfef45",  # vert clair
                "#3cb44b",  # vert vif
                "#469990",  # vert sarcelle
                "#f58231",  # orange vif
                "#42d4f4",  # cyan clair
                "#2a9ec4",  # cyan vif
                "#d1a3ff",  # violet clair
                "#911eb4",  # violet foncé
            ]



        d_data = {
        "Modèle": tmp_data.Model,
        "MASE": tmp_data[f'{element[0]} - MASE'],
        "Coverage quantile local": tmp_data[f'{element[0]} - Coverage'] / 100,
        "Moyenne quantile locaux" : tmp_data[f'{element[0]} - Normalized Interval Width']
        }

        df = pd.DataFrame(d_data)

        df['Modèle'] = df['Modèle'] + ' (MASE: ' + df['MASE'].astype(str) + ')'

        # Créer un graphique à bulles avec Plotly
        fig = px.scatter(df, x="Coverage quantile local", y="Moyenne quantile locaux",
                        size="MASE", 
                        color="Modèle",
                        color_discrete_sequence=colors,
                        size_max=60,
                        labels={"Coverage quantile local": "MCR",
                                "Moyenne quantile locaux": "MSIW",
                                "MASE": "MASE"})


        fig.add_trace(go.Scatter(
            x=df["Coverage quantile local"],
            y=df["Moyenne quantile locaux"],
            mode='markers',
            marker=dict(
                size=5,
                color=np.array(colors)[:len(df)],  # Use the same color as the bubbles
                line=dict(width=1, color='white')  # Optional: add a black border to the center points
            ),
            showlegend=False  # Do not show in legend
        ))


        y_min = df['Moyenne quantile locaux'].min() - df['Moyenne quantile locaux'].min()*0.2
        y_max = df['Moyenne quantile locaux'].max() + df['Moyenne quantile locaux'].max()*0.2


        fig.add_trace(go.Scatter(x=[0.90, 0.90], 
                                y = [y_min, y_max], 
                                line=dict(color='black', dash='dot', width=.4), 
                                mode='lines',
                                showlegend=False))
        

        fig.update_layout(title = f'{element[0]} - Horizon {element[1]}',
                        showlegend=True)
        
        fig.update_yaxes(range=[y_min, y_max])
        fig.update_xaxes(range=[0.80, 1], tickformat=".0%")
        fig.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC # Dataset

# COMMAND ----------

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def get_ercot_sar(sample=True):
    url = "https://github.com/ourownstory/neuralprophet-data/raw/main/datasets_raw/energy/ERCOT_load_2004_2021Sept.csv"
    df = pd.read_csv(url).rename(columns = {'y' : 'target'})
    df['ds'] = pd.to_datetime(df['ds'])
    # There is only a single missing value per time series - forward fill them
    df.ffill(inplace=True)


    if sample:
        # df = df[45429+20:45429+20+8000-8][['ds', 'target']]
        df = df[(df.ds.dt.year >= 2018) & (df.ds.dt.year < 2020)][['ds', 'target']]
    ercot_df = df[['ds', 'target']]


    ercot_df = ercot_df.set_index('ds')
    idx = pd.date_range(start=ercot_df.index.min(), end=ercot_df.index.max(), freq='H')
    missing_dates = idx.difference(ercot_df.index)

    if len(missing_dates) > 0:
        for date in missing_dates:
            ercot_df.loc[date, 'target'] = np.nan
        
        ercot_df = ercot_df.sort_index()
        ercot_df.target = ercot_df.target.interpolate(method='linear')

    ercot_df = ercot_df.reset_index()

    return ercot_df


import pandas as pd

def get_nn5_daily_sar():
    df = pd.read_excel('./dataset/nn5_daily/NN5_Daily.xls')
    df_nn5 = df.iloc[16:, :101]
    df_nn5 = df_nn5.rename(columns = {df.columns[0] : 'ds'})
    df_nn5 = df_nn5.set_index(df_nn5.columns[0])
    df_nn5 = df_nn5.ffill()
    df_nn5.columns = list(range(df_nn5.shape[1]))
    # df_nn5 = df_nn5.astype('float32')
    return df_nn5


import pandas as pd

def get_nn5_weekly_sar():
    url = 'https://raw.githubusercontent.com/rakshitha123/WeeklyForecasting/master/datasets/nn5_weekly_dataset.txt'
    df = pd.read_csv(url, header=None).T
    df = df.set_index(pd.date_range(start='1990-01-01', periods = df.shape[0], freq='W')).astype('float64')
    return df.iloc[:, :100]


import pandas as pd

def get_m3_monthly_sar():
    #The path was chosen considering the location of our notebooks
    df_m3 = pd.read_excel('./dataset/m3_monthly/m3_monthly.xls', sheet_name = 'M3Month')
    col_to_keep = df_m3.columns[~df_m3.isna().any()]
    df_m3_f = df_m3[col_to_keep].T.iloc[6:]

    # We use the same dates range for all series even if it's not true
    df_m3_f = df_m3_f.set_index(pd.date_range(start = '1990-01-01', periods=len(df_m3_f), freq = 'M')).astype('float32')
    return df_m3_f

# COMMAND ----------

ercot_df = get_ercot_sar(sample=True)
df_m3_f = get_m3_monthly_sar()
df = get_nn5_weekly_sar()
df_nn5 = get_nn5_daily_sar()

# COMMAND ----------

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Créer une figure avec 3 lignes et 1 colonne
fig = make_subplots(rows=4, cols=1, subplot_titles=("ERCOT distribution",
                                                    "NN5 Daily distribution",
                                                    "NN5 Weekly distribution",
                                                    "M3 Monthly distribution"))

# Histogramme 1
fig1 = px.histogram(df_nn5, x=ercot_df.target.values.ravel(), nbins=20)
fig.add_trace(go.Histogram(x=fig1.data[0].x, nbinsx=20), row=1, col=1)

# Histogramme 2
fig2 = px.histogram(df_nn5, x=df_nn5.values.flatten(), nbins=20)
fig.add_trace(go.Histogram(x=fig2.data[0].x, nbinsx=20), row=2, col=1)

# Histogramme 3
fig3 = px.histogram(df, x=df.values.flatten(), nbins=20)
fig.add_trace(go.Histogram(x=fig3.data[0].x, nbinsx=20), row=3, col=1)

# Histogramme 4
fig4 = px.histogram(df_m3_f, x=df_m3_f.values.flatten(), nbins=2000)
fig.add_trace(go.Histogram(x=fig4.data[0].x, nbinsx=2000), row=4, col=1)

# Mettre à jour les axes et le layout
fig.update_xaxes(title_text="Value", row=1, col=1)
fig.update_xaxes(title_text="Value", row=2, col=1)
fig.update_xaxes(title_text="Value", row=3, col=1)
fig.update_xaxes(title_text="Value", row=4, col=1)

fig.update_yaxes(title_text="Count", row=1, col=1)
fig.update_yaxes(title_text="Count", row=2, col=1)
fig.update_yaxes(title_text="Count", row=3, col=1)
fig.update_yaxes(title_text="Count", row=4, col=1)

fig.update_layout(height=900, showlegend=False)

# Afficher la figure
fig.show()


# COMMAND ----------


