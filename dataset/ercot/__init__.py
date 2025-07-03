import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def get_ercot(sample=True):
    url = "https://github.com/ourownstory/neuralprophet-data/raw/main/datasets_raw/energy/ERCOT_load_2004_2021Sept.csv"
    df = pd.read_csv(url).rename(columns = {'y' : 'target'})
    df['ds'] = pd.to_datetime(df['ds'])
    df.ffill(inplace=True)


    if sample:
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