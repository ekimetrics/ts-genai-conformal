import pandas as pd

def get_nn5_weekly():
    url = 'https://raw.githubusercontent.com/rakshitha123/WeeklyForecasting/master/datasets/nn5_weekly_dataset.txt'
    df = pd.read_csv(url, header=None).T
    df = df.set_index(pd.date_range(start='1990-01-01', periods = df.shape[0], freq='W')).astype('float64')
    return df.iloc[:, :100]