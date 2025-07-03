import pandas as pd

def get_nn5_daily():
    df = pd.read_excel('../../dataset/nn5_daily/NN5_Daily.xls')
    df_nn5 = df.iloc[16:, :101]
    df_nn5 = df_nn5.rename(columns = {df.columns[0] : 'ds'})
    df_nn5 = df_nn5.set_index(df_nn5.columns[0])
    df_nn5 = df_nn5.ffill()
    df_nn5.columns = list(range(df_nn5.shape[1]))
    return df_nn5