import pandas as pd

def get_m3_monthly():
    #The path was chosen considering the location of our notebooks
    df_m3 = pd.read_excel('../../dataset/m3_monthly/m3_monthly.xls', sheet_name = 'M3Month')
    col_to_keep = df_m3.columns[~df_m3.isna().any()]
    df_m3_f = df_m3[col_to_keep].T.iloc[6:]

    # We use the same dates range for all series even if it's not true
    df_m3_f = df_m3_f.set_index(pd.date_range(start = '1990-01-01', periods=len(df_m3_f), freq = 'M')).astype('float32')
    return df_m3_f