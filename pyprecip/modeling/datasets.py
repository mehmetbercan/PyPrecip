import os
import numpy as np
import pandas as pd
import joblib

def load_station_inputs(stations, base_dir, train_col, other_cols):
    dfs = []
    for st in stations:
        path = os.path.join(base_dir, f"{st}_training_data.joblib")
        df = joblib.load(path).sort_index()
        df = df[[train_col] + other_cols]
        dfs.append(df.rename(columns=lambda c: f"{c}_{st}"))

    df_all = dfs[0]
    for d in dfs[1:]:
        df_all = df_all.join(d, how="inner")
    df_all = df_all.dropna()
    df_all = df_all.round(6)
    return df_all