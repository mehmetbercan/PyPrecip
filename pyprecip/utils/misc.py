import pandas as pd
import numpy as np
import pickle
import json

def add_normalized_time_features(df: pd.DataFrame):
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")

    df = df.copy()
    df['hour_norm'] = df.index.hour / 23.0
    df['sin_hour'] = (np.sin(2 * np.pi * df["hour_norm"]) + 1) / 2
    df['cos_hour'] = (np.cos(2 * np.pi * df["hour_norm"]) + 1) / 2

    is_leap = df.index.is_leap_year
    days_in_year = np.where(is_leap, 366, 365)

    df["dayofyear_norm"] = (df.index.dayofyear - 1) / (days_in_year - 1)
    df['sin_dayofyear'] = (np.sin(2 * np.pi * df["dayofyear_norm"]) + 1) / 2
    df['cos_dayofyear'] = (np.cos(2 * np.pi * df["dayofyear_norm"]) + 1) / 2

    df["week_norm"] = (df.index.isocalendar().week - 1) / 52.0
    df['sin_week'] = (np.sin(2 * np.pi * df["week_norm"]) + 1) / 2
    df['cos_week'] = (np.cos(2 * np.pi * df["week_norm"]) + 1) / 2

    df["month_norm"] = (df.index.month - 1) / 11.0
    df['sin_month'] = (np.sin(2 * np.pi * df["month_norm"]) + 1) / 2
    df['cos_month'] = (np.cos(2 * np.pi * df["month_norm"]) + 1) / 2

    del df['hour_norm'], df['dayofyear_norm'], df['week_norm'], df['month_norm']
    added = ['sin_hour','cos_hour','sin_dayofyear','cos_dayofyear','sin_week','cos_week','sin_month','cos_month']
    return df, added