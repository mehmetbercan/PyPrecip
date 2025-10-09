import pandas as pd
import numpy as np
import json
import os
from typing import Tuple, Union

def read_mgm_text(txtfile: str, column: str, is_windspeed: bool=False) -> Union[Tuple[pd.DataFrame, str], Tuple[pd.DataFrame, pd.DataFrame]]:
    # Reads Turkish State Meteorological Service text format data
    df = pd.read_csv(txtfile, parse_dates=[[2,3,4,5]], index_col=0,
                     date_format="%Y %m %d %H", sep='|')
    df.index.name = 'DateTime'
    stations = df['Istasyon_No'].unique()

    if is_windspeed:
        df['RUZGAR_YONU'] = df[column].str.slice(stop=3)
        df['RUZGAR_HIZI'] = pd.to_numeric(df[column].str.slice(start=3), downcast='float')
        df_stn_wd = pd.DataFrame()
        df_stn_ws = pd.DataFrame()
        for station in stations:
            _wd = df[['RUZGAR_YONU']].loc[df['Istasyon_No'] == station]
            _wd.columns = [f'{station}']
            df_stn_wd = pd.concat([df_stn_wd, _wd], axis=1)

            _ws = df[['RUZGAR_HIZI']].loc[df['Istasyon_No'] == station]
            _ws.columns = [f'{station}']
            df_stn_ws = pd.concat([df_stn_ws, _ws], axis=1)
        return df_stn_ws, df_stn_wd
    else:
        df_stn = pd.DataFrame()
        for station in stations:
            _v = df[[column]].loc[df['Istasyon_No'] == station]
            _v.columns = [f'{station}']
            df_stn = pd.concat([df_stn, _v], axis=1)
        return df_stn, pd.DataFrame()

def read_mgm_excel(io: str) -> pd.DataFrame:
    # Reads Turkish State Meteorological Service excel format data
    df_excel = pd.read_excel(io)
    indeces = []; values = []; df = pd.DataFrame()
    pre_gageid = 'na'; gageid = 'na'

    for i, row in df_excel.iterrows():
        if row.iloc[1]:
            if 'Yıl/Ay:' in str(row.iloc[1]):
                parts = row.iloc[1].split('/')
                year = int(parts[1].split(':')[1])
                month = int(parts[2].split(' ')[0])
                gageid = int(parts[-1])
                if i < 5:
                    pre_gageid = gageid
            if row.iloc[1] in range(1,32):
                day = int(row.iloc[1])
                linevalues = row.iloc[2:26]
                for hour in range(24):
                    if str(linevalues.iloc[hour])!='nan':
                        val = float(linevalues.iloc[hour])
                        dt = f'{month}/{day}/{year} {hour}:00'
                        indeces.append(dt); values.append(val)
            if pre_gageid != gageid:
                _df = pd.DataFrame(values, index=pd.DatetimeIndex(indeces))
                if not _df.empty:
                    _df.columns = [f'{pre_gageid}']
                    df = _df.copy(deep=True) if df.empty else pd.concat([df,_df], axis=1)
                pre_gageid = gageid
                indeces = []; values = []

    _df = pd.DataFrame(values, index=pd.DatetimeIndex(indeces))
    if not _df.empty:
        _df.columns = [f'{pre_gageid}']
        df = _df.copy(deep=True) if df.empty else pd.concat([df,_df], axis=1)

    # fill index and adjust timezone (+3h to TRT)
    index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='1h')
    df = pd.concat([pd.DataFrame(index=index), df], axis=1)
    df.index = pd.date_range(start=df.index.min()+pd.Timedelta(3, "h"),
                             end=df.index.max()+pd.Timedelta(3, "h"), freq='1h')
    return df

def read_mgm(folder: str, filename: str, column: str, is_windspeed: bool=False):
    # Reads Turkish State Meteorological Service excel or text format data
    df_wd = pd.DataFrame()
    if filename:
        if filename.split('.')[-1] == 'txt':
            df, df_wd = read_mgm_text(os.path.join(folder, filename),column, is_windspeed=is_windspeed)
        elif filename.split('.')[-1] == 'xlsx':
            df = read_mgm_excel(os.path.join(folder, filename))
        else:
            raise ValueError("{} file extension must be in .txt or .xlsx".format(filename))
    else:
        df = pd.DataFrame()

    return df, df_wd

