import os
import numpy as np
import pandas as pd
from ..io.TRdata_reader import read_mgm

class StationOrganizerTR:
    # Turkish State Meteorological Service data organizer
    def __init__(self, cfg):
        self.cfg = cfg
        self.out_dir = cfg.outputs_dir
        self.ranges = cfg.ranges
        os.makedirs(self.out_dir, exist_ok=True)

    def run(self):
        (stations, df_precip, df_tmp, df_wndsp, df_wnddir, df_mxwndsp,
        df_pressure, df_rhum, df_radsum, df_rad, df_insolationintensity,
        df_insolationtime, df_minsoiltmp0cm, df_soiltmp100cm, df_soiltmp50cm,
        df_soiltmp20cm, df_soiltmp10cm, df_soiltmp5cm) = self._get_tr_data_into_dataframes()

        # Ranges
        ranges = {k: tuple(v) for k, v in self.ranges.items()}

        for station in stations:
            st = self._build_station_dataframe(
                station,
                df_precip, df_tmp, df_wndsp, df_wnddir, df_mxwndsp,
                df_pressure, df_rhum, df_radsum, df_rad, df_insolationintensity,
                df_insolationtime, df_minsoiltmp0cm, df_soiltmp100cm, df_soiltmp50cm,
                df_soiltmp20cm, df_soiltmp10cm, df_soiltmp5cm,
                ranges
            )
            st.to_json(os.path.join(self.out_dir, f'station_{station}.json'))
            st_norm = self._normalize(st.copy(), ranges)
            st_norm.to_json(os.path.join(self.out_dir, f'station_{station}_normalized.json'))

    def _get_tr_data_into_dataframes(self):
        # Reads Turkish State Meteorological Service excel or text format data
        df_precip, _ = read_mgm(self.cfg.mgm_hourly_folder, self.cfg.pcp_file,
                             self.cfg.pcp_column, is_windspeed=False)
        df_tmp, _ = read_mgm(self.cfg.mgm_hourly_folder, self.cfg.tmp_file,
                             self.cfg.tmp_column, is_windspeed=False)
        df_wndsp, df_wnddir = read_mgm(self.cfg.mgm_hourly_folder, self.cfg.wnd_file,
                             self.cfg.wnd_column, is_windspeed=True)
        df_mxwndsp, _ = read_mgm(self.cfg.mgm_hourly_folder, self.cfg.maxwnd_file,
                             self.cfg.maxwnd_column, is_windspeed=True)
        df_pressure, _ = read_mgm(self.cfg.mgm_hourly_folder, self.cfg.pressure_file,
                             self.cfg.pressure_column, is_windspeed=False)
        df_rhum, _ = read_mgm(self.cfg.mgm_hourly_folder, self.cfg.rhum_file,
                             self.cfg.rhum_column, is_windspeed=False)
        df_radsum, _ = read_mgm(self.cfg.mgm_hourly_folder, self.cfg.radsum_file,
                             self.cfg.radsum_column, is_windspeed=False)
        df_rad, _ = read_mgm(self.cfg.mgm_hourly_folder, self.cfg.rad_file,
                             self.cfg.rad_column, is_windspeed=False)
        df_insolationintensity, _ = read_mgm(self.cfg.mgm_hourly_folder, self.cfg.insolationintensity_file,
                             self.cfg.insolationintensity_column, is_windspeed=False)
        df_insolationtime, _ = read_mgm(self.cfg.mgm_hourly_folder, self.cfg.insolationtime_file,
                             self.cfg.insolationtime_column, is_windspeed=False)
        df_minsoiltmp0cm, _ = read_mgm(self.cfg.mgm_hourly_folder, self.cfg.minsoiltmp0cm_file,
                             self.cfg.minsoiltmp0cm_column, is_windspeed=False)
        df_soiltmp100cm, _ = read_mgm(self.cfg.mgm_hourly_folder, self.cfg.soiltmp100cm_file,
                             self.cfg.soiltmp100cm_column, is_windspeed=False)
        df_soiltmp50cm, _ = read_mgm(self.cfg.mgm_hourly_folder, self.cfg.soiltmp50cm_file,
                             self.cfg.soiltmp50cm_column, is_windspeed=False)
        df_soiltmp20cm, _ = read_mgm(self.cfg.mgm_hourly_folder, self.cfg.soiltmp20cm_file,
                             self.cfg.soiltmp20cm_column, is_windspeed=False)
        df_soiltmp10cm, _ = read_mgm(self.cfg.mgm_hourly_folder, self.cfg.soiltmp10cm_file,
                             self.cfg.soiltmp10cm_column, is_windspeed=False)
        df_soiltmp5cm, _ = read_mgm(self.cfg.mgm_hourly_folder, self.cfg.soiltmp5cm_file,
                             self.cfg.soiltmp5cm_column, is_windspeed=False)
        # stations
        stations = [c for c in df_precip.columns]
        return (stations, df_precip, df_tmp, df_wndsp, df_wnddir, df_mxwndsp,
                df_pressure, df_rhum, df_radsum, df_rad, df_insolationintensity,
                df_insolationtime, df_minsoiltmp0cm, df_soiltmp100cm, df_soiltmp50cm,
                df_soiltmp20cm, df_soiltmp10cm, df_soiltmp5cm)

    def _build_station_dataframe(self, station, df_precip, df_tmp, df_wndsp, df_wnddir, df_mxwndsp,
                                 df_pressure, df_rhum, df_radsum, df_rad, df_insolationintensity,
                                 df_insolationtime, df_minsoiltmp0cm, df_soiltmp100cm, df_soiltmp50cm,
                                 df_soiltmp20cm, df_soiltmp10cm, df_soiltmp5cm, ranges):
        def _get(df, colname):
            try:
                c = df[[station]]
                c.columns = [colname]
            except Exception:
                c = pd.DataFrame(index=df_precip.index, columns=[colname], dtype=float)
            return c

        # Base index filled hourly
        df_stn = pd.DataFrame(index=pd.date_range(df_precip.index.min(), df_precip.index.max(), freq='1h'))
        df_stn = df_stn.join(_get(df_precip, "precip"), how="left")
        if not df_tmp.empty:
            df_stn = df_stn.join(_get(df_tmp, "tmp"), how="left")
        if not df_wndsp.empty:
            df_stn = df_stn.join(_get(df_wndsp, "wndsp"), how="left")
        if not df_wnddir.empty:
            # wind dir: letters -> numeric map
            _wd = _get(df_wnddir, "wnddir")
            if "wnddir" in _wd:
                _wd["wnddir"] = _wd["wnddir"].astype(str).str.strip()
                winddir2number = {'E':1,'ENE':2,'NE':3,'NNE':4,'N':5,'NNW':6,'NW':7,'WNW':8,'W':9,'WSW':10,'SW':11,'SSW':12,'S':13,'SSE':14,'SE':15,'ESE':16,'C':17}
                _wdn = _wd["wnddir"].map(winddir2number)
                df_stn["wnddir_nbr"] = _wdn
        if not df_mxwndsp.empty:
            df_stn = df_stn.join(_get(df_mxwndsp, "mxwndsp"), how="left")
        if not df_pressure.empty:
            df_stn = df_stn.join(_get(df_pressure, "pressure"), how="left")
        if not df_rhum.empty:
            df_stn = df_stn.join(_get(df_rhum, "rhum"), how="left")
        if not df_radsum.empty:
            df_stn = df_stn.join(_get(df_radsum, "radsum"), how="left")
        if not df_rad.empty:
            df_stn = df_stn.join(_get(df_rad, "rad"), how="left")
        if not df_insolationintensity.empty:
            df_stn = df_stn.join(_get(df_insolationintensity, "insolationintensity"), how="left")
        if not df_insolationtime.empty:
            df_stn = df_stn.join(_get(df_insolationtime, "insolationtime"), how="left")
        if not df_minsoiltmp0cm.empty:
            df_stn = df_stn.join(_get(df_minsoiltmp0cm, "minsoiltmp0cm"), how="left")
        if not df_soiltmp100cm.empty:
            df_stn = df_stn.join(_get(df_soiltmp100cm, "soiltmp100cm"), how="left")
        if not df_soiltmp50cm.empty:
            df_stn = df_stn.join(_get(df_soiltmp50cm, "soiltmp50cm"), how="left")
        if not df_soiltmp20cm.empty:
            df_stn = df_stn.join(_get(df_soiltmp20cm, "soiltmp20cm"), how="left")
        if not df_soiltmp10cm.empty:
            df_stn = df_stn.join(_get(df_soiltmp10cm, "soiltmp10cm"), how="left")
        if not df_soiltmp5cm.empty:
            df_stn = df_stn.join(_get(df_soiltmp5cm, "soiltmp5cm"), how="left")

        return df_stn

    def _normalize(self, df, ranges):
        for k, (mn, mx) in ranges.items():
            if k in df.columns:
                df[k] = (df[k] - mn) / (mx - mn)
        return df