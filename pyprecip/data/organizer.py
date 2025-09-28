import os
import numpy as np
import pandas as pd
from ..io.mgm_reader import read_mgm_text, read_mgm_excel

class StationOrganizer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.out_dir = cfg.outputs_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def run(self):
        # Read core text sources
        df_precip, _ = read_mgm_text(os.path.join(self.cfg.mgm_hourly_folder, self.cfg.prp_file),
                                     column='TOPLAM_YAGIS_OMGI_mm', is_windspeed=False)
        df_tmp, _ = read_mgm_text(os.path.join(self.cfg.mgm_hourly_folder, self.cfg.tmp_file),
                                  column='SICAKLIK_?C', is_windspeed=False)
        df_wndsp, df_wnddir = read_mgm_text(os.path.join(self.cfg.mgm_hourly_folder, self.cfg.wnd_file),
                                            column='RUZGAR_YONU_VE_HIZI', is_windspeed=True)
        df_mxwndsp, _ = read_mgm_text(os.path.join(self.cfg.mgm_hourly_folder, self.cfg.maxwnd_file),
                                      column='SAATLIK_MAKSIMUM_RUZGARIN_YONU_VE_HIZI', is_windspeed=True)
        df_pressure, _ = read_mgm_text(os.path.join(self.cfg.mgm_hourly_folder, self.cfg.pressure_file),
                                       column='AKTUEL_BASINC_hPa', is_windspeed=False)
        df_rhum, _ = read_mgm_text(os.path.join(self.cfg.mgm_hourly_folder, self.cfg.rhum_file),
                                   column='NISPI_NEM_%', is_windspeed=False)

        # Excel->DataFrame JSON-like readers
        def _load_excel(fname): return read_mgm_excel(os.path.join(self.cfg.mgm_hourly_folder, fname))
        df_radsum = _load_excel(self.cfg.radsum_file)
        df_rad = _load_excel(self.cfg.rad_file)
        df_insolationintensity = _load_excel(self.cfg.insolationintensity_file)
        df_insolationtime = _load_excel(self.cfg.insolationtime_file)
        df_minsoiltmp0cm = _load_excel(self.cfg.minsoiltmp0cm_file)
        df_soiltmp100cm = _load_excel(self.cfg.soiltmp100cm_file)
        df_soiltmp50cm = _load_excel(self.cfg.soiltmp50cm_file)
        df_soiltmp20cm = _load_excel(self.cfg.soiltmp20cm_file)
        df_soiltmp10cm = _load_excel(self.cfg.soiltmp10cm_file)
        df_soiltmp5cm = _load_excel(self.cfg.soiltmp5cm_file)

        # Manual QA/QC (adapted from your script)
        df_radsum = df_radsum[df_radsum<70000]
        df_rad = df_rad[df_rad<1200]
        df_insolationintensity = df_insolationintensity[df_insolationintensity<110]
        df_soiltmp100cm = df_soiltmp100cm[df_soiltmp100cm>4]
        df_soiltmp50cm = df_soiltmp50cm[df_soiltmp50cm>2]

        # Ranges
        ranges = dict(
            precip=(0,50), tmp=(-30,50), wndsp=(0,30), mxwndsp=(0,50),
            wnddir=(1,17), pressure=(700,1100), rhum=(0,100),
            radsum=(0,70000), rad=(0,1200), insolationintensity=(0,100),
            insolationtime=(0,1), minsoiltmp0cm=(-30,55), soiltmp100cm=(0,35),
            soiltmp50cm=(0,40), soiltmp20cm=(-5,50), soiltmp10cm=(-10,55),
            soiltmp5cm=(-10,65)
        )

        stations = [int(c) for c in df_precip.columns]
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
        df_stn = df_stn.join(_get(df_tmp, "tmp"), how="left")
        df_stn = df_stn.join(_get(df_wndsp, "wndsp"), how="left")
        # wind dir: letters -> numeric map
        _wd = _get(df_wnddir, "wnddir")
        if "wnddir" in _wd:
            _wd["wnddir"] = _wd["wnddir"].astype(str).str.strip()
            winddir2number = {'E':1,'ENE':2,'NE':3,'NNE':4,'N':5,'NNW':6,'NW':7,'WNW':8,'W':9,'WSW':10,'SW':11,'SSW':12,'S':13,'SSE':14,'SE':15,'ESE':16,'C':17}
            _wdn = _wd["wnddir"].map(winddir2number)
            df_stn["wnddir_nbr"] = _wdn
        df_stn = df_stn.join(_get(df_mxwndsp, "mxwndsp"), how="left")
        df_stn = df_stn.join(_get(df_pressure, "pressure"), how="left")
        df_stn = df_stn.join(_get(df_rhum, "rhum"), how="left")
        df_stn = df_stn.join(_get(df_radsum, "radsum"), how="left")
        df_stn = df_stn.join(_get(df_rad, "rad"), how="left")
        df_stn = df_stn.join(_get(df_insolationintensity, "insolationintensity"), how="left")
        df_stn = df_stn.join(_get(df_insolationtime, "insolationtime"), how="left")
        df_stn = df_stn.join(_get(df_minsoiltmp0cm, "minsoiltmp0cm"), how="left")
        df_stn = df_stn.join(_get(df_soiltmp100cm, "soiltmp100cm"), how="left")
        df_stn = df_stn.join(_get(df_soiltmp50cm, "soiltmp50cm"), how="left")
        df_stn = df_stn.join(_get(df_soiltmp20cm, "soiltmp20cm"), how="left")
        df_stn = df_stn.join(_get(df_soiltmp10cm, "soiltmp10cm"), how="left")
        df_stn = df_stn.join(_get(df_soiltmp5cm, "soiltmp5cm"), how="left")

        return df_stn

    def _normalize(self, df, ranges):
        for k, (mn, mx) in ranges.items():
            if k in df.columns:
                df[k] = (df[k] - mn) / (mx - mn)
        return df