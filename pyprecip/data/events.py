import os
import pandas as pd
import numpy as np
import joblib
from typing import List, Dict

class TrainingDataCreator:
    def __init__(self, cfg):
        self.cfg = cfg
        os.makedirs(self.cfg.out_dir, exist_ok=True)

    def run(self):
        stations = self.cfg.stations
        # read normalized per-station files
        station_df: Dict[int, pd.DataFrame] = {}
        station_events: Dict[int, pd.DataFrame] = {}

        for st in stations:
            df = pd.read_json(os.path.join(self.cfg.organized_dir, f'station_{st}_normalized.json'))
            df = df[list(self.cfg.cols)]
            df.columns = list(self.cfg.short_cols)

            # denormalize pcp, tmp, rhum using provided min/max
            df['pcp_mm'] = df['pcp'] * (self.cfg.max_precip_mm - self.cfg.min_precip_mm) + self.cfg.min_precip_mm
            df['rhum']   = df['rhum'] * (self.cfg.max_rhum_pct - self.cfg.min_rhum_pct) + self.cfg.min_rhum_pct
            df['tmp']    = df['tmp']  * (self.cfg.max_tmp_c   - self.cfg.min_tmp_c)   + self.cfg.min_tmp_c
            df.drop(columns=['pcp'], inplace=True)

            # Interpolate temp/rhum
            df["rhum"] = df["rhum"].interpolate(method="time")
            df["tmp"] = df["tmp"].interpolate(method="time")

            # diffs and rolling
            df["tmp_diff"] = df["tmp"] - df["tmp"].shift(1)
            df["rhum_diff"] = df["rhum"] - df["rhum"].shift(1)
            df["tmp_diff_roll10h"] = df["tmp_diff"].rolling("10H").mean()
            df["rhum_diff_roll10h"] = df["rhum_diff"].rolling("10H").mean()

            # detect events per station (MIT per season)
            ev_series = []
            event_id = 0
            last_wet = None
            for t, val in df["pcp_mm"].items():
                if pd.notna(val) and val > 0:
                    # cold: Oct–Mar (1h) vs warm: Apr–Sep (2h)
                    mit = self.cfg.mit_cold_hours if t.month in [10,11,12,1,2,3] else self.cfg.mit_warm_hours
                    if last_wet is None:
                        event_id += 1
                    else:
                        gap = int((t - last_wet).total_seconds() // 3600)
                        if gap > mit:
                            event_id += 1
                    last_wet = t
                    ev_series.append(event_id)
                else:
                    ev_series.append(np.nan)
            df["event"] = ev_series

            # build event summary
            ev_info = (
                df.dropna(subset=["event"])
                  .groupby("event")
                  .agg(
                      start=("pcp_mm", lambda x: x.index.min()),
                      end=("pcp_mm",   lambda x: x.index.max()),
                      duration=("pcp_mm", "count"),
                      total_precip=("pcp_mm", "sum")
                  )
            )

            # remove small events
            to_nan = []
            for ev in ev_info.index:
                if ev_info.loc[ev, "total_precip"] <= 1:
                    to_nan.append(ev)
                elif ev_info.loc[ev, "duration"] > 1 and ev_info.loc[ev, "total_precip"] <= 2:
                    to_nan.append(ev)
            if to_nan:
                df.loc[df["event"].isin(to_nan), ["pcp_mm", "event"]] = np.nan

            # recompute event summary
            ev_info = (
                df.dropna(subset=["event"])
                  .groupby("event")
                  .agg(
                      start=("pcp_mm", lambda x: x.index.min()),
                      end=("pcp_mm",   lambda x: x.index.max()),
                      duration=("pcp_mm", "count"),
                      total_precip=("pcp_mm", "sum")
                  )
            ).reset_index(drop=True)
            ev_info.index = range(1, len(ev_info)+1)

            station_df[st] = df
            station_events[st] = ev_info

        # create global events that occur in all stations
        buf = pd.Timedelta(hours=self.cfg.global_event_buffer_hours)
        all_events = []
        for st in stations:
            d = station_events[st].copy()
            d['station'] = st
            all_events.append(d)
        df_stnevents = pd.concat(all_events).sort_values("start").reset_index(drop=True)

        # assign global ids
        global_event = 0
        global_ids = []
        current_end = pd.Timestamp.min
        for _, row in df_stnevents.iterrows():
            if row["start"] > current_end + buf:
                global_event += 1
            global_ids.append(global_event)
            current_end = max(current_end, row["end"])
        df_stnevents["global_event"] = global_ids

        # keep only events present for all stations
        n_stations = df_stnevents["station"].nunique()
        counts = df_stnevents.groupby("global_event")["station"].nunique()
        valid = counts[counts == n_stations].index
        df_stnevents = df_stnevents[df_stnevents["global_event"].isin(valid)]

        # recompute global start/end
        df_globalsummary = (
            df_stnevents.groupby("global_event")
            .agg(new_start=("start", "min"), new_end=("end", "max"))
            .reset_index()
        )
        df_globalsummary['global_event'] = df_globalsummary.index + 1

        # build per-station cumulative event frames and save training inputs
        out_base = os.path.join(self.cfg.out_dir, f'Pirone_Nstn{len(stations)}_evnt_{"-".join(self.cfg.short_cols)}')
        os.makedirs(out_base, exist_ok=True)

        for st in stations:
            df = station_df[st].copy()
            cum_frames = []
            for _, row in df_globalsummary.iterrows():
                ev = int(row["global_event"])
                start, end = row["new_start"], row["new_end"]
                full_index = pd.date_range(start, end, freq="h")
                temp = df.loc[start:end].reindex(full_index)
                temp["pcp_mm"] = temp["pcp_mm"].fillna(0)
                temp["cum_pcp"] = temp["pcp_mm"].cumsum()
                temp["event"] = ev
                temp["cum_hour"] = (temp.index - temp.index[0]).total_seconds() / 3600
                cum_frames.append(temp)
            df_cum = pd.concat(cum_frames)
            keep = ["cum_pcp", "cum_hour", "rhum", "tmp", "pcp_mm", "rhum_diff", "tmp_diff", "tmp_diff_roll10h", "rhum_diff_roll10h"]
            df_out = df_cum[keep].copy()
            joblib.dump(df_out, os.path.join(out_base, f"{st}_training_data.joblib"))