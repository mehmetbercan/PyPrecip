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
        # read per-station files
        station_df: Dict[int, pd.DataFrame] = {}
        station_events: Dict[int, pd.DataFrame] = {}

        for st in stations:
            if self.cfg.use_normalized_data:
                df = pd.read_json(os.path.join(self.cfg.organized_dir, f'station_{st}_normalized.json'))
                df = df[list(self.cfg.cols)]
                df.columns = list(self.cfg.short_cols)
                # denormalize pcp using provided min/max
                min_precip = self.cfg.min_max_precip[0]; max_precip = self.cfg.min_max_precip[1]
                df['pcp_unit'] = df['pcp'] * (max_precip - min_precip) + min_precip
            else:
                df = pd.read_json(os.path.join(self.cfg.organized_dir, f'station_{st}.json'))
                df = df[list(self.cfg.cols)]
                df.columns = list(self.cfg.short_cols)
                df['pcp_unit'] = df['pcp'].copy()

            # Interpolate temp/rhum
            if self.cfg.interpolate_tmp_rhum:
                df["rhum"] = df["rhum"].interpolate(method="time")
                df["tmp"] = df["tmp"].interpolate(method="time")

            # diffs and rolling
            if self.cfg.calculate_diff_tmp_rhum:
                df["tmp_diff"] = df["tmp"] - df["tmp"].shift(1)
                df["rhum_diff"] = df["rhum"] - df["rhum"].shift(1)
                if self.cfg.rolling_mean_diff_tmp_rhum:
                    df["tmp_diff_roll10h"] = df["tmp_diff"].rolling(
                        "{}H".format(self.cfg.rolling_mean_diff_tmp_rhum_hour)).mean()
                    df["rhum_diff_roll10h"] = df["rhum_diff"].rolling(
                        "{}H".format(self.cfg.rolling_mean_diff_tmp_rhum_hour)).mean()

            # detect events per station (MIT per season)
            ev_series = []
            event_id = 0
            last_wet = None
            for t, val in df["pcp_unit"].items():
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
                      start=("pcp_unit", lambda x: x.index.min()),
                      end=("pcp_unit",   lambda x: x.index.max()),
                      duration=("pcp_unit", "count"),
                      total_precip=("pcp_unit", "sum")
                  )
            )

            # remove small events
            to_nan = []
            for ev in ev_info.index:
                if (ev_info.loc[ev, "total_precip"]
                    <= self.cfg.event_total_pcp_threshold_4_1hr_event):
                    to_nan.append(ev)
                elif (ev_info.loc[ev, "duration"] > 1
                    and ev_info.loc[ev, "total_precip"]
                    <= self.cfg.event_total_pcp_threshold_4_larger_events):
                    to_nan.append(ev)
            if to_nan:
                df.loc[df["event"].isin(to_nan), ["pcp_unit", "event"]] = np.nan

            # recompute event summary
            ev_info = (
                df.dropna(subset=["event"])
                  .groupby("event")
                  .agg(
                      start=("pcp_unit", lambda x: x.index.min()),
                      end=("pcp_unit",   lambda x: x.index.max()),
                      duration=("pcp_unit", "count"),
                      total_precip=("pcp_unit", "sum")
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
        out_base = os.path.join(self.cfg.out_dir, f'CumEvnt_Nstn{len(stations)}_{"-".join(self.cfg.short_cols)}')
        os.makedirs(out_base, exist_ok=True)

        for st in stations:
            df = station_df[st].copy()
            cum_frames = []
            for _, row in df_globalsummary.iterrows():
                ev = int(row["global_event"])
                start, end = row["new_start"], row["new_end"]
                full_index = pd.date_range(start, end, freq="h")
                temp = df.loc[start:end].reindex(full_index)
                temp["pcp_unit"] = temp["pcp_unit"].fillna(0)
                temp["cum_pcp"] = temp["pcp_unit"].cumsum()
                temp["event"] = ev
                temp["cum_hour"] = (temp.index - temp.index[0]).total_seconds() / 3600
                cum_frames.append(temp)
            df_cum = pd.concat(cum_frames)
            keep = ["cum_pcp", "cum_hour"] + self.cfg.short_cols
            if self.cfg.calculate_diff_tmp_rhum:
                keep += ["rhum_diff", "tmp_diff"]
                if self.cfg.rolling_mean_diff_tmp_rhum:
                    h = self.cfg.rolling_mean_diff_tmp_rhum_hour
                    keep += [f"tmp_diff_roll{h}h", f"rhum_diff_roll{h}h"]
            df_out = df_cum[keep].copy()
            joblib.dump(df_out, os.path.join(out_base, f"{st}_training_data.joblib"))