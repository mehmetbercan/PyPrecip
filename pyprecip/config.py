import yaml
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

@dataclass
class OrganizerConfig:
    mgm_hourly_folder: str
    outputs_dir: str
    ranges: Dict[str, Tuple[float, float]]
    # file names can be defined in YAML (eg. organizer_example_4TRstate.yaml)
    pcp_file:  str
    pcp_column:  str
    tmp_file: Optional[str] = None
    tmp_column: Optional[str] = None
    wnd_file: Optional[str] = None
    wnd_column: Optional[str] = None
    maxwnd_file: Optional[str] = None
    maxwnd_column: Optional[str] = None
    rhum_file: Optional[str] = None
    rhum_column: Optional[str] = None
    pressure_file: Optional[str] = None
    pressure_column: Optional[str] = None
    # Excel files (converted to JSON)
    radsum_file: Optional[str] = None
    radsum_column: Optional[str] = None
    rad_file: Optional[str] = None
    rad_column: Optional[str] = None
    insolationintensity_file: Optional[str] = None
    insolationintensity_column: Optional[str] = None
    insolationtime_file: Optional[str] = None
    insolationtime_column: Optional[str] = None
    minsoiltmp0cm_file: Optional[str] = None
    minsoiltmp0cm_column: Optional[str] = None
    soiltmp100cm_file: Optional[str] = None
    soiltmp100cm_column: Optional[str] = None
    soiltmp50cm_file: Optional[str] = None
    soiltmp50cm_column: Optional[str] = None
    soiltmp20cm_file: Optional[str] = None
    soiltmp20cm_column: Optional[str] = None
    soiltmp10cm_file: Optional[str] = None
    soiltmp10cm_column: Optional[str] = None
    soiltmp5cm_file: Optional[str] = None
    soiltmp5cm_column: Optional[str] = None

@dataclass
class CreateTrainingConfig:
    stations: List[int]
    organized_dir: str
    out_dir: str
    short_cols: List[str]
    cols: List[str]

    mit_cold_hours: int
    mit_warm_hours: int
    global_event_buffer_hours: int
    event_total_pcp_threshold_4_1hr_event: float
    event_total_pcp_threshold_4_larger_events: float

    use_normalized_data: Optional[bool] = None
    min_max_precip: Optional[Tuple[float, float]]  = None
    interpolate_tmp_rhum: Optional[bool] = None
    calculate_diff_tmp_rhum: Optional[bool] = None
    rolling_mean_diff_tmp_rhum: Optional[bool] = None
    rolling_mean_diff_tmp_rhum_hour: Optional[int]  = None

@dataclass
class TrainConfig:
    stations: List[int]
    target_station: int
    feature_cols: List[str] = ("cum_pcp", "pcp_mm", "rhum", "tmp", "rhum_diff", "tmp_diff", "tmp_diff_roll10h", "rhum_diff_roll10h", "cum_hour")
    train_col: str = "cum_pcp"
    inputs_dir: str = "outputs/training_inputs"
    model_dir: str = "outputs/models/cnn/v1"
    batch_size: int = 256
    epochs: int = 500
    patience: int = 10000
    hidden_units: int = 64
    class_intervals: Optional[list] = None  # list of (a,b)

def load_yaml(path, cls):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return cls(**data)