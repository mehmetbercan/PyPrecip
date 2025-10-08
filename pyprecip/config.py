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
    organized_dir: str = "outputs/organized"
    out_dir: str = "outputs/training_inputs"
    short_cols: List[str] = ("pcp", "rhum", "tmp")
    cols: List[str] = ("precip","rhum","tmp")
    min_precip_mm: float = 0.0
    max_precip_mm: float = 50.0
    min_tmp_c: float = -30.0
    max_tmp_c: float = 50.0
    min_rhum_pct: float = 0.0
    max_rhum_pct: float = 100.0
    mit_cold_hours: int = 1   # Oct–Mar
    mit_warm_hours: int = 2   # Apr–Sep
    global_event_buffer_hours: int = 1

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