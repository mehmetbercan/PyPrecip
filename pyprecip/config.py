import yaml
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class OrganizerConfig:
    mgm_hourly_folder: str
    outputs_dir: str = "outputs/organized"
    # file names can be overridden in YAML
    prp_file: str = "20240604C169-Saatlik Toplam Yağış (mm=kg÷m²) OMGİ.txt"
    tmp_file: str = "20240604C169-Saatlik Sıcaklık (°C).txt"
    wnd_file: str = "20240604C169-Saatlik Rüzgar Yönü ve Hızı (m÷sn).txt"
    maxwnd_file: str = "20240604C169-Saatlik Maksimum Rüzgarın hizı ve yönü (m÷sn).txt"
    rhum_file: str = "20240604C169-Saatlik Nispi Nem (%) .txt"
    pressure_file: str = "20240604C169-Saatlik Aktüel Basınç (hPa).txt"
    # Excel files (converted to JSON externally or via reader)
    radsum_file: str = "20240604C169-Saatlik Toplam Küresel Güneş Radyasyonu (watt÷m²).xlsx"
    rad_file: str = "20240604C169-Saatlik Küresel Güneş Radyasyonu (wattsaat÷m²).xlsx"
    insolationintensity_file: str = "20240604C169-Saatlik Güneşlenme Şiddeti (cal÷cm²).xlsx"
    insolationtime_file: str = "20240604C169-Saatlik Güneşlenme Süresi (saat).xlsx"
    minsoiltmp0cm_file: str = "20240604C169-Saatlik Toprak Üstü Minimum Sıcaklık (°C).xlsx"
    soiltmp100cm_file: str = "20240604C169-Saatlik 100 cm Toprak Sıcaklığı (°C).xlsx"
    soiltmp50cm_file: str = "20240604C169-Saatlik 50 cm Toprak Sıcaklığı (°C).xlsx"
    soiltmp20cm_file: str = "20240604C169-Saatlik 20 cm Toprak Sıcaklığı (°C).xlsx"
    soiltmp10cm_file: str = "20240604C169-Saatlik 10 cm Toprak Sıcaklığı (°C).xlsx"
    soiltmp5cm_file: str = "20240604C169-Saatlik 5 cm Toprak Sıcaklığı (°C).xlsx"

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