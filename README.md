# PyPrecip

![License: Research-Only](https://img.shields.io/badge/license-Research--Only-orange)

`PyPrecip` is a Python library for downloading, processing, and building AI models for precipitation **nowcasting** and **forecasting**.  
It starts with **station-based precipitation** and will later include **radar-based precipitation**.

---

## Features (planned)

- Data downloaders for precipitation stations (and later radar).
- Preprocessing and feature engineering pipelines.
- Baseline and advanced AI models for nowcasting/forecasting.
- Evaluation metrics and visualization tools.

---


## How to install and run
From the project root (where `pyproject.toml` resides):
```bash 
pip install -e .
```
## Command-line interface (CLI) usage: 

### 1) Organize raw data  
```bash 
pyprecip organize -c examples/configs/organizer_example.yaml 
``` 

### 2) Create event-based training inputs  
```bash 
pyprecip create-training -c examples/configs/create_training_example.yaml  
```

### 3) Train CNN  
```bash 
pyprecip train -c examples/configs/train_example.yaml  
```



