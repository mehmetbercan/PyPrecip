# PyPrecip

![License: Research-Only](https://img.shields.io/badge/license-Research--Only-orange)

`PyPrecip` is a Python library for downloading, processing, and building AI models for precipitation **nowcasting** and **forecasting**.  
It starts with **station-based precipitation** and will later include **radar-based precipitation**.

---

## Features 

- Data downloaders for precipitation stations and radar (planned). 
- Preprocessing and feature engineering pipelines.
- Baseline and advanced AI models for nowcasting.
- Baseline and advanced AI models for forecasting (planned).
- Evaluation metrics and visualization tools.

---

## How to install and run
For beginners, please read [Help for Novice Users.md](Help%20for%20Novice%20Users.md) first. Then, Open Command Prompt (CMD) in your project folder and run the following from the project root (where pyproject.toml is located):
```bash 
pip install -e .
```

### Running Tests 
```bash 
cd PyPrecip\tests
pytest test_cli_organize.py::test_organize_tr_cmd
pytest test_cli_create_training.py::test_create_training_cmd
pytest test_cli_train_cum_evnt.py::test_train_cum_evnt_cmd
```

## Command-line interface (CLI) usage: 
Note: If you include `-c file/to/yaml` in your command, ensure that the corresponding YAML file exists or has been updated for that command.
### 1) Organize raw data  
Use the following command to organize mixed data format from the Turkish State Meteorological Service to standard PyPrecip input format:
```bash 
pyprecip organize-tr -c examples/configs/organizer_tr_example.yaml 
``` 

### 2) Create training inputs 
Use the following command to generate event-based training inputs from the organized data in the standard PyPrecip training input format:
```bash 
pyprecip create-training -c examples/configs/create_training_example.yaml  
```
#### 2.1) Visualize data and create configuration interactively  

You can launch the visualization tool to build and preview a YAML configuration interactively before running the "create-training" command above:

```bash  
pyprecip config-builder-4-create-training -i D:/PROJECTS/PyPrecip/examples/outputs/organized  
```


### 3) Train CNN  
Use the following command to train a model optimized for event-based input data (with optional modifications to the CNN architecture in the YAML file):
```bash 
pyprecip train-cum-evnt -c examples/configs/train_cum_evnt_example.yaml  
```



