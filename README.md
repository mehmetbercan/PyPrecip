# 🌧️ PyPrecip  

![License: Research-Only](https://img.shields.io/badge/license-Research--Only-orange)
![Python](https://img.shields.io/badge/python-3.9+-blue)
![Status](https://img.shields.io/badge/status-early--stage-lightgrey)

`PyPrecip` is a Python toolkit for **precipitation nowcasting** and **forecasting**.  
It provides tools to **download**, **process**, and **model** precipitation data using AI-driven pipelines.  

> 🧪 Currently supports **station-based precipitation** (radar support coming soon).

---

## 🚀 Features

- 📥  Data downloaders (_coming soon_)  
- 🧹  Preprocessing and feature‑engineering pipelines  
- 🧠  Baseline and advanced AI models for nowcasting  
- 📈  Forecasting models (_planned_)  
- 🧪  Evaluation metrics, visualization, and interactive tools  

---

## ⚙️ Installation

If you’re new, please read [**Help for Novice Users.md**](Help%20for%20Novice%20Users.md) first.  
Then open a terminal (Command Prompt or PowerShell) in your project root (where `pyproject.toml` is) and run:

```bash
pip install -e .
```

---

## 🧩 Running Tests

```bash
cd PyPrecip/tests
pytest test_cli_organize.py::test_organize_tr_cmd
pytest test_cli_create_training.py::test_create_training_cmd
pytest test_cli_train_cum_evnt.py::test_train_cum_evnt_cmd
```

---

## 💻 Command‑Line Interface (CLI)

> **Note:**  
> When using `-c file/to/yaml`, make sure the specified YAML configuration file **exists** and is **up‑to‑date** for that command.

### 1️⃣ Organize Raw Data

Organize mixed precipitation data (e.g., from the Turkish State Meteorological Service) into the standard PyPrecip format:

```bash
pyprecip organize-tr -c examples/configs/organizer_tr_example.yaml
```

---

### 2️⃣ Create Training Inputs

Generate **event‑based** training inputs from the organized data:

```bash
pyprecip create-training -c examples/configs/create_training_example.yaml
```

#### 🧭 2.1  Interactive Config Builder

Launch the interactive visualization tool to build or adjust a YAML config **before** running the command above:

```bash
pyprecip config-builder-4-create-training -i D:/PROJECTS/PyPrecip/examples/outputs/organized
```

---

### 3️⃣ Train CNN Model

Train a convolutional model optimized for event‑based input (architecture config in YAML):

```bash
pyprecip train-cum-evnt -c examples/configs/train_cum_evnt_example.yaml
```

#### 🎨 3.1  Interactive Training Data Visualizer

Open the visualizer to explore target (`y`) data and fine‑tune class ranges homogeneously.  
Once intervals are updated, you can reuse the new YAML configuration from this tool when running `train-cum-evnt`.

```bash
pyprecip training-data-visualizer -c D:/PROJECTS/PyPrecip/examples/configs/train_cum_evnt_example.yaml
```

---

## 🧠 Notes

- All CLI commands support the `-c` flag for YAML‑based configs.  
- Interactive tools enable dynamic inspection + editing of parameters.
- Radar‑based datasets and forecasting models are **in active development**.

---

## 🧾 License

This research code is distributed under a **Research‑Only License**.  
See the license badge or accompanying documentation for details.