# Belt Health and RUL Fusion Model

This repository predicts conveyor belt health and Remaining Useful Life (RUL) using a hybrid approach:
- Machine Learning model (data-driven)
- Physics model (power law + Arrhenius thermal acceleration)
- Adaptive fusion model (combines ML + physics)

The project includes standalone runners for each model and a unified pipeline entrypoint.

## What This Project Does

- Preprocesses raw sensor data and builds engineered features
- Trains and evaluates ML models for health and RUL
- Estimates physics-based degradation using:
  - Power law shore-hardness aging
  - Arrhenius temperature acceleration
  - Optional elongation damage blending
- Fuses ML and physics outputs into final health and RUL predictions
- Classifies risk level and confidence

## Repository Layout

```text
belt-fusion-model/
├── main.py                          # Unified final prediction (ML + Physics + Fusion)
├── run_full_pipeline.ps1            # Orchestrates all stages and writes log files
├── requirements.txt
├── config/
│   ├── belts_metadata.json
│   ├── model_config.json
│   └── thresholds.json
├── data/
│   ├── belt_12.csv                  # Raw input data
│   └── processed/
│       ├── cleaned_sensor_data.csv
│       └── features.csv
├── ml_model/
│   ├── run_standalone.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── predictor.py
├── physics_model/
│   ├── run_standalone.py
│   ├── physics_engine.py
│   ├── shore_hardness.py
│   └── arrhenius.py
├── fusion_model/
│   ├── run_fusion.py
│   ├── fusion_engine.py
│   ├── risk_classifier.py
│   └── confidence_scorer.py
├── models/
│   └── saved_models/
└── results/
    └── ml_metrics_output.txt
```

## Prerequisites

- Python 3.10+ (recommended)
- PowerShell (for `run_full_pipeline.ps1`)

## Setup

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

## Running the Project

## Option A: Run complete orchestrated pipeline (recommended)

```powershell
.\run_full_pipeline.ps1
```

What it runs in order:
1. `python -m ml_model.run_standalone`
2. `python -m physics_model.run_standalone`
3. `python -m fusion_model.run_fusion`
4. `python main.py`

A timestamped log is saved under `results/` (for example: `pipeline_run_YYYYMMDD_HHMMSS.log`).

## Option B: Run modules individually

### ML standalone

```powershell
python -m ml_model.run_standalone
```

Expected behavior:
- Preprocesses `data/belt_12.csv`
- Writes:
  - `data/processed/cleaned_sensor_data.csv`
  - `data/processed/features.csv`
- Trains ML models and prints training/test metrics
- Runs inference on recent rows

### Physics standalone

```powershell
python -m physics_model.run_standalone
```

Expected behavior:
- Loads `data/processed/features.csv`
- Computes physics health and physics RUL
- Prints recent predictions

### Fusion standalone

```powershell
python -m fusion_model.run_fusion
```

Expected behavior:
- Generates ML and physics baselines
- Applies adaptive fusion
- Adds risk level and confidence score
- Prints recent fused predictions

### Unified final output

```powershell
python main.py
```

Expected behavior:
- Uses cached `data/processed/features.csv` if available, otherwise generates features
- Runs ML + physics + fusion on latest window
- Prints a final comparison table:
  - ML health/RUL
  - Physics health/RUL
  - Fusion health/RUL
  - Risk level
  - Confidence

## Input Data Requirements

Primary raw file:
- `data/belt_12.csv`

Core expected columns used by current pipeline:
- `temperature_boot_material/temperature` or `temperature_boot_material/temperature_avg`
- `current_transducer_head/current_avg`
- `ultrasonic_boot/elongation_avg`
- `@timestamp` (recommended for readable output)

## Key Configuration

All config files are in `config/`:

- `thresholds.json`
  - Physics parameters (e.g., `k`, `n`, `h_install`, `h_fail`)
  - Arrhenius constants and threshold values
- `model_config.json`
  - ML model and training settings
- `belts_metadata.json`
  - Belt metadata used during feature engineering

## Physics Model Notes

Implemented core concepts:
- Power law degradation: `ΔShore = k * t_eff^n`
- Arrhenius acceleration from material temperature
- Effective aging time accumulation
- RUL derived from remaining effective life to critical hardness

For deeper explanation, see:
- `PHYSICS_MODEL_DOCUMENTATION.md`
- `POWER_LAW_IMPLEMENTATION_SUMMARY.md`
- `IMPLEMENTATION_COMPLETE_SUMMARY.md`

## Troubleshooting

- `Features data not found ...`:
  - Run `python -m ml_model.run_standalone` first, or run full pipeline script.

- `Data not found at data/belt_12.csv`:
  - Ensure `data/belt_12.csv` exists.

- `Python executable not found` in orchestrator:
  - Pass explicit python path:

```powershell
.\run_full_pipeline.ps1 -UseVenv:$false -PythonPath "C:\Path\To\python.exe"
```

- Missing module errors:
  - Re-activate venv and reinstall dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Typical Workflow

1. Install dependencies.
2. Run `.\run_full_pipeline.ps1`.
3. Review terminal outputs and log files in `results/`.
4. Tune config values in `config/` if needed.
5. Re-run pipeline and compare output trends.

## Notes

- The unified entrypoint uses recent rows for lightweight ML/fusion output while preserving full-history context for physics aging.
- Existing generated artifacts under `data/processed/` and `models/saved_models/` are reused when present.
