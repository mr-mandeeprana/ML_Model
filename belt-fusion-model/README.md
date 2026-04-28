# Belt Predictive Maintenance ML Pipeline

Predict health and Remaining Useful Life (RUL) for industrial conveyor belts.

## Core Concepts: Lifecycle Health vs. Operational Risk

This pipeline distinguishes between two fundamental metrics:

1.  **Lifecycle Health (ML-Driven)**:
    *   **Baseline**: 2190 days (6 years).
    *   **Definition**: A percentage representing how much of the belt's total expected life remains.
    *   **Behavior**: At 2190 days RUL, health is 100%. At 90 days RUL (operational critical), health is naturally low (~4.1%). This score is intended for long-term maintenance planning.
2.  **Operational Risk (Fusion-Driven)**:
    *   **Inputs**: Live Sensor Thresholds + ML RUL Predictions + Degradation Trends.
    *   **Industrial Policy**: **Sensor Critical => CRITICAL RISK**. The ML model identifies pre-failure signatures (WARNING), but hard safety thresholds always take precedence.
    *   **States**: NORMAL, WARNING, CRITICAL, and LOW_CONFIDENCE (on data quality or model failure).

## Repository Layout

```text
belt-fusion-model/
├── main.py                          # Unified final prediction (ML-only)
├── run_full_pipeline.ps1            # Orchestrates training and inference
├── config/
│   ├── belts_metadata.json          # Belt metadata for target generation
│   ├── model_config.json            # Model architecture and feature contract
│   └── thresholds.json              # Sensor and risk thresholds
├── data/
│   ├── bucket_elevator_synthetic_failure_1min_4months.csv
│   └── processed/                   # Cached features and cleaned data
├── ml_model/
│   ├── run_standalone.py            # Main training script
│   ├── data_preprocessing.py        # Long-to-wide cleaning
│   ├── feature_engineering.py       # Failure-episode aware features
│   ├── model_training.py            # Weighted, chronological training
│   └── predictor.py                 # NaN-safe inference wrapper
├── models/
│   └── saved_models/                # pkl artifacts
└── results/                         # Execution logs
```

## Setup & Running

### 1. Environment Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run Complete Pipeline (Recommended)
This script handles preprocessing, feature engineering, training, and final inference reporting.
```powershell
.\run_full_pipeline.ps1
```

### 3. Individual Components
*   **Training Only**: `python -m ml_model.run_standalone`
*   **Predictive Dashboard Output**: `python main.py`

## Industrial Refinements

- **Leakage Protection**: No rule-based flags or labels are used in the training feature set.
- **Robust Target Engineering**: RUL points to the **start of the next critical episode**, ensuring the model learns to anticipate failures.
- **Weighted Validation**: Uses inverse-frequency sample weighting and strict chronological holdout splitting to ensure real-world reliability.
- **Fail-Fast Safety**: Training aborts automatically if negative performance metrics ($R^2$) are detected.
