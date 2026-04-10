"""
Run ML Standalone Pipeline
"""
import sys
import os
from pathlib import Path

# Add project root to sys.path to allow running as a standalone script
root_path = str(Path(__file__).parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

from ml_model.data_preprocessing import DataPreprocessor
from ml_model.feature_engineering import FeatureEngineer
from ml_model.model_training import MLTrainer
from ml_model.predictor import MLPredictor

from pathlib import Path
import os
import pandas as pd

def run_ml_pipeline():
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "data" / "data_25_transformed.csv"
    
    if not data_path.exists():
        print(f"Data not found at {data_path}. Attempting to copy from existing projects as fallback...")
        fallback_data = base_dir.parent / "model documentation Belt" / "data" / "data_25_transformed.csv"
        if fallback_data.exists():
            (base_dir / "data").mkdir(exist_ok=True)
            import shutil
            shutil.copy(fallback_data, data_path)
            print(f"Copied test data to {data_path}")
        else:
            print("No test data found. Standalone pipeline requires data_25_transformed.csv.")
            return
        
    print("1. Preprocessing data...")
    preprocessor = DataPreprocessor(str(base_dir / "config" / "thresholds.json"))
    cleaned_df = preprocessor.preprocess(str(data_path))
    cleaned_path = base_dir / "data" / "processed" / "cleaned_sensor_data.csv"
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(cleaned_path, index=False)
    
    print("2. Feature engineering...")
    engineer = FeatureEngineer(
        str(base_dir / "config" / "belts_metadata.json"),
        str(base_dir / "config" / "thresholds.json")
    )
    features_df = engineer.run(str(cleaned_path))
    features_path = base_dir / "data" / "processed" / "features.csv"
    features_df.to_csv(features_path, index=False)
    
    print("3. Training ML Models (Health and RUL)...")
    trainer = MLTrainer(str(base_dir / "config" / "model_config.json"))
    metrics = trainer.train_and_validate(str(features_path))
    
    # Format metrics output with R² and Adjusted R²
    print("\n" + "="*90)
    print("ML MODEL TRAINING METRICS")
    print("="*90)
    print(f"Samples: Train={metrics.get('n_train', 'N/A')}, Test={metrics.get('n_test', 'N/A')}, Features={metrics.get('n_features', 'N/A')}")
    print(f"Split Date (UTC): {metrics.get('split_time_utc', 'N/A')}")
    print("-"*90)
    print(f"{'Metric':<25} | {'Train R2':<12} | {'Train Adj-R2':<12} | {'Test R2':<12} | {'Test Adj-R2':<12}")
    print("-"*90)
    
    h_train_r2 = metrics.get('health_train_r2', 0)
    h_train_adj_r2 = metrics.get('health_train_adj_r2', 0)
    h_test_r2 = metrics.get('health_test_r2', 0)
    h_test_adj_r2 = metrics.get('health_test_adj_r2', 0)
    r_train_r2 = metrics.get('rul_train_r2', 0)
    r_train_adj_r2 = metrics.get('rul_train_adj_r2', 0)
    r_test_r2 = metrics.get('rul_test_r2', 0)
    r_test_adj_r2 = metrics.get('rul_test_adj_r2', 0)
    
    print(f"{'Health Score':<25} | {h_train_r2:>12.4f} | {h_train_adj_r2:>12.4f} | {h_test_r2:>12.4f} | {h_test_adj_r2:>12.4f}")
    print(f"{'RUL (days)':<25} | {r_train_r2:>12.4f} | {r_train_adj_r2:>12.4f} | {r_test_r2:>12.4f} | {r_test_adj_r2:>12.4f}")
    print("-"*90)
    
    # Feature usefulness check: R² vs Adj-R² penalty for feature count
    h_r2_penalty = (h_train_r2 - h_train_adj_r2) * 100  # In percentage points
    r_r2_penalty = (r_train_r2 - r_train_adj_r2) * 100
    
    n_features = metrics.get('n_features', 'N/A')
    print(f"Feature Usefulness Check (R2 vs Adjusted R2 penalty for {n_features} features):")
    print(f"  Health: R2 penalty = {h_r2_penalty:.3f}% (Adj-R2: {h_train_adj_r2:.4f})")
    print(f"  RUL:    R2 penalty = {r_r2_penalty:.3f}% (Adj-R2: {r_train_adj_r2:.4f})")
    
    if h_r2_penalty < 0.01 and r_r2_penalty < 0.01:
        print(f"  [OK] All {n_features} features are valuable (negligible penalty < 0.01%)")
    elif h_r2_penalty < 0.1 and r_r2_penalty < 0.1:
        print("  [OK] Features are effective (minimal penalty < 0.1%)")
    else:
        print("  [WARN] Some weak features detected (consider feature selection)")
    
    # Time-series generalization check
    h_gen_gap = h_train_r2 - h_test_r2
    r_gen_gap = r_train_r2 - r_test_r2
    print(f"\nTime-Series Generalization (Train->Test R2 gap):")
    print(f"  Health: {h_gen_gap:.4f} (train accurate on historical, test on future data)")
    print(f"  RUL:    {r_gen_gap:.4f} (expected gap in time-series models)")
    
    if h_gen_gap < 0.15 and r_gen_gap < 0.15:
        print("  [OK] Good generalization (gap < 0.15)")
    else:
        print("  [WARN] Notable gap (consider model complexity or data distribution changes)")
    
    print("="*90 + "\n")
    
    print("4. Testing ML Predictor (Inference)...")
    predictor = MLPredictor(str(base_dir / "models" / "saved_models"))
    
    # Predict on the last 10 rows
    tail_features = features_df.tail(10)
    preds = predictor.predict(tail_features)
    
    print("\n--- ML Model Standalone Output (Last 5 records) ---")
    output_cols = ['@timestamp', 'ml_health', 'ml_rul_days']
    print(preds[output_cols].tail(5).to_string(index=False))

if __name__ == "__main__":
    # Ensure working from correct root
    os.chdir(Path(__file__).parent.parent)
    run_ml_pipeline()
