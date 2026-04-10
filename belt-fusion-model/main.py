"""
Unified Entry Point for Belt Health & RUL ML Model
Runs the ML model and reports predicted belt health and remaining useful life.
"""
from ml_model.predictor import MLPredictor
from ml_model.data_preprocessing import DataPreprocessor
from ml_model.feature_engineering import FeatureEngineer

from pathlib import Path
import os
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

def main():
    print("="*70)
    print("     BELT HEALTH & RUL INITIALIZATION (ML ONLY)")
    print("="*70)
    
    base_dir = Path(__file__).parent
    data_path = base_dir / "data" / "processed" / "features.csv"
    
    # 1. Pipeline execution or load existing data
    if not data_path.exists():
        print("Preprocessed features not found. Running data pipeline first...")
        raw_data = base_dir / "data" / "data_25_transformed.csv"
        if not raw_data.exists():
             print(f"ERROR: Raw data missing at {raw_data}")
             return
             
        preprocessor = DataPreprocessor(str(base_dir / "config" / "thresholds.json"))
        engineer = FeatureEngineer(
            str(base_dir / "config" / "belts_metadata.json"),
            str(base_dir / "config" / "thresholds.json")
        )
        
        print("Preprocessing data...")
        cleaned = preprocessor.preprocess(str(raw_data))
        print("Feature engineering...")
        df = engineer.run(str(cleaned))
        
        # Save for future runs
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)
    else:
        print("Loading cached feature data...")
        df = pd.read_csv(data_path)
        
    print(f"Data ready: {len(df)} records.\n")
    
    print("Initializing ML Predictor...")
    # Initialize ML model
    ml_predictor = MLPredictor(str(base_dir / "models" / "saved_models"))
    
    print("Running predictions (taking latest record)...")

    # Keep runtime light for ML
    recent_df = df.tail(10).copy()
    
    # ML Model
    recent_df = ml_predictor.predict(recent_df)
    
    # Extract latest results
    latest = recent_df.iloc[-1]
    
    ml_health = latest.get('ml_health', 0.0)
    ml_rul = latest.get('ml_rul_days', 0.0)
    
    # Print Result Table
    print("\n" + "+" + "-"*62 + "+")
    print("|" + "BELT HEALTH & RUL - ML PREDICTION".center(62) + "|")
    print("+" + "-"*62 + "+")
    print(f"| {'Item':<15} | {'Value':<12} | {'Description':<27} |")
    print("+" + "-"*62 + "+")
    print(f"| {'Health Score':<15} | {ml_health:>12.1f} | {'Predicted belt health %':<27} |")
    print(f"| {'RUL (days)':<15} | {ml_rul:>12.0f} | {'Remaining Useful Life':<27} |")
    print(f"| {'RUL (months)':<15} | {ml_rul/30.4:>12.1f} | {'Approximate months rem.':<27} |")
    print("+" + "-"*62 + "+\n")

if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    main()
