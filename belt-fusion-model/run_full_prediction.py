import pandas as pd
from pathlib import Path
from ml_model.predictor import MLPredictor

def run_full_prediction():
    base_dir = Path(".")
    features_path = base_dir / "data" / "processed" / "features.csv"
    
    if not features_path.exists():
        print("Features CSV not found. Please run main.py first.")
        return
        
    print("Loading features...")
    df = pd.read_csv(features_path)
    df["@timestamp"] = pd.to_datetime(df["@timestamp"], utc=True)
    
    print(f"Initializing predictor...")
    ml_predictor = MLPredictor(
        models_dir=str(base_dir / "models" / "saved_models"),
        thresholds_config_path=str(base_dir / "config" / "thresholds.json"),
    )
    
    print(f"Running predictions on {len(df)} records...")
    pred_df = ml_predictor.predict(df)
    
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    output_path = results_dir / "predictions.csv"
    pred_df.to_csv(output_path, index=False)
    
    print(f"Predictions saved to {output_path}")
    
    latest = pred_df.iloc[-1]
    print("\nLatest Prediction Summary:")
    print(f"Timestamp: {latest['@timestamp']}")
    print(f"Health Score: {latest['final_health']:.1f}%")
    print(f"Final RUL: {latest['final_rul_days']:.1f} days")
    print(f"Confidence: {latest['confidence_level']}")

if __name__ == "__main__":
    run_full_prediction()
