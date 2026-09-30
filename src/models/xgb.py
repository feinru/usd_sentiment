import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import ParameterGrid

def run_xgboost():
    train_path = "data/splits/train.csv"
    val_path = "data/splits/val.csv"
    
    print(f"Loading datasets from {train_path} and {val_path}...")
    if not os.path.exists(train_path):
        print("Error: Splits not found. Please wait for arimax.py to finish running first.")
        return
        
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    y_train = train_df['kurs']
    y_val = val_df['kurs']
    
    exog_cols = [c for c in train_df.columns if 'kurs' not in c and c != 'tanggal']
    X_train = train_df[exog_cols]
    X_val = val_df[exog_cols]
    
    print(f"\nTraining XGBoost on ALL {len(exog_cols)} features (Lexicon + SVD)...")
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7]
    }
    
    best_rmse = float('inf')
    best_model = None
    best_params = None
    
    print("Searching for best XGBoost parameters...")
    for params in ParameterGrid(param_grid):
        model = xgb.XGBRegressor(
            **params,
            random_state=42,
            objective='reg:squarederror'
        )
        model.fit(X_train, y_train, verbose=False)
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_params = params
            
    print(f"\nBest Parameters Found: {best_params}")
    
    predictions = best_model.predict(X_val)
    
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    mae = mean_absolute_error(y_val, predictions)
    
    print(f"\nValidation Performance:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE:  {mae:.2f}")

if __name__ == "__main__":
    run_xgboost()
