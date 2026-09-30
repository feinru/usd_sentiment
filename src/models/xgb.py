import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import ParameterGrid

def directional_accuracy(y_true, y_pred):
    true_dir = np.sign(y_true)
    pred_dir = np.sign(y_pred)
    return np.mean(true_dir == pred_dir) * 100

def run_xgboost():
    train_path = "data/splits/train.csv"
    val_path = "data/splits/val.csv"
    
    if not os.path.exists(train_path):
        print("Error: Splits not found. Please run arimax.py first.")
        return
        
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # Target is the change in exchange rate
    y_train = train_df['kurs_diff']
    y_val = val_df['kurs_diff']
    
    exog_cols = [c for c in train_df.columns if c not in ['tanggal', 'kurs', 'kurs_diff', 'news_count']]
    X_train = train_df[exog_cols]
    X_val = val_df[exog_cols]
    
    print(f"\nTraining XGBoost on ALL {len(exog_cols)} features (predicting kurs_diff)...")
    
    param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.05],
        'max_depth': [3, 5]
    }
    
    best_rmse = float('inf')
    best_model = None
    best_params = None
    
    for params in ParameterGrid(param_grid):
        model = xgb.XGBRegressor(**params, random_state=42, objective='reg:squarederror')
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
    dir_acc = directional_accuracy(y_val, predictions)
    
    # Naive Baseline (predict 0 change, which is the same as predicting yesterday's level)
    naive_preds = np.zeros(len(y_val))
    naive_rmse = np.sqrt(mean_squared_error(y_val, naive_preds))
    naive_mae = mean_absolute_error(y_val, naive_preds)
    
    print(f"\n--- VALIDATION PERFORMANCE ---")
    print(f"Naive Baseline (Zero Diff): RMSE {naive_rmse:.2f} | MAE {naive_mae:.2f}")
    print(f"XGBoost Model:              RMSE {rmse:.2f} | MAE {mae:.2f}")
    print(f"Directional Accuracy:       {dir_acc:.1f}%")
    
    importance = pd.DataFrame({
        'Feature': exog_cols,
        'Importance': best_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(importance.head(10).to_string(index=False))

if __name__ == "__main__":
    run_xgboost()
