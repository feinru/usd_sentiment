import os
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import itertools
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

def run_arimax():
    features_path = "data/features/data.csv"
    rate_path = "data/dateadjusted/rate.csv"
    
    print(f"Loading features from {features_path}...")
    features_df = pd.read_csv(features_path)
    if 'kurs' in features_df.columns:
        features_df = features_df.drop(columns=['kurs'])
    features_df['tanggal'] = pd.to_datetime(features_df['tanggal'])
    
    print(f"Loading rates from {rate_path}...")
    rate_df = pd.read_csv(rate_path)
    rate_df['tanggal'] = pd.to_datetime(rate_df['tanggal'])
    
    data = pd.merge(rate_df, features_df, on='tanggal', how='left')
    data = data.sort_values('tanggal').reset_index(drop=True)
    
    # --- METHODOLOGICAL FIXES ---
    print("\nApplying Methodological Fixes (Shifting & Differencing)...")
    
    # 1. Target is the difference (change in exchange rate)
    data['kurs_diff'] = data['kurs'].diff()
    
    # 2. Add autoregressive lag (yesterday's diff)
    data['kurs_diff_lag1'] = data['kurs_diff'].shift(1)
    
    # 3. Shift ALL sentiment features by 1 to prevent lookahead bias (nowcasting -> forecasting)
    sentiment_cols = [c for c in data.columns if c not in ['tanggal', 'kurs', 'kurs_diff', 'kurs_diff_lag1']]
    data[sentiment_cols] = data[sentiment_cols].shift(1)
    
    # 4. Fill days without news with 0.0 (neutral sentiment)
    data[sentiment_cols] = data[sentiment_cols].fillna(0.0)
    
    # Drop rows with NaNs introduced by shifting and diffing (only checking target lags)
    data = data.dropna(subset=['kurs_diff', 'kurs_diff_lag1']).reset_index(drop=True)
    
    n = len(data)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    train_df = data.iloc[:train_end]
    val_df = data.iloc[train_end:val_end]
    test_df = data.iloc[val_end:]
    
    os.makedirs("data/splits", exist_ok=True)
    train_df.to_csv("data/splits/train.csv", index=False)
    val_df.to_csv("data/splits/val.csv", index=False)
    test_df.to_csv("data/splits/test.csv", index=False)
    
    print(f"Data split into 3 parts:")
    print(f"  Train: {train_df.shape[0]} rows")
    print(f"  Val:   {val_df.shape[0]} rows")
    print(f"  Test:  {test_df.shape[0]} rows")
    
    # --- ARIMAX MODELING ---
    # Target is now kurs_diff!
    y_train = train_df['kurs_diff']
    y_val = val_df['kurs_diff']
    
    # For ARIMAX we must severely limit features to prevent singular matrix errors (crashes).
    # Let's use just two main sentiment aggregations.
    selected_exog = ['title_vader_compound_mean', 'title_lm_polarity_mean']
    
    # Ensure they exist (fallback if missing)
    selected_exog = [c for c in selected_exog if c in train_df.columns]
    
    X_train = train_df[selected_exog].copy()
    X_val = val_df[selected_exog].copy()
    
    print(f"\nTraining ARIMAX on {len(selected_exog)} exogenous variables (predicting kurs_diff)...")
    
    warnings.simplefilter('ignore', ConvergenceWarning)
    warnings.filterwarnings("ignore")
    
    # Since target is already differenced, d=0 (ARMA model on diffs)
    p_values = [0, 1] 
    d_values = [0] 
    q_values = [0, 1]
    
    best_rmse = float('inf')
    best_order = None
    fitted_model = None
    
    for p, d, q in itertools.product(p_values, d_values, q_values):
        try:
            model = ARIMA(endog=y_train, exog=X_train, order=(p, d, q))
            fitted = model.fit()
            preds = fitted.forecast(steps=len(y_val), exog=X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_order = (p, d, q)
                fitted_model = fitted
        except Exception:
            continue
            
    print(f"\nBest ARIMA Order Found: {best_order}")
    
    if fitted_model is None:
        print("\nERROR: All ARIMA models failed to converge even with reduced features.")
        return
        
    print("\nModel Summary (abbreviated):")
    print(fitted_model.summary().tables[0])
    
    # 4. Naive Baseline (predict 0 change)
    naive_preds = np.zeros(len(y_val))
    naive_rmse = np.sqrt(mean_squared_error(y_val, naive_preds))
    naive_mae = mean_absolute_error(y_val, naive_preds)
    
    predictions = fitted_model.forecast(steps=len(y_val), exog=X_val)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    mae = mean_absolute_error(y_val, predictions)
    
    print(f"\n--- VALIDATION PERFORMANCE ---")
    print(f"Naive Baseline (Zero Diff): RMSE {naive_rmse:.2f} | MAE {naive_mae:.2f}")
    print(f"ARIMAX Model:               RMSE {rmse:.2f} | MAE {mae:.2f}")

if __name__ == "__main__":
    run_arimax()
