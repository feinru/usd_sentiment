import os
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import Ridge
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
    
    data = pd.merge(rate_df, features_df, on='tanggal', how='inner')
    
    data = data.sort_values('tanggal').reset_index(drop=True)
    
    data = data.ffill().fillna(0)
    
    output_data_path = "data/features/data.csv"
    data.to_csv(output_data_path, index=False)
    print(f"\nMerged dataset saved to {output_data_path} with shape {data.shape}")
    
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
    
    y_train = train_df['kurs']
    y_val = val_df['kurs']
    
    exog_cols = [c for c in data.columns if c not in ['tanggal', 'kurs']]
    X_train = train_df[exog_cols].copy()
    X_val = val_df[exog_cols].copy()
    
    svd_cols = [c for c in exog_cols if 'svd' in c]
    print(f"\nTraining Ridge Regression on {len(svd_cols)} TF-IDF SVD features to create a meta-feature...")
    
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train[svd_cols], y_train)
    
    X_train['tfidf_meta_pred'] = ridge.predict(X_train[svd_cols])
    X_val['tfidf_meta_pred'] = ridge.predict(X_val[svd_cols])
    
    selected_exog = [c for c in exog_cols if 'lm' in c or 'vader' in c] + ['tfidf_meta_pred']
    
    print(f"\nTraining ARIMAX on {len(selected_exog)} exogenous variables: {selected_exog}")
    print("Searching for best ARIMAX (p,d,q) parameters...")
    
    warnings.simplefilter('ignore', ConvergenceWarning)
    warnings.filterwarnings("ignore")
    
    p_values = [0, 1, 2]
    d_values = [0, 1]
    q_values = [0, 1, 2]
    
    best_rmse = float('inf')
    best_order = None
    fitted_model = None
    
    for p, d, q in itertools.product(p_values, d_values, q_values):
        try:
            model = ARIMA(endog=y_train, exog=X_train[selected_exog], order=(p, d, q))
            fitted = model.fit()
            preds = fitted.forecast(steps=len(y_val), exog=X_val[selected_exog])
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_order = (p, d, q)
                fitted_model = fitted
        except Exception:
            continue
            
    print(f"\nBest ARIMA Order Found: {best_order}")
    
    print("\nModel Summary (abbreviated):")
    print(fitted_model.summary().tables[0])
    
    predictions = fitted_model.forecast(steps=len(y_val), exog=X_val[selected_exog])
    
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    mae = mean_absolute_error(y_val, predictions)
    
    print(f"\nValidation Performance:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE:  {mae:.2f}")

if __name__ == "__main__":
    run_arimax()
