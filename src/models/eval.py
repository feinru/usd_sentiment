import os
import pandas as pd
import numpy as np
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

def directional_accuracy(y_true, y_pred):
    true_dir = np.sign(np.array(y_true))
    pred_dir = np.sign(np.array(y_pred))
    return np.mean(true_dir == pred_dir) * 100

def evaluate_test():
    train_path = "data/splits/train.csv"
    val_path = "data/splits/val.csv"
    test_path = "data/splits/test.csv"
    
    if not os.path.exists(test_path):
        print("Error: Splits not found.")
        return
        
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    # Combine Train + Val for final model training
    full_train = pd.concat([train_df, val_df], ignore_index=True)
    
    y_full_train = full_train['kurs_diff']
    y_test = test_df['kurs_diff']
    
    print("========================================")
    print(f"FINAL TEST SET EVALUATION ({len(y_test)} days)")
    print("========================================\n")
    
    # --- NAIVE BASELINE ---
    naive_preds = np.zeros(len(y_test))
    naive_rmse = np.sqrt(mean_squared_error(y_test, naive_preds))
    naive_mae = mean_absolute_error(y_test, naive_preds)
    naive_dir_acc = np.mean(np.sign(y_test) == 0) * 100
    
    print(f"--- Naive Baseline ---")
    print(f"RMSE:                 {naive_rmse:.2f}")
    print(f"MAE:                  {naive_mae:.2f}")
    
    
    # --- ARIMAX ---
    print(f"\n--- ARIMAX (0, 0, 1) ---")
    warnings.simplefilter('ignore', ConvergenceWarning)
    warnings.filterwarnings("ignore")
    
    arimax_features = ['title_vader_compound_mean', 'title_lm_polarity_mean']
    arimax_features = [c for c in arimax_features if c in full_train.columns]
    
    X_full_train_arimax = full_train[arimax_features]
    X_test_arimax = test_df[arimax_features]
    
    model_arimax = ARIMA(endog=y_full_train, exog=X_full_train_arimax, order=(0, 0, 1))
    fitted_arimax = model_arimax.fit()
    preds_arimax = fitted_arimax.forecast(steps=len(y_test), exog=X_test_arimax)
    
    rmse_arimax = np.sqrt(mean_squared_error(y_test, preds_arimax))
    mae_arimax = mean_absolute_error(y_test, preds_arimax)
    dir_acc_arimax = directional_accuracy(y_test, preds_arimax)
    
    print(f"RMSE:                 {rmse_arimax:.2f}")
    print(f"MAE:                  {mae_arimax:.2f}")
    print(f"Directional Accuracy: {dir_acc_arimax:.1f}%")


    # --- XGBoost ---
    print(f"\n--- XGBoost (lr=0.01, depth=5, trees=100) ---")
    xgb_features = [c for c in full_train.columns if c not in ['tanggal', 'kurs', 'kurs_diff']]
    
    X_full_train_xgb = full_train[xgb_features]
    X_test_xgb = test_df[xgb_features]
    
    model_xgb = xgb.XGBRegressor(
        learning_rate=0.01,
        max_depth=5,
        n_estimators=100,
        random_state=42,
        objective='reg:squarederror'
    )
    
    model_xgb.fit(X_full_train_xgb, y_full_train, verbose=False)
    preds_xgb = model_xgb.predict(X_test_xgb)
    
    rmse_xgb = np.sqrt(mean_squared_error(y_test, preds_xgb))
    mae_xgb = mean_absolute_error(y_test, preds_xgb)
    dir_acc_xgb = directional_accuracy(y_test, preds_xgb)
    
    print(f"RMSE:                 {rmse_xgb:.2f}")
    print(f"MAE:                  {mae_xgb:.2f}")
    print(f"Directional Accuracy: {dir_acc_xgb:.1f}%")
    print("========================================")

if __name__ == "__main__":
    evaluate_test()
