import os
import pandas as pd
import numpy as np
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

def directional_accuracy(y_true, y_pred):
    true_dir = np.sign(np.array(y_true))
    pred_dir = np.sign(np.array(y_pred))
    # Ignore days where true diff is exactly 0
    mask = true_dir != 0
    return np.mean(true_dir[mask] == pred_dir[mask]) * 100

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
    
    full_train = pd.concat([train_df, val_df], ignore_index=True)
    
    y_full_train = full_train['kurs_diff'].values
    y_test = test_df['kurs_diff'].values
    
    print("========================================")
    print(f"FINAL TEST SET EVALUATION ({len(y_test)} days)")
    print("========================================\n")
    
    # --- NAIVE BASELINES ---
    # 1. Random Walk (Zero Diff)
    naive_preds = np.zeros(len(y_test))
    naive_rmse = np.sqrt(mean_squared_error(y_test, naive_preds))
    naive_mae = mean_absolute_error(y_test, naive_preds)
    
    # 2. Majority Class Direction (Always UP or Always DOWN)
    majority_dir = 1 if np.sum(np.sign(y_test) > 0) > len(y_test) / 2 else -1
    maj_preds = np.full(len(y_test), majority_dir)
    maj_da = directional_accuracy(y_test, maj_preds)
    
    # 3. Persistence Direction (Same direction as yesterday)
    pers_preds = test_df['kurs_diff_lag1'].values
    pers_da = directional_accuracy(y_test, pers_preds)
    
    print(f"--- Naive Baselines ---")
    print(f"Zero Diff RMSE:                   {naive_rmse:.2f}")
    print(f"Zero Diff MAE:                    {naive_mae:.2f}")
    print(f"Majority Class DA:                {maj_da:.1f}%")
    print(f"Persistence (Yesterday's Dir) DA: {pers_da:.1f}%\n")
    
    # --- ARIMAX ---
    print(f"--- ARIMAX (Rolling Forecast) ---")
    warnings.simplefilter('ignore', ConvergenceWarning)
    warnings.filterwarnings("ignore")
    
    # Let's use 4 features now that we have standard scaler
    arimax_features = ['title_vader_compound_mean', 'title_lm_polarity_mean', 'text_vader_compound_mean', 'text_lm_polarity_mean']
    arimax_features = [c for c in arimax_features if c in full_train.columns]
    
    X_full_train_arimax = full_train[arimax_features].values
    X_test_arimax = test_df[arimax_features].values
    
    scaler = StandardScaler()
    X_full_train_arimax_scaled = scaler.fit_transform(X_full_train_arimax)
    X_test_arimax_scaled = scaler.transform(X_test_arimax)
    
    # Rolling 1-step ahead forecast
    history_y = list(y_full_train)
    history_X = list(X_full_train_arimax_scaled)
    preds_arimax = []
    
    for t in range(len(y_test)):
        model = ARIMA(endog=history_y, exog=history_X, order=(0, 0, 1))
        fitted = model.fit()
        yhat = fitted.forecast(steps=1, exog=X_test_arimax_scaled[t:t+1])[0]
        preds_arimax.append(yhat)
        history_y.append(y_test[t])
        history_X.append(X_test_arimax_scaled[t])
        
    rmse_arimax = np.sqrt(mean_squared_error(y_test, preds_arimax))
    mae_arimax = mean_absolute_error(y_test, preds_arimax)
    dir_acc_arimax = directional_accuracy(y_test, preds_arimax)
    
    print(f"RMSE:                 {rmse_arimax:.2f}")
    print(f"MAE:                  {mae_arimax:.2f}")
    print(f"Directional Accuracy: {dir_acc_arimax:.1f}%\n")

    # --- XGBoost ---
    print(f"--- XGBoost ---")
    xgb_features = [c for c in full_train.columns if c not in ['tanggal', 'kurs', 'kurs_diff', 'news_count']]
    
    X_full_train_xgb = full_train[xgb_features]
    X_test_xgb = test_df[xgb_features]
    
    model_xgb = xgb.XGBRegressor(
        learning_rate=0.01,
        max_depth=3,
        n_estimators=200,
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
