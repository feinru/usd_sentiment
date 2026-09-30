"""
Baseline TIME-ONLY: hanya histori kurs, tanpa berita.

Model:
  1. naive_majority     : selalu prediksi kelas mayoritas train (lower bound).
  2. naive_persistence  : prediksi = arah kemarin (ret_1 > 0). Uji momentum FX.
  3. logreg             : LogisticRegression + StandardScaler (linear wajar).
  4. gradboost          : HistGradientBoostingClassifier (pengganti XGBoost
                          berbasis sklearn agar bebas dependency berat;
                          API/hyperparam setara untuk baseline awal).

Jalankan setelah build_dataset.py:
  python -m src.models.train_baseline
Output: outputs/results/baseline_metrics.json + .csv, model di outputs/models/.
"""

import json
import os

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.metrics import classification_metrics, summarize_results

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPLIT_DIR = os.path.join(ROOT, "data", "splits")
RES_DIR = os.path.join(ROOT, "outputs", "results")
MODEL_DIR = os.path.join(ROOT, "outputs", "models")


def load_splits():
    """Muat train/val/test + daftar fitur tech."""
    train = pd.read_csv(os.path.join(SPLIT_DIR, "train.csv"), parse_dates=["tanggal"])
    val = pd.read_csv(os.path.join(SPLIT_DIR, "val.csv"), parse_dates=["tanggal"])
    test = pd.read_csv(os.path.join(SPLIT_DIR, "test.csv"), parse_dates=["tanggal"])
    with open(os.path.join(SPLIT_DIR, "feature_list.json")) as f:
        groups = json.load(f)
    return train, val, test, groups["tech"]


def persistence_pred(df: pd.DataFrame):
    """Prediksi naive: arah kemarin (ret_1 > 0 -> NAIK)."""
    return (df["ret_1"] > 0).astype(int).values


def train_and_eval():
    """Latih semua baseline time-only, evaluasi di val & test."""
    train, val, test, tech_cols = load_splits()
    Xtr, ytr = train[tech_cols].values, train["direction"].values
    Xva, yva = val[tech_cols].values, val["direction"].values
    Xte, yte = test[tech_cols].values, test["direction"].values

    models = {
        "logreg": make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, C=1.0)),
        "gradboost": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                    max_depth=None, random_state=42),
    }
    results = {}
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Naive baselines (tanpa training)
    maj = int(train["direction"].mode()[0])
    for split_name, y, df in (("val", yva, val), ("test", yte, test)):
        results[f"naive_majority@{split_name}"] = classification_metrics(
            y, [maj] * len(y))
        results[f"naive_persistence@{split_name}"] = classification_metrics(
            y, persistence_pred(df))

    # Model terlatih
    for name, clf in models.items():
        clf.fit(Xtr, ytr)
        joblib.dump(clf, os.path.join(MODEL_DIR, f"baseline_{name}.joblib"))
        for split_name, X, y in (("val", Xva, yva), ("test", Xte, yte)):
            proba = clf.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else None
            results[f"{name}@{split_name}"] = classification_metrics(
                y, clf.predict(X), proba)

    os.makedirs(RES_DIR, exist_ok=True)
    with open(os.path.join(RES_DIR, "baseline_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(results).T.to_csv(os.path.join(RES_DIR, "baseline_metrics.csv"))
    print(summarize_results(results))
    print(f"\nModel -> {MODEL_DIR}/ | metrik -> {RES_DIR}/baseline_metrics.*")
    return results


if __name__ == "__main__":
    train_and_eval()
