"""
Model GABUNGAN AWAL: histori kurs + fitur NLP (lexicon & TF-IDF).

Ablation (kolom yang dipakai tiap varian):
  - tech_only        : TECH saja (rujukan, harus ≈ baseline gradboost)
  - +lex_title       : TECH + VADER/LM dari JUDUL
  - +lex_full        : TECH + VADER/LM dari ISI
  - +lex_all         : TECH + lex judul + lex isi
  - +all             : TECH + semua lex + TF-IDF-SVD judul & isi
  - +tfidf_only      : TECH + TF-IDF-SVD saja (tanpa lexicon)

Setiap varian dilatih dengan 2 learner (logreg, gradboost) agar efek fitur
terpisah dari efek pilihan model. Evaluasi di val & test.

Jalankan setelah build_dataset.py:
  python -m src.models.train_combined
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


def load_all():
    train = pd.read_csv(os.path.join(SPLIT_DIR, "train.csv"), parse_dates=["tanggal"])
    val = pd.read_csv(os.path.join(SPLIT_DIR, "val.csv"), parse_dates=["tanggal"])
    test = pd.read_csv(os.path.join(SPLIT_DIR, "test.csv"), parse_dates=["tanggal"])
    with open(os.path.join(SPLIT_DIR, "feature_list.json")) as f:
        groups = json.load(f)
    return train, val, test, groups


def variants(groups: dict) -> dict:
    """Daftar varian fitur untuk ablation judul vs isi vs TF-IDF."""
    t, lt, lf = groups["tech"], groups["lex_title"], groups["lex_full"]
    tt, tf = groups["tfidf_title"], groups["tfidf_full"]
    return {
        "tech_only": t,
        "+lex_title": t + lt,
        "+lex_full": t + lf,
        "+lex_all": t + lt + lf,
        "+tfidf_only": t + tt + tf,
        "+all": t + lt + lf + tt + tf,
    }


def train_and_eval():
    train, val, test, groups = load_all()
    ytr, yva, yte = train["direction"].values, val["direction"].values, test["direction"].values
    results = {}
    os.makedirs(MODEL_DIR, exist_ok=True)

    for vname, cols in variants(groups).items():
        Xtr, Xva, Xte = train[cols].values, val[cols].values, test[cols].values
        learners = {
            "logreg": make_pipeline(StandardScaler(),
                                    LogisticRegression(max_iter=2000, C=1.0)),
            "gradboost": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                        max_depth=None, random_state=42),
        }
        for lname, clf in learners.items():
            clf.fit(Xtr, ytr)
            safe = vname.replace("+", "plus")
            joblib.dump(clf, os.path.join(MODEL_DIR, f"combined_{safe}_{lname}.joblib"))
            for split_name, X, y in (("val", Xva, yva), ("test", Xte, yte)):
                proba = clf.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else None
                results[f"{vname}/{lname}@{split_name}"] = classification_metrics(
                    y, clf.predict(X), proba)

    os.makedirs(RES_DIR, exist_ok=True)
    with open(os.path.join(RES_DIR, "combined_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(results).T.to_csv(os.path.join(RES_DIR, "combined_metrics.csv"))
    print(summarize_results(results))
    print(f"\nMetrik -> {RES_DIR}/combined_metrics.*")
    return results


if __name__ == "__main__":
    train_and_eval()
