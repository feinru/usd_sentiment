"""
Metrik evaluasi: klasifikasi arah (utama) + regresi (ekstensi).

Tugas utama proposal = Klasifikasi Biner NAIK/TURUN:
  - accuracy == directional accuracy (proporsi arah benar)
  - precision / recall / f1 (binary & macro)
  - confusion matrix
  - AUC (jika probabilitas tersedia)

Fungsi regresi (rmse/mae) disediakan untuk ekstensi
"prediksi kurs besok" tanpa mengubah API evaluasi.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
)


def classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Hitung metrik klasifikasi arah.

    Args:
        y_true: label benar (0/1).
        y_pred: label prediksi (0/1).
        y_proba: probabilitas kelas 1 (opsional, untuk AUC).

    Returns:
        Dict metrik + confusion matrix sebagai list [[TN, FP], [FN, TP]].
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    out = {
        "n": int(len(y_true)),
        "positif_rate_true": float(y_true.mean()) if len(y_true) else 0.0,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "directional_accuracy": float(accuracy_score(y_true, y_pred)),  # alias eksplisit
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_proba is not None:
        try:
            out["auc"] = float(roc_auc_score(y_true, np.asarray(y_proba)))
        except ValueError:
            out["auc"] = None  # mis. hanya satu kelas di split kecil
    else:
        out["auc"] = None
    return out


def regression_metrics(y_true, y_pred) -> dict:
    """Hitung RMSE/MAE untuk formulasi regresi (nilai penutupan besok)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "n": int(len(y_true)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def summarize_results(results: dict) -> str:
    """Format ringkas dict {nama_model: metrik} menjadi tabel teks."""
    lines = [f"{'model':35s} {'acc':>6s} {'f1':>6s} {'f1_mac':>7s} {'auc':>6s}"]
    for name, m in results.items():
        auc = f"{m['auc']:.3f}" if m.get("auc") is not None else "  -  "
        lines.append(f"{name:35s} {m['accuracy']:6.3f} {m['f1']:6.3f} {m['f1_macro']:7.3f} {auc:>6s}")
    return "\n".join(lines)
