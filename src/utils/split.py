"""
Split kronologis & pembuatan target untuk data time-series USD/IDR.

Aturan anti-leakage:
  1. Target hari T memakai kurs T+1 (shift(-1)) -> tidak ada info masa depan di fitur.
  2. Split berdasarkan URUTAN WAKTU (tanpa shuffle). Model hanya lihat masa lalu.
  3. Vectorizer/scaler di-fit HANYA di train (dilakukan di build_dataset.py).

Rasio default: 70% train / 15% val / 15% test (kronologis).
"""

import json
import os
import pandas as pd


def make_target(rate_df: pd.DataFrame) -> pd.DataFrame:
    """Buat kolom target klasifikasi arah dari dataframe kurs.

    Kolom baru:
      - kurs_next   : kurs hari trading berikutnya
      - ret_next    : return (kurs_next - kurs) / kurs
      - direction   : 1 = NAIK (kurs_next > kurs), 0 = TURUN/tetap
      - tanggal_next: tanggal hari trading berikutnya (untuk audit)

    Baris terakhir (tanpa hari berikutnya) dibuang.

    Args:
        rate_df: kolom ['tanggal', 'kurs'], sudah terurut menaik.

    Returns:
        DataFrame dengan kolom target, tanpa baris terakhir.
    """
    df = rate_df.copy()
    df["tanggal"] = pd.to_datetime(df["tanggal"])
    df = df.sort_values("tanggal").reset_index(drop=True)
    df["kurs_next"] = df["kurs"].shift(-1)
    df["tanggal_next"] = df["tanggal"].shift(-1)
    df["ret_next"] = (df["kurs_next"] - df["kurs"]) / df["kurs"]
    df["direction"] = (df["kurs_next"] > df["kurs"]).astype(int)
    # Baris terakhir tidak punya masa depan -> buang agar tidak bocor/NaN
    df = df.dropna(subset=["kurs_next"]).reset_index(drop=True)
    return df


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    date_col: str = "tanggal",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Bagi DataFrame harian menjadi train/val/test secara kronologis.

    Args:
        df: data harian terurut waktu (harus sudah termasuk target).
        train_ratio: proporsi train.
        val_ratio: proporsi validation (sisanya untuk test).
        date_col: nama kolom tanggal.

    Returns:
        (train, val, test, info) dengan info berisi batas tanggal & jumlah baris.
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = df.iloc[:n_train].reset_index(drop=True)
    val = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test = df.iloc[n_train + n_val:].reset_index(drop=True)
    info = {
        "n_total": n,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": round(1 - train_ratio - val_ratio, 4),
        "train_start": str(train[date_col].min()),
        "train_end": str(train[date_col].max()),
        "val_start": str(val[date_col].min()) if len(val) else None,
        "val_end": str(val[date_col].max()) if len(val) else None,
        "test_start": str(test[date_col].min()) if len(test) else None,
        "test_end": str(test[date_col].max()) if len(test) else None,
    }
    return train, val, test, info


def save_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                info: dict, out_dir: str) -> None:
    """Simpan train/val/test ke CSV + info split ke JSON."""
    os.makedirs(out_dir, exist_ok=True)
    train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(out_dir, "val.csv"), index=False)
    test.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    with open(os.path.join(out_dir, "split_info.json"), "w") as f:
        json.dump(info, f, indent=2, default=str)
