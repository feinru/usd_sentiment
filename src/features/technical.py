"""
Fitur teknikal dari histori kurs JISDOR (hanya kolom 'kurs' tersedia).

Fitur yang dibuat (semua memakai data <= hari T, tanpa look-ahead):
  - lag_1..lag_5       : kurs T-1 .. T-5
  - ret_1/2/3/5        : return (kurs_T - kurs_T-k) / kurs_T-k
  - ma_5/10/20         : rata-rata gerak (rolling mean, hanya masa lalu)
  - std_5/20           : volatilitas rolling (std return/kurs)
  - dist_ma5 / dist_ma20 : jarak relatif kurs ke MA ((kurs-MA)/MA)
  - dayofweek, month   : kalender (Jumat/Senin sering berbeda perilaku FX)

Baris awal (20 hari pertama) dibuang karena MA-20 belum valid.
"""

import pandas as pd


def add_technical_features(rate_df: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan fitur teknikal ke dataframe kurs harian.

    Args:
        rate_df: kolom ['tanggal', 'kurs'] (+ kolom target bila sudah ada).

    Returns:
        DataFrame dengan kolom fitur baru; 20 baris pertama dibuang.
    """
    df = rate_df.copy()
    df["tanggal"] = pd.to_datetime(df["tanggal"])
    df = df.sort_values("tanggal").reset_index(drop=True)

    for k in (1, 2, 3, 4, 5):
        df[f"lag_{k}"] = df["kurs"].shift(k)
    for k in (1, 2, 3, 5):
        df[f"ret_{k}"] = (df["kurs"] - df["kurs"].shift(k)) / df["kurs"].shift(k)
    for w in (5, 10, 20):
        df[f"ma_{w}"] = df["kurs"].shift(1).rolling(w).mean()
    for w in (5, 20):
        df[f"std_{w}"] = df["kurs"].shift(1).rolling(w).std()
    df["dist_ma5"] = (df["kurs"] - df["ma_5"]) / df["ma_5"]
    df["dist_ma20"] = (df["kurs"] - df["ma_20"]) / df["ma_20"]
    df["dayofweek"] = df["tanggal"].dt.dayofweek  # 0=Senin..4=Jumat
    df["month"] = df["tanggal"].dt.month

    df = df.dropna().reset_index(drop=True)  # buang warm-up MA-20
    return df


TECH_COLS = [
    "kurs", "lag_1", "lag_2", "lag_3", "lag_4", "lag_5",
    "ret_1", "ret_2", "ret_3", "ret_5",
    "ma_5", "ma_10", "ma_20", "std_5", "std_20",
    "dist_ma5", "dist_ma20", "dayofweek", "month",
]
