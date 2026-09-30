"""
Orkestrator pembangunan dataset modeling (dijalankan sekali per eksperimen).

Alur (urutan penting agar tidak bocor):
  1. Muat rate.csv + cnbc.csv (dateadjusted) — sudah selaras temporal
     (berita >=15:00 WIB -> H+1, weekend/libur -> hari trading berikut).
  2. Buat target arah (kurs T+1 vs T) di LEVEL RATE, lalu fitur teknikal.
  3. Tentukan batas split kronologis 70/15/15 dari tanggal (belum join berita).
  4. Lexicon (VADER+LM): scoring per artikel untuk title & full_text,
     agregasi harian. (Scoring per artikel tidak butuh fit, aman.)
  5. TF-IDF: fit vectorizer+SVD HANYA pada artikel dengan adjusted_date
     di rentang TRAIN, lalu transform semua artikel -> agregasi harian.
  6. Join semua fitur ke index tanggal trading; hari tanpa berita = 0/netral.
  7. Split -> data/splits/{train,val,test}.csv + split_info.json + feature_list.json.

Contoh:
  python -m src.features.build_dataset
  python -m src.features.build_dataset --train-ratio 0.7 --val-ratio 0.15
"""

import argparse
import json
import os

import pandas as pd

from src.features.lexicon import SCORE_COLS, aggregate_daily, score_articles
from src.features.technical import TECH_COLS, add_technical_features
from src.features.vectorize import daily_matrix, fit_vectorizers
from src.utils.split import chronological_split, make_target, save_splits

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATEADJ = os.path.join(ROOT, "data", "dateadjusted")
SPLIT_DIR = os.path.join(ROOT, "data", "splits")


def load_inputs():
    """Muat rate + berita hasil penyelarasan temporal."""
    rate = pd.read_csv(os.path.join(DATEADJ, "rate.csv"))
    news = pd.read_csv(os.path.join(DATEADJ, "cnbc.csv"))
    news["adjusted_date"] = pd.to_datetime(news["adjusted_date"])
    return rate, news


def build_master(train_ratio=0.70, val_ratio=0.15,
                 tfidf_features=3000, svd_dim=50):
    """Bangun tabel harian master (belum di-split). Mengembalikan (master, info)."""
    rate, news = load_inputs()

    # 1-2. Target + fitur teknikal di level rate
    rated = make_target(rate)
    rated = add_technical_features(rated)

    # 3. Batas split kronologis (pakai tanggal master teknikal)
    _, _, _, split_info = chronological_split(
        rated[["tanggal"]], train_ratio=train_ratio, val_ratio=val_ratio)
    train_start, train_end = pd.to_datetime(split_info["train_start"]), pd.to_datetime(split_info["train_end"])

    # Buang berita di luar rentang rate (mis. Sep 2026 > kurs terakhir)
    valid_dates = set(pd.to_datetime(rated["tanggal"]).dt.date)
    news = news[news["adjusted_date"].dt.date.isin(valid_dates)].reset_index(drop=True)

    # 4. Lexicon harian untuk judul (t_) & isi (f_)
    lex_frames = [rated[["tanggal"]].copy()]
    for text_col, prefix in (("title", "t_"), ("full_text", "f_")):
        scored = score_articles(news, text_col=text_col, prefix=prefix)
        cols = [prefix + c for c in SCORE_COLS]
        daily = aggregate_daily(scored, date_col="adjusted_date",
                                prefix=prefix, score_cols=cols)
        lex_frames.append(daily)
    lex = lex_frames[0]
    for d in lex_frames[1:]:
        lex = lex.merge(d, on="tanggal", how="left")

    # 5. TF-IDF+SVD: fit HANYA di artikel train
    train_news = news[(news["adjusted_date"] >= train_start)
                      & (news["adjusted_date"] <= train_end)]
    print(f"Artikel train untuk fit TF-IDF: {len(train_news)} dari {len(news)}")
    vec_frames = []
    for text_col, prefix in (("title", "t_tfidf_"), ("full_text", "f_tfidf_")):
        vec, svd = fit_vectorizers(train_news[text_col].fillna("").tolist(),
                                   method="tfidf", max_features=tfidf_features,
                                   svd_dim=svd_dim)
        daily = daily_matrix(news, date_col="adjusted_date", text_col=text_col,
                             vectorizer=vec, svd=svd, prefix=prefix)
        vec_frames.append((prefix, daily, vec, svd))

    # 6. Join semuanya ke index tanggal trading
    master = rated.copy()
    master = master.merge(lex, on="tanggal", how="left")
    for prefix, daily, _, _ in vec_frames:
        master = master.merge(daily, on="tanggal", how="left")
    master = master.fillna(0)  # hari tanpa berita -> netral/nol

    # Kumpulkan daftar kolom fitur per kelompok (untuk ablation)
    lex_t = [c for c in master.columns if c.startswith("t_") and "tfidf" not in c]
    lex_f = [c for c in master.columns if c.startswith("f_") and "tfidf" not in c]
    tfidf_t = [c for c in master.columns if c.startswith("t_tfidf_")]
    tfidf_f = [c for c in master.columns if c.startswith("f_tfidf_")]
    feature_groups = {
        "tech": [c for c in TECH_COLS if c in master.columns],
        "lex_title": lex_t, "lex_full": lex_f,
        "tfidf_title": tfidf_t, "tfidf_full": tfidf_f,
    }
    return master, split_info, feature_groups, len(train_news), len(news)


def main():
    ap = argparse.ArgumentParser(description="Bangun dataset modeling USD/IDR.")
    ap.add_argument("--train-ratio", type=float, default=0.70)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--tfidf-features", type=int, default=3000)
    ap.add_argument("--svd-dim", type=int, default=50)
    args = ap.parse_args()

    master, split_info, groups, n_tr_news, n_news = build_master(
        args.train_ratio, args.val_ratio, args.tfidf_features, args.svd_dim)

    # 7. Split master & simpan (kolom tanggal sudah terurut)
    master = master.sort_values("tanggal").reset_index(drop=True)
    n = len(master)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    train = master.iloc[:n_train].reset_index(drop=True)
    val = master.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test = master.iloc[n_train + n_val:].reset_index(drop=True)
    split_info.update({
        "n_total": n, "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "train_start": str(train["tanggal"].min()), "train_end": str(train["tanggal"].max()),
        "val_start": str(val["tanggal"].min()), "val_end": str(val["tanggal"].max()),
        "test_start": str(test["tanggal"].min()), "test_end": str(test["tanggal"].max()),
        "n_news_total": int(n_news), "n_news_train_fit": int(n_tr_news),
        "target": "direction = 1 bila kurs_next > kurs (NAIK), else 0 (TURUN)",
        "target_pos_rate_train": float(train["direction"].mean()),
        "target_pos_rate_val": float(val["direction"].mean()),
        "target_pos_rate_test": float(test["direction"].mean()),
    })
    save_splits(train, val, test, split_info, SPLIT_DIR)
    with open(os.path.join(SPLIT_DIR, "feature_list.json"), "w") as f:
        json.dump({k: v for k, v in groups.items()}, f, indent=2)

    print(f"Master: {master.shape} | train {train.shape} val {val.shape} test {test.shape}")
    print(f"Kolom fitur: tech={len(groups['tech'])} lex_t={len(groups['lex_title'])} "
          f"lex_f={len(groups['lex_full'])} tfidf_t={len(groups['tfidf_title'])} "
          f"tfidf_f={len(groups['tfidf_full'])}")
    print(f"Tersimpan di {SPLIT_DIR}/")


if __name__ == "__main__":
    main()
