"""
Pipeline filtering noise untuk dataset berita geopolitik hasil scraping.
Disesuaikan dengan karakteristik cnbc.csv (single-source, 11714 baris).
"""

import pandas as pd
import re

INPUT_PATH = "data/raw/news/cnbc.csv"
OUTPUT_PATH = "data/processed/cnbc.csv"

df = pd.read_csv(INPUT_PATH)
report = {"awal": len(df)}

# 1. Drop baris dengan full_text kosong (gagal scraping / paywall / video-only)
df = df.dropna(subset=["full_text"])
report["setelah_drop_missing_full_text"] = len(df)

# 2. Drop duplikat exact berdasarkan full_text (artikel ke-scrape >1x)
df = df.drop_duplicates(subset=["full_text"], keep="first")
report["setelah_drop_duplikat_exact"] = len(df)

# 3. Bersihkan boilerplate "In this article" (prefix widget ticker saham CNBC)
df["full_text"] = df["full_text"].apply(
    lambda t: re.sub(r"^In this article\s*", "", t).strip()
)

# 4. Drop artikel yang terlalu pendek untuk analisis konten (< 30 kata)
#    Threshold dipilih dari distribusi kata di dataset ini (median ~566, hampir semua > 100)
word_count = df["full_text"].str.split().str.len()
df = df[word_count >= 30]
report["setelah_drop_terlalu_pendek"] = len(df)

# 5. (Opsional, tidak dijalankan default) Near-duplicate detection via TF-IDF cosine similarity
#    Berguna kalau nanti kamu gabungkan sumber lain yang bisa republish artikel yang sama.
RUN_NEAR_DUP_CHECK = False
if RUN_NEAR_DUP_CHECK:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    vec = TfidfVectorizer(max_features=5000, stop_words="english")
    X = vec.fit_transform(df["full_text"])
    sim = cosine_similarity(X, dense_output=False)
    to_drop = set()
    coo = sim.tocoo()
    for i, j, v in zip(coo.row, coo.col, coo.data):
        if i < j and v > 0.9:
            to_drop.add(j)
    df = df.drop(df.index[list(to_drop)])
    report["setelah_near_dup_check"] = len(df)

# 6. Reset index & simpan
df = df.reset_index(drop=True)
df.to_csv(OUTPUT_PATH, index=False)

# 7. Hilangkan link
df["full_text"] = (
    df["full_text"]
    .str.replace(r"https?://\S+|www\.\S+", "", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

print("=== Laporan Filtering ===")
for k, v in report.items():
    print(f"{k}: {v}")
print(f"\nTotal baris akhir: {len(df)}")
print(f"Disimpan ke: {OUTPUT_PATH}")
