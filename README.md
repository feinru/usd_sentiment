# USD/IDR Sentiment — Prediksi Arah Kurs dari Berita + Histori Kurs

Pipeline NLP + machine learning untuk memprediksi **arah harian USD/IDR
(NAIK/TURUN)** dari berita geopolitik/ekonomi CNBC dan histori kurs JISDOR
Bank Indonesia. Branch aktif: **`rayyan-test`**.

## 1. Ringkasan hasil baseline

| Model (fitur) | Val acc / F1-macro | Test acc / F1-macro |
|---|---|---|
| Naive mayoritas | 0.520 / 0.342 | 0.579 / 0.367 |
| Naive persistensi (arah kemarin) | 0.514 / 0.513 | 0.545 / 0.533 |
| LogReg, kurs saja | 0.531 / 0.494 | 0.522 / 0.520 |
| GradBoost, kurs saja | 0.497 / 0.495 | 0.483 / 0.458 |
| **GradBoost, kurs + sentimen judul** | 0.503 / 0.496 | **0.590 / 0.579** |
| **LogReg, kurs + sentimen judul** | 0.520 / 0.462 | **0.601 / 0.552** |

Kesimpulan jujur: arah FX harian hampir acak untuk model kurs-saja
(akurasi ≈ 0.50). Menambah sentimen **judul** memberi kenaikan moderat di
test (+6–8 pp akurasi), tetapi **tidak terkonfirmasi di validation**
(≈ 0.50) — mengindikasikan pergeseran rezim 2025–2026, bukan sinyal yang
sudah stabil. Korelasi harian skor VADER judul vs return besok ≈ +0.04
(sangat lemah). Langkah berikut yang disarankan: walk-forward validation,
kalibrasi ambang probabilitas, dan fitur event/keyword. Detail di
`notebook/02_baseline_experiments.ipynb`.

## 2. Arsitektur

```
data/dateadjusted/{rate,cnbc}.csv        <- input selaras temporal
        │   (berita >=15:00 WIB -> H+1; weekend/libur -> hari trading berikut,
        │    lihat src/news/adjustdate.py)
        ▼
src/features/build_dataset.py
        ├── target: direction = 1 bila kurs[T+1] > kurs[T]
        ├── teknikal: lag/return/MA/volatilitas/kalender (19 kolom)
        ├── lexicon: VADER + Loughran-McDonald per artikel -> agregasi
        │            harian, judul (t_*) & isi (f_*) terpisah (33+33 kolom)
        └── TF-IDF(3000, 1-2gram) -> SVD(50), FIT HANYA di artikel train
             -> rata-rata harian judul & isi (50+50 kolom)
        ▼
data/splits/{train,val,test}.csv        <- 826 / 177 / 178 hari (kronologis)
        ├── src/models/train_baseline.py  (kurs saja: naive, logreg, gradboost)
        └── src/models/train_combined.py  (ablation: +lex_title, +lex_full,
                                           +lex_all, +tfidf_only, +all)
        ▼
outputs/results/{baseline,combined}_metrics.{json,csv}
```

Tidak ada pre-trained embeddings / transformer di pipeline ini (sesuai
ketentuan): hanya lexicon + TF-IDF/BoW + model klasik.

## 3. Struktur repo

```
data/
  raw/            hasil scraping mentah (cnbc.csv, rate.xlsx)
  processed/      hasil filtering (cnbc.csv, rate.csv)
  dateadjusted/   hasil penyelarasan temporal (+ adjusted_date)
  splits/         train/val/test modeling + split_info.json + README.md
src/
  news/           scraping & cleaning (cnbc.py, filtering.py, adjustdate.py)
  usd/            unduhan kurs JISDOR (rate.py)
  features/       technical.py, lexicon.py, vectorize.py, build_dataset.py
  models/         train_baseline.py, train_combined.py, evaluate.py
  utils/          split.py (target+split kronologis), metrics.py
notebook/
  01_eda.ipynb                 EDA kurs, berita, sentimen
  02_baseline_experiments.ipynb hasil & analisis eksperimen
outputs/
  results/        metrik JSON/CSV (di-commit)
  models/         file .joblib (di-gitignore, regenerasi lokal)
```

## 4. Cara menjalankan

```bash
pip install -r requirements.txt

# 1. Bangun dataset modeling (sekali jalan, ±1-2 menit)
python -m src.features.build_dataset

# 2. Baseline kurs-saja
python -m src.models.train_baseline

# 3. Model gabungan (ablation NLP)
python -m src.models.train_combined

# 4. Perbandingan baseline vs gabungan
python -m src.models.evaluate --best-on val
```

## 5. Desain anti-leakage

1. Berita digeser ke hari trading yang benar **sebelum** dipakai
   (`adjustdate.py`): berita sore/malam memprediksi hari berikut, bukan
   hari yang sama.
2. Target memakai `kurs[T+1]`; fitur hari `T` hanya dari info ≤ `T`.
3. Split **kronologis** 70/15/15 tanpa shuffle.
4. `TfidfVectorizer` + `SVD` + `StandardScaler` di-fit **hanya di train**;
   val/test hanya di-transform.
5. `StandardScaler` dipakai di dalam `Pipeline` LogReg sehingga statistik
   scaling tidak bocor antar split.

## 6. Metrik

Klasifikasi arah: **accuracy (= directional accuracy)**, precision, recall,
F1, F1-macro, AUC, confusion matrix. Modul `src/utils/metrics.py` juga
menyediakan `regression_metrics` (RMSE/MAE) untuk ekstensi regresi
"prediksi kurs besok" tanpa mengubah API.

## 7. Keterbatasan & next step

* Sinyal judul > isi (judul padat, isi noisy) — konsisten di LogReg &
  GradBoost; perlu uji signifikansi (McNemar / Diebold-Mariano).
* Val vs test tidak konsisten → tambah walk-forward / purged CV.
* TF-IDF global (semua topik CNBC, termasuk kredit mahasiswa AS) → coba
  filter topik FX-relevan dulu, atau BoW keyword Soros/Fed/BI.
* Model .joblib tidak di-commit; jalankan training untuk meregenerasi.
