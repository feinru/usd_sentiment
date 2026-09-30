# Data Splits — Klasifikasi Arah USD/IDR (NAIK/TURUN)

> Hasil `python -m src.features.build_dataset` dari `data/dateadjusted/`
> (kurs JISDOR + berita CNBC yang sudah selaras temporal).

## File

| File | Isi |
|---|---|
| `train.csv` | 826 hari trading, 2021-09-29 → 2025-02-25 |
| `val.csv` | 177 hari trading, 2025-02-26 → 2025-11-25 |
| `test.csv` | 178 hari trading, 2025-11-26 → 2026-08-31 |
| `split_info.json` | Batas tanggal, rasio, proporsi kelas, jumlah berita |
| `feature_list.json` | Daftar kolom per kelompok fitur (`tech`, `lex_title`, `lex_full`, `tfidf_title`, `tfidf_full`) |

## Target

`direction = 1` (NAIK) bila `kurs_next > kurs`, else `0` (TURUN).
`kurs_next` = kurs hari trading berikutnya. Baris terakhir tiap rentang
waktu dibuang karena tidak punya masa depan.

Proporsi kelas positif (NAIK): train **55.8%**, val **52.0%**, test **57.9%**.

## Kolom (190 total)

* Identitas & target: `tanggal`, `kurs`, `kurs_next`, `tanggal_next`,
  `ret_next`, `direction`.
* `tech_*` (19): `lag_1..5`, `ret_1/2/3/5`, `ma_5/10/20`, `std_5/20`,
  `dist_ma5`, `dist_ma20`, `dayofweek`, `month`.
* `t_*` (33): agregasi harian VADER + Loughran-McDonald dari **judul**
  (`mean/std/min/max` + `t_news_count`).
* `f_*` (33): agregasi yang sama dari **isi penuh**.
* `t_tfidf_svd_*` / `f_tfidf_svd_*` (50+50): rata-rata harian vektor
  TF-IDF→SVD artikel. Vectorizer + SVD **hanya di-fit pada 1810 artikel
  train** (berita dengan `adjusted_date` di rentang train).

Hari trading tanpa berita → fitur sentimen netral/nol, `news_count = 0`.

## Regenerasi

```bash
python -m src.features.build_dataset
python -m src.features.build_dataset --train-ratio 0.7 --val-ratio 0.15
```
