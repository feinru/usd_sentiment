"""
Vectorization klasik: TF-IDF & Bag-of-Words + SVD (TANPA embeddings/transformer).

Keputusan desain (penting untuk anti-leakage):
  1. Vectorizer di-FIT HANYA pada artikel TRAIN (berdasarkan adjusted_date).
     Val/test hanya di-transform.
  2. SVD (TruncatedSVD) di-fit pada matriks TF-IDF train, lalu dipakai untuk
     memadatkan vektor artikel -> agregasi harian = rata-rata vektor SVD
     artikel pada hari itu. Tanpa SVD, 3000 kolom TF-IDF terlalu lebar untuk
     1200 baris harian.
  3. Hari tanpa berita -> vektor nol (eksplisit, bukan NaN).

API:
  fit_vectorizers(train_texts, method='tfidf', max_features=3000, svd_dim=50)
  daily_matrix(article_df, date_col, vectorizer, svd, prefix) -> DataFrame harian
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def fit_vectorizers(train_texts: list, method: str = "tfidf",
                    max_features: int = 3000, svd_dim: int = 50):
    """Fit vectorizer + SVD pada teks train saja.

    Args:
        train_texts: list string artikel train.
        method: 'tfidf' atau 'bow'.
        max_features: batas kosakata.
        svd_dim: dimensi output SVD (disesuaikan bila fitur < svd_dim).

    Returns:
        (vectorizer, svd) yang sudah di-fit.
    """
    train_texts = [t if isinstance(t, str) else "" for t in train_texts]
    if method == "bow":
        vec = CountVectorizer(max_features=max_features, ngram_range=(1, 2),
                              stop_words="english", min_df=5)
    else:
        vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2),
                              stop_words="english", min_df=5)
    X_train = vec.fit_transform(train_texts)
    k = max(1, min(svd_dim, X_train.shape[1] - 1))
    svd = TruncatedSVD(n_components=k, random_state=42)
    svd.fit(X_train)
    return vec, svd


def daily_matrix(article_df: pd.DataFrame, date_col: str,
                 text_col: str, vectorizer, svd, prefix: str) -> pd.DataFrame:
    """Transform artikel -> SVD -> rata-rata harian.

    Args:
        article_df: semua artikel (train+val+test), satu baris per artikel.
        date_col: kolom tanggal selaras ('adjusted_date').
        text_col: 'title' / 'full_text'.
        vectorizer, svd: hasil fit_vectorizers (fit di train saja!).
        prefix: mis. 't_tfidf_' / 'f_tfidf_'.

    Returns:
        DataFrame ['tanggal', f'{prefix}svd_0' ...] satu baris per tanggal.
    """
    texts = article_df[text_col].fillna("").astype(str).tolist()
    X = svd.transform(vectorizer.transform(texts))  # (n_artikel, k)
    k = X.shape[1]
    cols = [f"{prefix}svd_{i}" for i in range(k)]
    emb = pd.DataFrame(X, columns=cols)
    emb[date_col] = pd.to_datetime(article_df[date_col].values)
    daily = emb.groupby(date_col)[cols].mean().reset_index()
    daily = daily.rename(columns={date_col: "tanggal"})
    return daily
