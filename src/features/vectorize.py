import pandas as pd
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def fit_tfidf_svd(train_texts: List[str], max_features: int = 3000, svd_dim: int = 50) -> Tuple[TfidfVectorizer, TruncatedSVD]:
    train_texts = [t if isinstance(t, str) else "" for t in train_texts]
    
    vectorizer = TfidfVectorizer(
        max_features=max_features, 
        ngram_range=(1, 2),
        stop_words="english", 
        min_df=5
    )
    
    X_train = vectorizer.fit_transform(train_texts)
    
    n_components = max(1, min(svd_dim, X_train.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(X_train)
    
    return vectorizer, svd

def transform_daily_tfidf(
    article_df: pd.DataFrame, 
    date_col: str,
    text_col: str, 
    vectorizer: TfidfVectorizer, 
    svd: TruncatedSVD, 
    prefix: str = ""
) -> pd.DataFrame:
    texts = article_df[text_col].fillna("").astype(str).tolist()
    
    # Transform
    X_tfidf = vectorizer.transform(texts)
    X_svd = svd.transform(X_tfidf)
    
    # Create DataFrame of embeddings
    n_components = X_svd.shape[1]
    cols = [f"{prefix}svd_{i}" for i in range(n_components)]
    emb_df = pd.DataFrame(X_svd, columns=cols)
    
    # Add date column and group
    emb_df[date_col] = pd.to_datetime(article_df[date_col].values)
    daily_df = emb_df.groupby(date_col)[cols].mean().reset_index()
    daily_df = daily_df.rename(columns={date_col: "tanggal"})
    
    return daily_df
