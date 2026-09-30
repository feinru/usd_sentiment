import os
import pandas as pd
from src.features.lexicon import score_dataframe, aggregate_daily
from src.features.vectorize import fit_tfidf_svd, transform_daily_tfidf

def run_extraction():
    data_path = "data/dateadjusted/cnbc.csv"
    output_dir = "data/features"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    df = df.dropna(subset=['title', 'full_text']).reset_index(drop=True)
    
    print("Scoring 'title' with VADER and LM...")
    df = score_dataframe(df, text_col='title', prefix='title_')
    
    print("Scoring 'full_text' with VADER and LM...")
    df = score_dataframe(df, text_col='full_text', prefix='text_')
    
    score_cols = [c for c in df.columns if c.startswith('title_vader') or c.startswith('title_lm') or c.startswith('text_vader') or c.startswith('text_lm')]
    
    print(f"Aggregating {len(score_cols)} lexicon features to daily level...")
    daily_lexicon = aggregate_daily(df, date_col='adjusted_date', prefix='', score_cols=score_cols)
    
    print("Fitting TF-IDF and SVD for 'title'...")
    vec_title, svd_title = fit_tfidf_svd(df['title'].tolist(), max_features=1000, svd_dim=25)
    daily_tfidf_title = transform_daily_tfidf(df, 'adjusted_date', 'title', vec_title, svd_title, prefix='title_')
    
    print("Fitting TF-IDF and SVD for 'full_text'...")
    vec_text, svd_text = fit_tfidf_svd(df['full_text'].tolist(), max_features=3000, svd_dim=50)
    daily_tfidf_text = transform_daily_tfidf(df, 'adjusted_date', 'full_text', vec_text, svd_text, prefix='text_')
    
    print("Merging daily features...")
    final_df = daily_lexicon.merge(daily_tfidf_title, on='tanggal', how='left')
    final_df = final_df.merge(daily_tfidf_text, on='tanggal', how='left')
    
    out_path = os.path.join(output_dir, "data.csv")
    final_df.to_csv(out_path, index=False)
    print(f"Success! Saved {final_df.shape[0]} daily rows and {final_df.shape[1]} features to {out_path}.")

if __name__ == "__main__":
    run_extraction()
