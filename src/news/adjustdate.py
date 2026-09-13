import pandas as pd
import os

def adjust_dates(root_dir):
    processed_dir = os.path.join(root_dir, 'data', 'processed')
    output_dir = os.path.join(root_dir, 'data', 'dateadjusted')
    
    # Siapkan direktori output
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {processed_dir}...")
    
    # Memuat data rate kurs (hari pasar aktif tanpa libur/weekend)
    rate_file = os.path.join(processed_dir, 'rate.csv')
    rate_df = pd.read_csv(rate_file)
    rate_df['tanggal'] = pd.to_datetime(rate_df['tanggal']).dt.date
    valid_dates = sorted(rate_df['tanggal'].unique())
    
    # Memuat data berita
    news_file = os.path.join(processed_dir, 'cnbc.csv')
    news_df = pd.read_csv(news_file)
    
    print("Parsing dates and applying logical rules...")
    # Parsing 'date' ke UTC tz-aware
    news_df['datetime'] = pd.to_datetime(news_df['date'], utc=True)
    
    # Konversi ke zona waktu lokal
    news_df['datetime_jkt'] = news_df['datetime'].dt.tz_convert('Asia/Jakarta')
    
    # Aturan 1: Berita rilis >= 15:00 WIB diselaraskan ke hari berikutnya
    news_df['adj_date'] = news_df['datetime_jkt'].dt.date
    mask_after_close = news_df['datetime_jkt'].dt.hour >= 15
    news_df.loc[mask_after_close, 'adj_date'] = news_df.loc[mask_after_close, 'adj_date'] + pd.Timedelta(days=1)
    
    # Aturan 2: Geser hari libur/weekend ke hari pasar terdekat (acuan rate.csv)
    import bisect
    def get_next_valid_date(d):
        idx = bisect.bisect_left(valid_dates, d)
        if idx < len(valid_dates):
            return valid_dates[idx]
        # Jika melebihi rentang rate.csv, geser ke hari kerja (Senin-Jumat)
        while d.weekday() >= 5: # 5=Sabtu, 6=Minggu
            d += pd.Timedelta(days=1)
        return d
    
    news_df['adjusted_date'] = news_df['adj_date'].apply(get_next_valid_date)
    
    # Hapus kolom sementara
    news_df = news_df.drop(columns=['datetime', 'datetime_jkt', 'adj_date'])
    
    # Simpan berita hasil penyesuaian
    output_news_file = os.path.join(output_dir, 'cnbc.csv')
    news_df.to_csv(output_news_file, index=False)
    print(f"Saved adjusted news data to {output_news_file}")
    
    # Salin rate.csv ke output untuk konsistensi
    output_rate_file = os.path.join(output_dir, 'rate.csv')
    rate_df.to_csv(output_rate_file, index=False)
    print(f"Saved rate data to {output_rate_file}")
    
    print("Penyelarasan data temporal selesai")

if __name__ == "__main__":
    # Resolusi jalur dinamis dari file saat ini
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_directory = os.path.abspath(os.path.join(current_dir, '..', '..'))
    
    if not os.path.exists(os.path.join(root_directory, 'data')):
        print("Warning: Direktori 'data' tidak ditemukan di root:", root_directory)
        
    adjust_dates(root_directory)
