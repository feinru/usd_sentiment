import pandas as pd
import os

def adjust_dates(root_dir):
    processed_dir = os.path.join(root_dir, 'data', 'processed')
    output_dir = os.path.join(root_dir, 'data', 'dateadjusted')
    
    # Buat direktori output jika belum ada
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {processed_dir}...")
    
    # Memuat data rate (kurs) untuk mendapatkan hari-hari pasar (valid market days)
    # Ini akan secara otomatis mengecualikan weekend dan hari libur nasional
    rate_file = os.path.join(processed_dir, 'rate.csv')
    rate_df = pd.read_csv(rate_file)
    rate_df['tanggal'] = pd.to_datetime(rate_df['tanggal']).dt.date
    valid_dates = sorted(rate_df['tanggal'].unique())
    
    # Memuat data berita
    news_file = os.path.join(processed_dir, 'cnbc.csv')
    news_df = pd.read_csv(news_file)
    
    print("Parsing dates and applying logical rules...")
    # Parsing kolom 'date' ke datetime (pastikan menjadi tz-aware UTC)
    news_df['datetime'] = pd.to_datetime(news_df['date'], utc=True)
    
    # Konversi ke waktu lokal (Asia/Jakarta)
    news_df['datetime_jkt'] = news_df['datetime'].dt.tz_convert('Asia/Jakarta')
    
    # Aturan Logis: 
    # Jika berita diterbitkan setelah penutupan pasar (misal >= 15:00 WIB),
    # maka berita tersebut akan diselaraskan dengan hari berikutnya.
    news_df['adj_date'] = news_df['datetime_jkt'].dt.date
    mask_after_close = news_df['datetime_jkt'].dt.hour >= 15
    news_df.loc[mask_after_close, 'adj_date'] = news_df.loc[mask_after_close, 'adj_date'] + pd.Timedelta(days=1)
    
    # Aturan Logis 2:
    # Jika berita diterbitkan pada hari libur atau weekend, geser ke hari kerja pasar terdekat.
    # Kita menggunakan data tanggal pada rate.csv sebagai acuan "hari kerja pasar".
    def get_next_valid_date(d):
        for vd in valid_dates:
            if vd >= d:
                return vd
        # Jika tanggal melebihi data rate yang tersedia, geser ke hari kerja biasa (Senin-Jumat)
        while d.weekday() >= 5: # 5=Sabtu, 6=Minggu
            d += pd.Timedelta(days=1)
        return d
    
    news_df['date'] = news_df['adj_date'].apply(get_next_valid_date)
    
    # Hapus kolom sementara agar dataframe tetap rapi
    news_df = news_df.drop(columns=['datetime', 'datetime_jkt', 'adj_date'])
    
    # Simpan hasil berita yang tanggalnya sudah disesuaikan
    output_news_file = os.path.join(output_dir, 'cnbc.csv')
    news_df.to_csv(output_news_file, index=False)
    print(f"Saved adjusted news data to {output_news_file}")
    
    # Simpan juga data rate.csv ke dateadjusted untuk konsistensi folder (opsional namun direkomendasikan)
    output_rate_file = os.path.join(output_dir, 'rate.csv')
    rate_df.to_csv(output_rate_file, index=False)
    print(f"Saved rate data to {output_rate_file}")
    
    print("Selesai menyesuaikan tanggal!")

if __name__ == "__main__":
    # Output data setelah tanggal disesuaikan
    # Struktur: root/src/news/adjustdate.py 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_directory = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
    
    # Directory yang dituju
    expected_root = r"d:\CS24\NLP\usd_sentiment"
    if not os.path.exists(os.path.join(root_directory, 'data')):
        root_directory = expected_root
        
    adjust_dates(root_directory)
