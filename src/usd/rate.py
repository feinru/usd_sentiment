import os
import pandas as pd
from playwright.sync_api import sync_playwright

def download_and_convert_jisdor():
    url = "https://www.bi.go.id/id/statistik/informasi-kurs/jisdor/default.aspx"
    raw_dir = "data/raw"
    os.makedirs(raw_dir, exist_ok=True)
    
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    with sync_playwright() as p:
        print("Launching browser...")
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
            accept_downloads=True
        )
        
        print("Navigating to BI JISDOR page...")
        page.goto(url, wait_until="networkidle", timeout=60000)
        
        print("Filling date range: 01/09/2021 to 01/09/2026")
        page.evaluate('document.getElementById("TextBoxFrom").value = "01/09/2021"')
        page.evaluate('document.getElementById("TextBoxDateTo").value = "01/09/2026"')
        
        print("Clicking search...")
        page.get_by_role("button", name="Cari").click()
        page.wait_for_load_state("networkidle", timeout=60000)
        
        print("Initiating download...")
        with page.expect_download(timeout=60000) as download_info:
            page.locator("input[value='Unduh']").first.click()
            
        download = download_info.value
        
        xlsx_path = os.path.join(raw_dir, "rate.xlsx")
        download.save_as(xlsx_path)
        print(f"Downloaded Excel file to {xlsx_path}")
        
        browser.close()
        
    print("Converting to CSV...")
    try:
        df = pd.read_excel(xlsx_path, skiprows=4)
        
        df = df[["Tanggal", "Kurs"]]
        df = df.rename(columns={"Tanggal": "tanggal", "Kurs": "kurs"})
        df["tanggal"] = pd.to_datetime(df["tanggal"], format="mixed").dt.strftime("%Y-%m-%d")
        df = df.sort_values("tanggal").reset_index(drop=True)
        
        csv_path = os.path.join(processed_dir, "rate.csv")
        df.to_csv(csv_path, index=False)
        print(f"Successfully converted and cleaned to {csv_path}")
    except Exception as e:
        print(f"Error converting to CSV: {e}")

if __name__ == "__main__":
    download_and_convert_jisdor()
