import requests
import os
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
import concurrent.futures

def fetch_full_text(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
        }
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            body = soup.find("div", class_="ArticleBody-articleBody")
            if body:
                pars = body.find_all("p")
                text = " ".join([p.get_text().strip() for p in pars])
                return text
            return "No full text available (Video/Live Blog)"
    except Exception as e:
        return f"Error fetching text"
    return ""

def scrape_cnbc(keywords=None):
    if keywords is None:
        # A broad basket of keywords that drive USD/IDR and global EM sentiment
        keywords = [
            "IDR", "Rupiah", "USD/IDR", "Bank Indonesia", 
            "Federal Reserve", "Inflation", "Emerging Markets", 
            "Treasury Yields", "OPEC", "Supply Chain", 
            "Geopolitics", "Interest Rates", "US Dollar", "Trade War"
        ]
        
    print("Scraping CNBC...")
    url = "https://api.queryly.com/cnbc/json.aspx"
    
    all_articles = []
    
    print("Harvesting URLs from Search API...")
    for keyword in keywords:
        endindex = 0
        print(f"  Querying '{keyword}'")
        
        while True:
            params = {
                "queryly_key": "31a35d40a9a64ab3",
                "query": keyword,
                "endindex": endindex,
                "batchsize": 50,
                "sort": "date"
            }
            
            try:
                res = requests.get(url, params=params).json()
                results = res.get("results", [])
                
                if not results:
                    break
                    
                for item in results:
                    date_str = item.get("datePublished", "")
                    cn_type = item.get("cn:type", "")
                    
                    if date_str and date_str < "2021-09-01":
                        continue
                        
                    # Filter out video clips since they don't have article bodies
                    if "video" in cn_type.lower() or "live" in cn_type.lower():
                        continue
                        
                    all_articles.append({
                        "source": "CNBC",
                        "title": item.get("cn:title", ""),
                        "url": item.get("url", ""),
                        "date": date_str,
                        # We will replace this later
                        "full_text": ""
                    })
                
                endindex += 50
                if endindex > 2000:
                    break
                    
            except Exception as e:
                print(f"Error querying CNBC: {e}")
                break
                
    df = pd.DataFrame(all_articles)
    
    if not df.empty:
        df = df[df["date"] >= "2021-09-01"]
        df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
        
        print(f"Scraping {len(df)} URLs for full text...")
        urls = df["url"].tolist()
        
        full_texts = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = executor.map(fetch_full_text, urls)
            
            for i, text in enumerate(results):
                full_texts.append(text)
                if (i + 1) % 100 == 0:
                    print(f"  -> Scraped {i + 1}/{len(urls)} articles")
                    
        df["full_text"] = full_texts
        
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    cnbc_dir = os.path.join(project_root, "data", "raw", "news")
    os.makedirs(cnbc_dir, exist_ok=True)
    
    out_path = os.path.join(cnbc_dir, "cnbc.csv")
    df.to_csv(out_path, index=False)
    print(f"CNBC saved to {out_path} ({len(df)} articles)")
    return df

if __name__ == "__main__":
    import sys
    # If arguments are provided, use them as keywords. Otherwise use the default inside the function.
    if len(sys.argv) > 1:
        custom_keywords = sys.argv[1:]
        scrape_cnbc(keywords=custom_keywords)
    else:
        scrape_cnbc()
