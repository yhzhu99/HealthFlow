import os
import csv
import time
import requests
import urllib3
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

#  忽略 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

#  使用 Session + 自动重试
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503])
session.mount("https://", HTTPAdapter(max_retries=retries))

# 模拟浏览器 UA
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

# 期刊与年份设置
journals = [
    "nature", "nmeth", "ncomms", "nm", "natcomputsci", "nathumbehav",
    "npjdigitalmed", "natbiomedeng", "natmachintell", "sdata",
    "naturehealth", "nprot"
]
years = list(range(2020, 2026))

# 输出目录
output_root = "nature_articles_clean"
base_url = "https://www.nature.com"
url_template = "https://www.nature.com/{journal}/research-articles?year={year}&page={page}"

for journal in journals:
    for year in years:
        results = []
        page = 1
        while True:
            url = url_template.format(journal=journal, year=year, page=page)
            print(f" Scraping {journal} {year} page {page}...")

            try:
                response = session.get(url, headers=headers, timeout=10, verify=False)
                soup = BeautifulSoup(response.content, "html.parser")
            except Exception as e:
                print(f" Failed to load page: {e}")
                break

            articles = soup.select("article.c-card")
            if not articles:
                print(" No more articles found.")
                break

            for art in articles:
                try:
                    title_tag = art.select_one("h3.c-card__title a")
                    title = title_tag.get_text(strip=True)
                    link = base_url + title_tag.get("href")
                    date_tag = art.select_one("time")
                    date = date_tag.get_text(strip=True) if date_tag else "N/A"
                    results.append([date, title, link])
                except Exception:
                    continue

            page += 1
            time.sleep(1.2)

        # 保存为 CSV
        journal_dir = os.path.join(output_root, journal)
        os.makedirs(journal_dir, exist_ok=True)
        csv_path = os.path.join(journal_dir, f"{year}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Date", "Title", "URL"])
            writer.writerows(results)

        print(f" Saved {len(results)} articles to {csv_path}")
