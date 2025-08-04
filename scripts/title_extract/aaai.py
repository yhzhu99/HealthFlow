import requests
from bs4 import BeautifulSoup
import csv
import time
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Create output directory
os.makedirs("aaai", exist_ok=True)

# Session + Retry settings
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retries)
session.mount("http://", adapter)
session.mount("https://", adapter)

headers = {
    "User-Agent": "Mozilla/5.0"
}

# Year mapping
year_map = {
    34: 2020,
    35: 2021,
    36: 2022,
    37: 2023,
    38: 2024,
    39: 2025
}

base_url_template = "https://aaai.org/proceeding/aaai-{}-{}/"

# Extract technical paper links (with track name)
def get_technical_track_links(year_num, url):
    print(f"🧭 Visiting: {url}")
    try:
        res = session.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        links = []
        for a in soup.find_all('a', href=True):
            text = a.text.strip()
            href = a['href']
            if 'Technical Track' in text:
                if href.startswith("/"):
                    href = "https://ojs.aaai.org" + href
                links.append((text, href))  # (Track Name, URL)
            elif 'technical track' in text.lower() and href.startswith("https://aaai.org/proceeding"):
                links.append((text, href))
        return links
    except Exception as e:
        print(f"❌ Failed to get technical track links for AAAI-{year_num}: {e}")
        return []

# Determine if it's a legacy structure link
def href_is_legacy(href):
    return href.startswith("https://aaai.org/proceeding/vol") or href.startswith("https://aaai.org/proceeding/0")

# Extract papers from new OJS format pages
def extract_papers_ojs(issue_url, track_name):
    print(f" [OJS] Extracting from: {issue_url}")
    papers = []
    try:
        res = session.get(issue_url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        for article in soup.select('ul.cmp_article_list > li'):
            title_tag = article.select_one('h3.title a')
            pdf_tag = article.select_one('a.obj_galley_link.pdf')
            if title_tag and pdf_tag:
                title = title_tag.text.strip()
                pdf_link = pdf_tag['href']
                if not pdf_link.startswith("http"):
                    pdf_link = "https://ojs.aaai.org" + pdf_link
                papers.append((title, pdf_link, track_name))
    except Exception as e:
        print(f" [OJS] Failed to extract: {e}")
    return papers

# Extract papers from legacy structure pages
def extract_papers_legacy(issue_url, track_name):
    print(f" [Legacy] Extracting from: {issue_url}")
    papers = []
    try:
        res = session.get(issue_url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')

        for li in soup.select('li.paper-wrap'):
            title_tag = li.find('h5')
            pdf_tag = li.find('a', class_='wp-block-button', href=True)

            if title_tag and title_tag.a:
                title = title_tag.a.get_text(strip=True)
                pdf_link = pdf_tag['href'] if pdf_tag else 'N/A'
                papers.append((title, pdf_link, track_name))
    except Exception as e:
        print(f" [Legacy] Failed to extract: {e}")
    return papers


# Main process
for num in range(34, 40):  # AAAI-34 到 AAAI-39
    year = year_map[num]
    url = base_url_template.format(num, year)
    track_links = get_technical_track_links(num, url)
    all_papers = []

    for track_name, track_url in track_links:
        if "ojs.aaai.org" in track_url:
            papers = extract_papers_ojs(track_url, track_name)
        else:
            papers = extract_papers_legacy(track_url, track_name)
        all_papers.extend(papers)
        time.sleep(2)  # Avoid IP blocking

    # Save CSV for current year
    csv_filename = f"aaai/AAAI-{num}-{year}.csv"
    with open(csv_filename, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Title", "PDF_Link", "Track"])
        writer.writerows(all_papers)

    print(f" Saved {len(all_papers)} papers to {csv_filename}")
