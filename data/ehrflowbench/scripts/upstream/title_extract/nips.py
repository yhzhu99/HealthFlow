import os
import csv
import requests
from bs4 import BeautifulSoup

def fetch_nips_papers(year, output_dir):
    if year <= 2021:
        li_class = "none"
    else:
        li_class = "conference"

    url = f"https://papers.nips.cc/paper_files/paper/{year}"
    try:
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        res.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Year {year}: Failed to fetch page - {e}")
        return

    soup = BeautifulSoup(res.text, "html.parser")
    paper_entries = soup.find_all("li", class_=li_class)
    papers = []

    for li in paper_entries:
        a_tag = li.find("a", title="paper title")
        if not a_tag: continue
        title = a_tag.text.strip().strip('"')
        href = a_tag["href"]
        full_url = "https://papers.nips.cc" + href
        papers.append((title, full_url))

    if not papers:
        print(f"Year {year}: No papers found or parsing failed.")
        return

    output_path = os.path.join(output_dir, f"nips_{year}.csv")
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Title", "URL"])
        writer.writerows(papers)

    print(f"Year {year}: {len(papers)} papers saved to {output_path}")

if __name__ == "__main__":
    output_folder = "nips"
    os.makedirs(output_folder, exist_ok=True)
    for y in range(2020, 2026):
        fetch_nips_papers(y, output_folder)
