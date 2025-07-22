import os
import csv
import requests
from bs4 import BeautifulSoup

def scrape_ijcai_papers(start_year=2020, end_year=2024, output_dir="ijcai"):
    base_url = "https://www.ijcai.org/proceedings/"
    os.makedirs(output_dir, exist_ok=True)

    for year in range(start_year, end_year + 1):
        print(f"Processing IJCAI {year}...")
        url = f"{base_url}{year}/"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Failed to retrieve data for {year}: {e}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        records = []
        current_section = ""

        for div in soup.find_all("div", class_=["section", "paper_wrapper"]):
            # Detect section title
            section_title_tag = div.find("div", class_="section_title")
            if section_title_tag:
                current_section = section_title_tag.get_text(strip=True)

            # Extract paper info
            if "paper_wrapper" in div.get("class", []):
                title_tag = div.find("div", class_="title")
                pdf_link_tag = div.find("a", href=lambda href: href and href.endswith(".pdf"))

                title = title_tag.get_text(strip=True) if title_tag else ""
                pdf_url = base_url + f"{year}/" + pdf_link_tag['href'] if pdf_link_tag else ""

                if title and pdf_url:
                    records.append([title, pdf_url, current_section])

        output_file = os.path.join(output_dir, f"ijcai_{year}.csv")
        with open(output_file, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Title", "URL", "Section"])
            writer.writerows(records)

        print(f"Saved {len(records)} records to {output_file}")

if __name__ == "__main__":
    scrape_ijcai_papers()
