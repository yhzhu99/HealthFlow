import os
import csv
import bibtexparser
from collections import defaultdict

input_dir = "www_origin"
output_dir = "www"
os.makedirs(output_dir, exist_ok=True)

bib_files = [f for f in os.listdir(input_dir) if f.endswith(".bib")]
records_by_year = defaultdict(list)

for bib_file in bib_files:
    path = os.path.join(input_dir, bib_file)
    with open(path, encoding="utf-8") as f:
        bib_database = bibtexparser.load(f)

    for entry in bib_database.entries:
        title = entry.get("title", "").strip()
        url = entry.get("url", "").strip()
        year = entry.get("year", "").strip()

        if not (title and url and year.isdigit()):
            continue

        records_by_year[year].append([title, url, year])

# Write CSV files (grouped by year)
for year, records in records_by_year.items():
    output_path = os.path.join(output_dir, f"www_{year}.csv")
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Title", "URL", "Year"])
        writer.writerows(records)
    print(f"Year {year}: {len(records)} records saved to {output_path}")
