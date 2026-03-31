import xml.sax
import csv
import os
from collections import defaultdict

class ICLRHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.in_inproceedings = False
        self.current_key = ""
        self.current_data = ""
        self.title = ""
        self.year = ""
        self.booktitle = ""
        self.records = []

    def startElement(self, tag, attributes):
        self.current_data = tag
        if tag == "inproceedings":
            self.in_inproceedings = True
            self.current_key = attributes.get("key", "")
            self.title = ""
            self.year = ""
            self.booktitle = ""

    def endElement(self, tag):
        if tag == "inproceedings":
            if self.booktitle == "ICLR" and self.year.isdigit():
                y = int(self.year)
                if 2020 <= y <= 2025:
                    self.records.append({
                        "key": self.current_key,
                        "title": self.title.strip(),
                        "year": self.year
                    })
            self.in_inproceedings = False
        self.current_data = ""

    def characters(self, content):
        if not self.in_inproceedings:
            return
        if self.current_data == "title":
            self.title += content
        elif self.current_data == "year":
            self.year += content
        elif self.current_data == "booktitle":
            self.booktitle += content

# Main program entry point
if __name__ == "__main__":
    xml_path = "dblp.xml"  # Replace with your path
    handler = ICLRHandler()
    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    parser.setContentHandler(handler)
    parser.parse(xml_path)

    print(f"Found: {len(handler.records)} records")

    # Create target directory
    os.makedirs("iclr", exist_ok=True)

    # Group by year
    year_dict = defaultdict(list)
    for rec in handler.records:
        year_dict[rec["year"]].append(rec)

    # Write CSV files for each year
    for year, items in year_dict.items():
        file_path = os.path.join("iclr", f"iclr_{year}.csv")
        with open(file_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["key", "title", "year"])
            writer.writeheader()
            writer.writerows(items)
        print(f"Saved: {file_path}")
