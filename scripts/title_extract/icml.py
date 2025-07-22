import os
import csv
import xml.sax
from collections import defaultdict

class ICMLHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.in_inproceedings = False
        self.current_key = ""
        self.current_data = ""
        self.title = ""
        self.year = ""
        self.booktitle = ""
        self.records_by_year = defaultdict(list)

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
            if self.booktitle == "ICML" and self.year.isdigit():
                y = int(self.year)
                if 2020 <= y <= 2025:
                    self.records_by_year[y].append({
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

if __name__ == "__main__":
    xml_path = "dblp.xml"
    output_dir = "icml"
    os.makedirs(output_dir, exist_ok=True)

    handler = ICMLHandler()
    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    parser.setContentHandler(handler)
    parser.parse(xml_path)

    total = 0
    for year, records in handler.records_by_year.items():
        file_path = os.path.join(output_dir, f"icml_{year}.csv")
        with open(file_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["key", "title", "year"])
            writer.writeheader()
            writer.writerows(records)
        print(f"Saved {len(records)} records to {file_path}")
        total += len(records)

    print(f"Total records found: {total}")
