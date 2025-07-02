# Title Extraction Scripts

This folder contains conference- and journal-specific scripts to extract paper titles and metadata from various sources. The outputs are saved as structured CSV files for downstream analysis.

## Sources and Extraction Methods

- **KDD / WWW**: Extracted from ACM Digital Library via exported `.bib` files. Scripts parse the `.bib` and convert to CSV.
- **ICLR / ICML**: Extracted from the `dblp.xml` dataset via XML parsing using `xml.sax`. Only records from 2020–2025 are retained.
- **Nature Series / AAAI / NIPS / IJCAI / NEJM AI**: Extracted via web scraping from official publisher websites, tailored per journal/venue.


