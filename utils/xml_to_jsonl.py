import xml.etree.ElementTree as ET
import json
import argparse
import sys
import os
import re
import html # Standard library module for escaping HTML/XML characters

def fix_malformed_xml_content(xml_string: str) -> str:
    """
    Pre-processes a string containing malformed XML.

    It finds the content within <category>, <task>, and <answer> tags and
    escapes special XML characters (&, <, >) within that content. This
    "fixes" the string so the standard XML parser can handle it.

    Args:
        xml_string (str): The raw string read from the malformed XML file.

    Returns:
        str: A "fixed" XML string ready for parsing.
    """
    # This regex finds a tag (<category>, <task>, or <answer>), captures the
    # content inside it, and then finds its corresponding closing tag.
    # It works across multiple lines due to re.DOTALL.
    # Breakdown of the regex:
    # - (                            # Start of group 1 (the full opening tag)
    #   <(category|task|answer)>     #   Literal '<', then one of the tag names (captured in group 2), then '>'
    #   )                            # End of group 1
    # - (.*?)                        # Group 3: The content. .*? is a non-greedy match for any characters.
    # - (</\2>)                      # Group 4: The closing tag. </ followed by a backreference to group 2 (the tag name)
    pattern = re.compile(r"(<(category|task|answer)>)(.*?)(</\2>)", re.DOTALL)

    def escape_content(match):
        """A replacer function for re.sub"""
        opening_tag = match.group(1)
        content = match.group(3)
        closing_tag = match.group(4)

        # Use html.escape to safely escape &, <, > and other characters
        # in the content part ONLY.
        escaped_content = html.escape(content)

        # Reconstruct the element with the now-safe content
        return opening_tag + escaped_content + closing_tag

    # Use re.sub with our replacer function to fix the string
    return pattern.sub(escape_content, xml_string)


def convert_xml_to_jsonl(xml_file_path: str, jsonl_file_path: str) -> bool:
    """
    Parses a potentially malformed XML file by first fixing its content
    and then converts each '<item>' element into a JSON line.

    Args:
        xml_file_path (str): The path to the input XML file.
        jsonl_file_path (str): The path for the output JSONL file.

    Returns:
        bool: True if conversion was successful, False otherwise.
    """
    try:
        # 1. Read the entire potentially malformed XML file into a string.
        print(f"Reading and fixing malformed XML file: {xml_file_path}...")
        with open(xml_file_path, 'r', encoding='utf-8') as f_in:
            raw_xml_string = f_in.read()

        # 2. Pre-process the string to escape illegal characters inside tags.
        #    This is the crucial step to avoid the ParseError.
        fixed_xml_string = fix_malformed_xml_content(raw_xml_string)

        # 3. Parse the corrected XML string from memory.
        #    Use ET.fromstring() because we are parsing a string, not a file.
        print("Parsing corrected XML data...")
        root = ET.fromstring(fixed_xml_string)

        item_count = 0

        # 4. Open the target file for writing.
        print(f"Writing to JSONL file: {jsonl_file_path}...")
        with open(jsonl_file_path, 'w', encoding='utf-8') as f_out:
            # 5. Iterate through each '<item>' tag in the XML.
            for item_element in root.findall('item'):
                data_dict = {}

                # 6. Extract text from sub-elements. This will now work correctly.
                category_elem = item_element.find('category')
                data_dict['category'] = category_elem.text.strip() if category_elem is not None and category_elem.text else ""

                task_elem = item_element.find('task')
                data_dict['task'] = task_elem.text.strip() if task_elem is not None and task_elem.text else ""

                answer_elem = item_element.find('answer')
                data_dict['answer'] = answer_elem.text.strip() if answer_elem is not None and answer_elem.text else ""

                # 7. Serialize the dictionary to a JSON string.
                json_line = json.dumps(data_dict, ensure_ascii=False)

                # 8. Write the JSON line followed by a newline character.
                f_out.write(json_line + '\n')
                item_count += 1

        print(f"\n✅ Success! Converted {item_count} items.")
        print(f"Output saved to: {os.path.abspath(jsonl_file_path)}")
        return True

    except FileNotFoundError:
        print(f"❌ Error: The input file was not found at '{xml_file_path}'", file=sys.stderr)
        return False
    except ET.ParseError as e:
        print(f"❌ Error: Failed to parse XML file. It may be malformed in a way that the script can't fix (e.g., mismatched tags).", file=sys.stderr)
        print(f"   Details: {e}", file=sys.stderr)
        return False
    except PermissionError:
        print(f"❌ Error: Permission denied. Check read permissions for '{xml_file_path}' "
              f"and write permissions for '{jsonl_file_path}'.", file=sys.stderr)
        return False
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}", file=sys.stderr)
        return False

def main():
    """
    Main function to parse command-line arguments and run the conversion.
    """
    parser = argparse.ArgumentParser(
        description="Convert a malformed XML file with <item> tags to a JSONL file using only standard libraries.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input_xml",
        help="Path to the source XML file."
    )
    parser.add_argument(
        "output_jsonl",
        help="Path for the destination JSONL file."
    )
    args = parser.parse_args()

    if not convert_xml_to_jsonl(args.input_xml, args.output_jsonl):
        sys.exit(1)

if __name__ == "__main__":
    main()