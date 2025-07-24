"""
Data probe tool for examining data structures and files.
"""
import logging
import os
import json
from typing import Any, Dict, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class DataProbeTool:
    """Tool for probing and examining data structures."""

    def __init__(self):
        self.description = "Probe and examine data structures, files, and datasets to understand their format and contents"
        self.parameters = {
            "target": "File path or data structure to examine",
            "operation": "Type of probe: 'structure', 'head', 'info', 'describe', 'columns'"
        }

    def execute(self, probe_input: str) -> str:
        """Execute data probing operation."""
        try:
            # Parse input
            if isinstance(probe_input, str):
                if probe_input.strip().startswith('{'):
                    # Try to parse as JSON
                    try:
                        parsed = json.loads(probe_input)
                        target = parsed.get('target', probe_input)
                        operation = parsed.get('operation', 'structure')
                    except:
                        target = probe_input
                        operation = 'structure'
                else:
                    target = probe_input
                    operation = 'structure'
            else:
                target = str(probe_input)
                operation = 'structure'

            # Check if target is a file path
            if os.path.exists(target):
                return self._probe_file(target, operation)
            else:
                return f"File or path '{target}' does not exist."

        except Exception as e:
            logger.error(f"Data probe error: {e}")
            return f"Error probing data: {str(e)}"

    def _probe_file(self, filepath: str, operation: str) -> str:
        """Probe a file and return information about its structure."""
        try:
            file_ext = os.path.splitext(filepath)[1].lower()
            file_size = os.path.getsize(filepath)

            result = [f"File: {filepath}"]
            result.append(f"Size: {file_size:,} bytes")
            result.append(f"Extension: {file_ext}")

            if file_ext in ['.csv', '.tsv']:
                return self._probe_csv(filepath, operation, result)
            elif file_ext in ['.json', '.jsonl']:
                return self._probe_json(filepath, operation, result)
            elif file_ext in ['.xlsx', '.xls']:
                return self._probe_excel(filepath, operation, result)
            elif file_ext in ['.txt', '.log']:
                return self._probe_text(filepath, operation, result)
            else:
                result.append(f"Unsupported file type: {file_ext}")
                return '\n'.join(result)

        except Exception as e:
            return f"Error probing file '{filepath}': {str(e)}"

    def _probe_csv(self, filepath: str, operation: str, result: List[str]) -> str:
        """Probe CSV file."""
        try:
            # Read a sample to understand structure
            df = pd.read_csv(filepath, nrows=5)

            result.append(f"Format: CSV")
            result.append(f"Columns: {len(df.columns)}")
            result.append(f"Column names: {list(df.columns)}")

            if operation in ['info', 'structure']:
                result.append(f"Data types:")
                for col, dtype in df.dtypes.items():
                    result.append(f"  {col}: {dtype}")

            if operation == 'head':
                result.append("First 5 rows:")
                result.append(df.to_string())

            if operation == 'describe':
                # Get full dataset for description
                full_df = pd.read_csv(filepath)
                result.append(f"Total rows: {len(full_df)}")
                result.append("Statistical summary:")
                result.append(full_df.describe().to_string())

            return '\n'.join(result)

        except Exception as e:
            result.append(f"Error reading CSV: {str(e)}")
            return '\n'.join(result)

    def _probe_json(self, filepath: str, operation: str, result: List[str]) -> str:
        """Probe JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                if filepath.endswith('.jsonl'):
                    # JSONL format
                    lines = f.readlines()
                    result.append(f"Format: JSONL")
                    result.append(f"Lines: {len(lines)}")

                    if lines:
                        first_obj = json.loads(lines[0])
                        result.append(f"First object keys: {list(first_obj.keys())}")

                        if operation == 'head':
                            result.append("First 3 objects:")
                            for i, line in enumerate(lines[:3]):
                                result.append(f"  {i+1}: {json.loads(line)}")
                else:
                    # Regular JSON
                    data = json.load(f)
                    result.append(f"Format: JSON")
                    result.append(f"Type: {type(data).__name__}")

                    if isinstance(data, dict):
                        result.append(f"Keys: {list(data.keys())}")
                    elif isinstance(data, list):
                        result.append(f"Items: {len(data)}")
                        if data:
                            result.append(f"First item type: {type(data[0]).__name__}")
                            if isinstance(data[0], dict):
                                result.append(f"First item keys: {list(data[0].keys())}")

                    if operation == 'head':
                        result.append("Content preview:")
                        if isinstance(data, list):
                            result.append(str(data[:3]))
                        else:
                            result.append(str(data))

            return '\n'.join(result)

        except Exception as e:
            result.append(f"Error reading JSON: {str(e)}")
            return '\n'.join(result)

    def _probe_excel(self, filepath: str, operation: str, result: List[str]) -> str:
        """Probe Excel file."""
        try:
            # Get sheet names
            excel_file = pd.ExcelFile(filepath)
            sheet_names = excel_file.sheet_names

            result.append(f"Format: Excel")
            result.append(f"Sheets: {len(sheet_names)}")
            result.append(f"Sheet names: {sheet_names}")

            # Read first sheet
            df = pd.read_excel(filepath, sheet_name=sheet_names[0], nrows=5)
            result.append(f"First sheet '{sheet_names[0]}' columns: {list(df.columns)}")

            if operation == 'head':
                result.append("First 5 rows of first sheet:")
                result.append(df.to_string())

            return '\n'.join(result)

        except Exception as e:
            result.append(f"Error reading Excel: {str(e)}")
            return '\n'.join(result)

    def _probe_text(self, filepath: str, operation: str, result: List[str]) -> str:
        """Probe text file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            result.append(f"Format: Text")
            result.append(f"Lines: {len(lines)}")
            result.append(f"Characters: {sum(len(line) for line in lines)}")

            if operation == 'head':
                result.append("First 10 lines:")
                for i, line in enumerate(lines[:10]):
                    result.append(f"  {i+1}: {line.strip()}")

            return '\n'.join(result)

        except Exception as e:
            result.append(f"Error reading text file: {str(e)}")
            return '\n'.join(result)

    async def aexecute(self, probe_input: str) -> str:
        """Async version of execute."""
        return self.execute(probe_input)
