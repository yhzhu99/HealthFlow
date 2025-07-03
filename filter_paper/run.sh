#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# Set the venue and year you want to process.
VENUE="aaai"
YEAR="2020"

# --- Script Execution ---
echo "Starting paper filtering process..."
echo "Venue: ${VENUE}"
echo "Year:  ${YEAR}"
echo "------------------------------------"

# Navigate to the script's directory to ensure correct relative paths
# This makes the script runnable from anywhere in the project
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE}" )" &> /dev/null && pwd )"

# Run the Python script with the configured arguments
python3 "${SCRIPT_DIR}/filter.py" --venue "${VENUE}" --year "${YEAR}"

echo "------------------------------------"
echo "Script finished successfully."
echo "Check the results in: filter_paper/results/${VENUE}/${YEAR}.csv"
