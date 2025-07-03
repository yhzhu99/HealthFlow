#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# Define lists of venues and years to process.
# Use spaces to separate items within the parentheses.
VENUES=("aaai")
YEARS=("2020")

# Define which LLMs to run. These names must match the keys in the Python script's LLM_MODELS_SETTINGS.
LLMS_TO_RUN=("deepseek-v3-official")

# Define performance parameters.
BATCH_SIZE=20
CONCURRENCY=4
# --- End of Configuration ---


# Get the directory where the script is located to ensure correct relative paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE}" )" &> /dev/null && pwd )"

# Main processing loop
echo "Starting batch paper filtering process..."

for VENUE in "${VENUES[@]}"; do
  for YEAR in "${YEARS[@]}"; do
    echo ""
    echo "======================================================================"
    echo "Processing: VENUE=${VENUE}, YEAR=${YEAR}"
    echo "LLMs:       ${LLMS_TO_RUN[*]}"
    echo "Config:     Batch Size=${BATCH_SIZE}, Concurrency=${CONCURRENCY}"
    echo "======================================================================"

    # Run the Python script with the configured arguments
    # The "@" in "${LLMS_TO_RUN[@]}" ensures that LLM names with spaces are handled correctly
    python3 "${SCRIPT_DIR}/filter.py" \
      --venue "${VENUE}" \
      --year "${YEAR}" \
      --llms "${LLMS_TO_RUN[@]}" \
      --batch_size "${BATCH_SIZE}" \
      --concurrency "${CONCURRENCY}"

    echo "Finished processing for ${VENUE} ${YEAR}."
  done
done

echo ""
echo "------------------------------------"
echo "All processing jobs are complete."
echo "Check the results in the 'filter_paper/results/' directory."