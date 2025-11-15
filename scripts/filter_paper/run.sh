#!/bin/bash
# file: filter_paper/run.sh

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# Define lists of venues and years to process.
# Use spaces to separate items within the parentheses.
# Example: VENUES=("aaai" "iclr" "icml")
VENUES=("aaai")
YEARS=("2020")

# Define which LLMs to run. These names must match the keys in the Python script's LLM_MODELS_SETTINGS.
# Example: LLMS_TO_RUN=("deepseek-chat-official" "qwen3-235b-a22b")
LLMS_TO_RUN=("deepseek-chat-official" "qwen3-235b-a22b")

# Define performance parameters.
BATCH_SIZE=20
CONCURRENCY=4
# --- End of Configuration ---


# Get the directory where the script is located to ensure correct relative paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PYTHON_SCRIPT_PATH="${SCRIPT_DIR}/filter.py"

# Check if the python script exists
if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
    echo "Error: Python script not found at ${PYTHON_SCRIPT_PATH}"
    exit 1
fi

echo "Starting batch paper filtering process..."
echo "===================================="
echo "Venues to process: ${VENUES[*]}"
echo "Years to process:  ${YEARS[*]}"
echo "LLMs to run:       ${LLMS_TO_RUN[*]}"
echo "Batch Size:        ${BATCH_SIZE}"
echo "Concurrency:       ${CONCURRENCY}"
echo "===================================="

# It's good practice to ensure your Python environment is active.
# For example, if you use a virtual environment:
# source /path/to/your/venv/bin/activate

# Run the Python script with the configured arguments.
# The "@" in "${VAR[@]}" is crucial for correctly handling arguments with spaces.
python3 "${PYTHON_SCRIPT_PATH}" \
  --venues "${VENUES[@]}" \
  --years "${YEARS[@]}" \
  --llms "${LLMS_TO_RUN[@]}" \
  --batch_size "${BATCH_SIZE}" \
  --concurrency "${CONCURRENCY}"

echo ""
echo "===================================="
echo "All processing jobs are complete."
echo "Check the results in the 'filter_paper/results/' directory."