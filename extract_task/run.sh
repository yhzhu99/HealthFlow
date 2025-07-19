#!/bin/bash

# A script to run EHR task extraction concurrently on a list of paper IDs.
#
# This script performs the following steps:
# 1. (Assumes Python environment is active)
# 2. Reads paper IDs from a specified file.
# 3. For each ID, it launches the `extract.py` script as a background job.
# 4. Manages the number of concurrent jobs using modern bash features (`jobs` and `wait -n`)
#    to avoid overwhelming system resources or API rate limits.
# 5. Waits for all jobs to complete.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# The LLM to use. This must be a key from the LLM_MODELS_SETTINGS dictionary in extract.py.
# Available options typically include: "deepseek-v3-official", "deepseek-r1-official", "qwen3-235b-a22b"
LLM_MODEL="deepseek-r1-official"
# Number of parallel jobs to run. Adjust based on your machine and API rate limits.
MAX_CONCURRENT=1
# Input file containing paper IDs, one per line.
ID_FILE="filter_paper/results/final_selected_ID.txt"
# Directory containing the source paper markdown folders.
MARKDOWNS_DIR="extract_task/assets/markdowns"
# Directory to save the output JSON files.
OUTPUT_DIR="extract_task/tasks"

# --- Ensure Output Directory Exists ---
mkdir -p "$OUTPUT_DIR"
echo ">>> Output directory '$OUTPUT_DIR' is ready."

# --- Trap for cleanup ---
# A simple trap ensures a final message is always printed on exit.
trap "echo; echo '>>> Script finished.'" EXIT

# --- Main Execution Loop ---
echo ">>> Starting task extraction for paper IDs in $ID_FILE using LLM: $LLM_MODEL..."
if [ ! -f "$ID_FILE" ]; then
    # Send error messages to stderr
    echo "Error: ID file not found at $ID_FILE" >&2
    exit 1
fi

while IFS= read -r paper_id || [[ -n "$paper_id" ]]; do
    # Skip empty lines or lines that start with a #
    if [[ -z "$paper_id" || "$paper_id" =~ ^# ]]; then
        continue
    fi

    # Wait for a slot to become available if we've reached the max number of concurrent jobs.
    # `jobs -p` lists the PIDs of all background jobs. `wc -l` counts them.
    while (( $(jobs -p | wc -l) >= MAX_CONCURRENT )); do
        # `wait -n` pauses the script until any single background job finishes.
        # This is more efficient and reliable than polling with `sleep`.
        # `|| true` prevents the script from exiting if `wait` fails (e.g., no jobs left).
        wait -n || true
    done

    echo "Starting job for paper ID: $paper_id"

    # Run the python script in the background.
    # This is safer and cleaner than building a command string and using `eval`.
    python extract_task/extract.py \
        --paper_id "$paper_id" \
        --markdowns-dir "$MARKDOWNS_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --llm "$LLM_MODEL" &

done < "$ID_FILE"

# Wait for all remaining background jobs to complete.
echo ">>> All jobs launched. Waiting for the remaining jobs to finish..."
wait

echo ">>> All extraction tasks have been processed successfully."