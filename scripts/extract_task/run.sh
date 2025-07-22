#!/bin/bash

# A script to run EHR task extraction concurrently on a list of paper IDs.
#
# This script performs the following steps:
# 1. Reads paper IDs from a specified file.
# 2. For each ID, it first checks if the output file already exists.
# 3. If no output exists, it launches the `extract.py` script as a background job.
# 4. Manages the number of concurrent jobs to avoid overwhelming system resources or API rate limits.
# 5. The Python script itself handles locking to prevent multiple script instances
#    from processing the same paper ID simultaneously.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# The LLM to use. Must be a key from LLM_MODELS_SETTINGS in extract.py.
# Options: "deepseek-v3-official", "deepseek-r1-official", "qwen3-235b-a22b", "gemini-2.5-pro"
LLM_MODEL=gemini-2.5-pro

# Number of parallel jobs to run. Adjust based on your machine and API rate limits.
MAX_CONCURRENT=8 # Increased for demonstration; adjust as needed.

# Input file containing paper IDs, one per line.
ID_FILE="filter_paper/results/final_selected_ID.txt"

# Directory containing the source paper markdown folders.
MARKDOWNS_DIR="extract_task/assets/markdowns"

# Directory to save the output JSONL files.
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
    echo "Error: ID file not found at $ID_FILE" >&2
    exit 1
fi

# Read all paper IDs into an array to have a total count
mapfile -t ALL_IDS < <(grep -v -e '^$' -e '^#' "$ID_FILE")
TOTAL_IDS=${#ALL_IDS[@]}
PROCESSED_COUNT=0

for paper_id in "${ALL_IDS[@]}"; do
    PROCESSED_COUNT=$((PROCESSED_COUNT + 1))
    echo ">>> [${PROCESSED_COUNT}/${TOTAL_IDS}] Checking paper ID: $paper_id"

    # --- Pre-flight Check: Skip if output already exists ---
    # This is an efficient check to avoid launching Python unnecessarily.
    # The Python script has its own locking for true concurrent safety.
    OUTPUT_FILE="${OUTPUT_DIR}/${paper_id}_tasks.jsonl"
    if [ -f "$OUTPUT_FILE" ]; then
        echo "    Output for $paper_id already exists. Skipping."
        continue
    fi

    # Wait for a slot to become available if we've reached the max number of concurrent jobs.
    # `jobs -p` lists the PIDs of all background jobs. `wc -l` counts them.
    while (( $(jobs -p | wc -l) >= MAX_CONCURRENT )); do
        # `wait -n` pauses the script until any single background job finishes.
        # This is more efficient than polling with `sleep`.
        wait -n || true
    done

    echo "    Starting job for paper ID: $paper_id"

    # Run the python script in the background.
    # Logging is redirected to a file for better organization.
    python extract_task/extract.py \
        --paper_id "$paper_id" \
        --markdowns-dir "$MARKDOWNS_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --llm "$LLM_MODEL" &

done

# Wait for all remaining background jobs to complete.
echo ">>> All