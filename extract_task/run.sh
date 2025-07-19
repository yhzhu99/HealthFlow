#!/bin/bash

# A script to run EHR task extraction concurrently on a list of paper IDs.
#
# This script performs the following steps:
# 1. Sets up the Python virtual environment using `uv`.
# 2. Installs required packages from `extract_task/requirements.txt`.
# 3. Reads paper IDs from `filter_paper/results/final_selected_ID.txt`.
# 4. For each ID, it launches the `extract.py` script as a background job.
# 5. Manages the number of concurrent jobs to avoid overwhelming system resources or API rate limits.
# 6. Waits for all jobs to complete.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# Number of parallel jobs to run. Adjust based on your machine and API rate limits.
MAX_CONCURRENT=8
# Input file containing paper IDs, one per line.
ID_FILE="filter_paper/results/final_selected_ID.txt"
# Directory containing the source paper markdown folders.
MARKDOWNS_DIR="extract_task/assets/markdowns"
# Directory to save the output JSON files.
OUTPUT_DIR="extract_task/tasks"

# --- Ensure Output Directory Exists ---
mkdir -p "$OUTPUT_DIR"
echo ">>> Output directory '$OUTPUT_DIR' is ready."

# --- Concurrency Management ---
# Create a temporary file to track the PIDs of background jobs.
TEMP_FILE=$(mktemp)
# Ensure the temp file is removed on script exit, even if it errors.
trap "rm -f $TEMP_FILE; echo '>>> Script finished.'" EXIT

# Function to run a command in the background and manage concurrency.
run_command() {
    local cmd="$1"

    # Wait if the number of current jobs has reached the maximum.
    while [ "$(wc -l < "$TEMP_FILE")" -ge "$MAX_CONCURRENT" ]; do
        # Check for completed jobs and remove their PIDs from the tracking file.
        for pid in $(cat "$TEMP_FILE"); do
            if ! kill -0 "$pid" 2>/dev/null; then
                # If kill -0 fails, the process is gone. Remove it from the list.
                grep -v "^$pid$" "$TEMP_FILE" > "${TEMP_FILE}.new" && mv "${TEMP_FILE}.new" "$TEMP_FILE"
            fi
        done
        # Sleep for a short duration before checking again.
        sleep 1
    done

    # Run the command in the background.
    echo "Starting job: $cmd"
    eval "$cmd" &

    # Record the new job's PID.
    echo $! >> "$TEMP_FILE"
}

# --- Main Execution Loop ---
echo ">>> Starting task extraction for paper IDs in $ID_FILE..."
if [ ! -f "$ID_FILE" ]; then
    echo "Error: ID file not found at $ID_FILE"
    exit 1
fi

while IFS= read -r paper_id || [[ -n "$paper_id" ]]; do
    # Skip empty lines.
    if [ -z "$paper_id" ]; then
        continue
    fi

    # Define the command to be executed for the current paper ID.
    cmd="python extract_task/extract.py --paper_id \"$paper_id\" --markdowns-dir \"$MARKDOWNS_DIR\" --output-dir \"$OUTPUT_DIR\""

    # Run the command using the concurrency manager.
    run_command "$cmd"
done < "$ID_FILE"

# Wait for all remaining background jobs to complete.
echo ">>> All jobs launched. Waiting for remaining jobs to complete..."
wait

echo ">>> All extraction tasks have been processed successfully."