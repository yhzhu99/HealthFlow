# file: filter_paper/filter.py

import os
import sys
import argparse
import json
import logging
import time
import pandas as pd
import threading
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Use the standard synchronous OpenAI library
from openai import OpenAI, OpenAIError, RateLimitError, APIConnectionError

# --- Load Environment Variables ---
try:
    project_root = Path(__file__).resolve().parent.parent
    dotenv_path = project_root / '.env'
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
    else:
        print(f"Warning: .env file not found at {dotenv_path}. Relying on system environment variables.")
except Exception as e:
    print(f"Error loading .env file: {e}")

# --- Static Configuration ---
LLM_MODELS_SETTINGS = {
    "deepseek-v3-official": {
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": "https://api.deepseek.com",
        "model_name": "deepseek-chat",
    },
    "deepseek-r1-official": {
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": "https://api.deepseek.com",
        "model_name": "deepseek-reasoner",
    },
    "qwen3-235b-a22b": {
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_name": "qwen3-235b-a22b",
    },
}

API_TIMEOUT_SECONDS = 120
MAX_RETRIES = 3

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
)
logging.getLogger("httpx").setLevel(logging.WARNING)


def generate_prompt(titles_batch: List[str]) -> str:
    """Generates the prompt for the LLM with a list of titles."""
    numbered_titles = "\n".join([f"{i+1}. {title}" for i, title in enumerate(titles_batch)])
    prompt = f"""
You are an AI research assistant specializing in scientific literature. Your task is to identify research papers focused on applying AI to Electronic Health Records (EHR).

**Classification Criteria:**
A paper is **relevant (1)** if its title indicates the use of AI, machine learning, or data science techniques on data explicitly from **Electronic Health Records (EHR)**. This includes:
- Clinical notes (free-text)
- Structured data (diagnosis codes like ICD, procedure codes)
- Time-series data from EHR (lab results, vital signs)

A paper is **NOT relevant (0)** if the title suggests a focus on:
- Medical Imaging (e.g., MRI, CT scans, X-rays, pathology slides)
- Genomics, proteomics, or any '-omics' data.
- Public health policy, hospital administration, or bioinformatics without direct patient-level EHR data analysis.
- Drug discovery or molecular modeling.
- Signal processing of physiological signals like ECG, EEG, unless contextually tied to an EHR system analysis.
- The term "administrative claims" or "billing data" alone, as this often lacks clinical depth.

**Your Task:**
Review the following list of paper titles.

---
{numbered_titles}
---

**Output Format:**
Respond with a single, valid JSON object. This object must have one key: "selected_indices". The value must be an array of integers representing the 1-based index of the titles you identified as relevant.

**Example:**
If you determine that papers 2, 5, and 19 are relevant, your response MUST be exactly:
```json
{{"selected_indices": [2, 5, 19]}}
```
Do not provide any explanations, apologies, or any text outside of the JSON object.
"""
    return prompt.strip()


def parse_llm_response(response_text: str, batch_size: int) -> Optional[List[int]]:
    """Parses the JSON response from the LLM, ensuring validity."""
    try:
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()

        data = json.loads(response_text)

        if not isinstance(data, dict) or "selected_indices" not in data:
            logging.warning(f"Invalid JSON structure: 'selected_indices' key missing. Response: {response_text[:100]}")
            return None

        indices = data["selected_indices"]
        if not isinstance(indices, list) or not all(isinstance(i, int) for i in indices):
            logging.warning(f"Invalid data type: 'selected_indices' is not a list of integers. Response: {response_text[:100]}")
            return None

        valid_indices = [i for i in indices if 1 <= i <= batch_size]
        if len(valid_indices) != len(indices):
            logging.warning(f"Response contains out-of-range indices. Original: {indices}, Filtered: {valid_indices}")

        return valid_indices
    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON from response: {response_text[:200]}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error parsing response: {e} | Response: {response_text[:200]}")
        return None


class LLMFilter:
    """A class to handle API interactions for a specific LLM using synchronous requests."""
    def __init__(self, model_key: str):
        if model_key not in LLM_MODELS_SETTINGS:
            raise ValueError(f"Model key '{model_key}' not found in LLM_MODELS_SETTINGS.")

        settings = LLM_MODELS_SETTINGS[model_key]
        if not settings.get("api_key"):
            raise ValueError(f"API key for '{model_key}' is not set. Check your .env or environment variables.")

        self.model_name = settings["model_name"]
        self.client = OpenAI(
            api_key=settings["api_key"],
            base_url=settings["base_url"],
            timeout=API_TIMEOUT_SECONDS,
            max_retries=0 # We handle retries manually
        )
        logging.info(f"Initialized LLMFilter for model: {self.model_name}")

    def call_api(self, titles_batch: List[str]) -> Optional[List[int]]:
        """Calls the LLM API for a single batch of titles with retry logic."""
        prompt = generate_prompt(titles_batch)
        messages = [{"role": "system", "content": "You are a helpful assistant that only responds in the specified JSON format."}, {"role": "user", "content": prompt}]

        for attempt in range(MAX_RETRIES):
            try:
                # Use streaming for compatibility with all models, including Qwen
                stream = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.0,
                    stream=True,
                )
                response_chunks = [chunk.choices[0].delta.content for chunk in stream if chunk.choices[0].delta.content]

                full_response = "".join(response_chunks)
                if full_response:
                    return parse_llm_response(full_response, len(titles_batch))
                else:
                    logging.warning("API returned an empty response.")
                    return None

            except (RateLimitError, APIConnectionError) as e:
                wait_time = 2 ** (attempt + 2) # Exponential backoff (4s, 8s, 16s)
                logging.warning(f"API connection/rate limit error (Attempt {attempt+1}/{MAX_RETRIES}): {e}. Retrying in {wait_time}s...")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(wait_time)
                else:
                    logging.error(f"API error after {MAX_RETRIES} retries. Giving up on this batch.")
            except OpenAIError as e:
                logging.error(f"A non-retriable OpenAI API error occurred (Attempt {attempt+1}/{MAX_RETRIES}): {e}")
                break # Don't retry on auth errors, etc.
            except Exception as e:
                logging.error(f"An unexpected error occurred during API call (Attempt {attempt+1}/{MAX_RETRIES}): {e}")

        return None # Return None if all retries fail


def run_llm_filter_on_df(df: pd.DataFrame, llm_name: str, batch_size: int, concurrency: int, output_path: Path):
    """
    Processes a DataFrame for a single LLM using a thread pool for concurrency.
    Updates and saves the DataFrame in a thread-safe manner.
    """
    col_name = f"{llm_name}_select"
    if col_name not in df.columns:
        df[col_name] = pd.NA

    df_to_process = df[df[col_name].isnull()]

    if df_to_process.empty:
        logging.info(f"All titles already processed for {llm_name}. Skipping.")
        return

    logging.info(f"Found {len(df_to_process)} unprocessed titles to filter using {llm_name}.")

    try:
        llm_filter = LLMFilter(model_key=llm_name)
    except ValueError as e:
        logging.error(f"Could not initialize LLMFilter for '{llm_name}': {e}. Skipping this LLM.")
        return

    batches = [{
        "titles": batch_df["Title"].tolist(),
        "original_indices": batch_df.index.tolist()
    } for _, batch_df in df_to_process.groupby(df_to_process.index // batch_size)]

    # A lock is crucial to prevent race conditions when updating the DataFrame and writing the CSV
    file_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        # Map futures to the original batch info to process results correctly
        future_to_batch = {executor.submit(llm_filter.call_api, batch['titles']): batch for batch in batches}

        # Use tqdm to show progress as futures complete
        for future in tqdm(as_completed(future_to_batch), total=len(batches), desc=f"Processing batches for {llm_name}", unit="batch"):
            batch_info = future_to_batch[future]
            try:
                selected_indices = future.result()
            except Exception as e:
                logging.error(f"A batch failed with an exception: {e}")
                selected_indices = None # Treat as a failed batch

            # Thread-safe update and save
            with file_lock:
                # If API call was successful and returned valid data
                if selected_indices is not None:
                    # Convert 1-based indices from LLM to a set of 0-based batch indices
                    selected_batch_indices_0_based = {idx - 1 for idx in selected_indices}

                    for i, original_df_index in enumerate(batch_info["original_indices"]):
                        # Mark as 1 if selected, otherwise 0
                        df.loc[original_df_index, col_name] = 1 if i in selected_batch_indices_0_based else 0
                else:
                    # Mark failed batches with NA to be re-processed in a future run.
                    # This ensures we don't incorrectly mark them as 0.
                    logging.warning(f"Batch starting with title '{batch_info['titles'][0][:50]}...' failed. Marking for retry.")
                    for original_df_index in batch_info["original_indices"]:
                        df.loc[original_df_index, col_name] = pd.NA

                # --- Real-time saving ---
                # Save the file after each batch completes. The lock ensures this is safe.
                df.to_csv(output_path, index=False)

    # Final conversion to a nullable integer type
    with file_lock:
        df[col_name] = df[col_name].astype('Int64')
        df.to_csv(output_path, index=False)

    logging.info(f"Finished processing for {llm_name}. Results updated in {output_path}.")


def main():
    """Main function to parse arguments and run the filtering process."""
    parser = argparse.ArgumentParser(description="Filter research papers related to EHR using LLMs.")
    parser.add_argument("--venues", nargs='+', required=True, help="List of venue names (e.g., 'aaai' 'iclr').")
    parser.add_argument("--years", nargs='+', required=True, help="List of years (e.g., '2020' '2021').")
    parser.add_argument("--llms", nargs='+', required=True, help=f"List of LLM names. Available: {list(LLM_MODELS_SETTINGS.keys())}")
    parser.add_argument("--batch_size", type=int, default=20, help="Number of titles per API call.")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of concurrent API calls.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    for venue in args.venues:
        for year in args.years:
            logging.info(f"\n{'='*60}\nProcessing: Venue={venue}, Year={year}\n{'='*60}")

            input_path = project_root / "title_extract" / "results" / venue / f"{year}.csv"
            output_dir = project_root / "filter_paper" / "results" / venue
            output_path = output_dir / f"{year}.csv"

            output_dir.mkdir(parents=True, exist_ok=True)

            if not input_path.is_file():
                logging.error(f"Input file not found: {input_path}. Skipping.")
                continue

            # Load existing output file or create a new DataFrame from the input
            if output_path.is_file():
                logging.info(f"Resuming from existing output file: {output_path}")
                df = pd.read_csv(output_path, keep_default_na=False, na_values=[''])
            else:
                logging.info(f"Starting from new input file: {input_path}")
                df = pd.read_csv(input_path)
                if 'ID' not in df.columns:
                    df.insert(0, 'ID', range(len(df)))

            if 'Title' in df.columns:
                 df['Title'] = df['Title'].astype(str).fillna('') # Ensure titles are strings

            for llm_name in args.llms:
                if llm_name not in LLM_MODELS_SETTINGS:
                    logging.warning(f"LLM '{llm_name}' not found in settings. Skipping.")
                    continue

                logging.info(f"--- Starting processing for LLM: {llm_name} ---")
                run_llm_filter_on_df(df, llm_name, args.batch_size, args.concurrency, output_path)
                # The function now blocks until all threads for the current LLM are done.
                # A final save happens inside the function, but another one here is harmless.
                logging.info(f"--- Completed processing for LLM: {llm_name} ---")

    logging.info("\nAll specified filtering jobs are complete.")

if __name__ == "__main__":
    main()