# file: filter_paper/filter.py

import os
import sys
import argparse
import json
import logging
import asyncio
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
# Use the official OpenAI library, which is compatible with many other APIs
from openai import AsyncOpenAI, OpenAIError, RateLimitError, APIConnectionError

# --- Load Environment Variables ---
# Assumes .env file is in the project root, one level above this script's directory
try:
    project_root = Path(__file__).parent.parent
    dotenv_path = project_root / '.env'
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
    else:
        print(f"Warning: .env file not found at {dotenv_path}. Relying on environment variables.")
except Exception as e:
    print(f"Error loading .env file: {e}")


# --- Static Configuration ---
# LLM settings are now more flexible, just needing api_key, base_url, and model_name
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
    # Add other models here in the future, e.g., OpenAI, Anthropic, etc.
    # "openai-gpt4": {
    #     "api_key": os.getenv("OPENAI_API_KEY"),
    #     "base_url": "https://api.openai.com/v1",
    #     "model_name": "gpt-4-turbo-preview",
    # },
}

API_TIMEOUT_SECONDS = 120
MAX_RETRIES = 3

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
)
# Suppress noisy logs from the HTTP library
logging.getLogger("httpx").setLevel(logging.WARNING)


def generate_prompt(titles_batch: List[str]) -> str:
    """Generates the prompt for the LLM with a list of titles."""
    numbered_titles = "\n".join([f"{i+1}. {title}" for i, title in enumerate(titles_batch)])
    prompt = f"""
You are an AI research assistant. Your task is to identify research papers focused on "AI for Healthcare" from the provided list.

Review the following paper titles, which are numbered starting from 1.

**Scope Definition:**
A paper is "AI for Healthcare" if it applies AI/ML concepts to medical or healthcare problems.
- **AI/ML Concepts Include:** Artificial Intelligence, Machine Learning, Deep Learning, Large Language Models (LLMs), Neural Networks, Natural Language Processing (NLP), Computer Vision, Predictive Modeling, etc.
- **Healthcare Applications Include:** Diagnostics, drug discovery, medical imaging analysis, electronic health records (EHR), genomics, patient treatment, personalized medicine, clinical trial optimization, etc.

**Paper Titles:**
---
{numbered_titles}
---

**Instructions:**
Respond with a single JSON object. The object must contain one key, "selected_indices", whose value is a JSON array of integers. These integers must correspond to the numbers of the titles that fall within the "AI for Healthcare" scope.

Example: If titles 1, 3, and 8 are relevant, your response must be exactly:
{{"selected_indices": [1, 3, 8]}}

Do not include any other text, explanations, or markdown formatting. Your response MUST be a valid JSON object.
"""
    return prompt.strip()

def parse_llm_response(response_text: str) -> Optional[List[int]]:
    """Parses the JSON response from the LLM, handling common formatting issues."""
    try:
        # Handle cases where the JSON is wrapped in markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```").strip()

        data = json.loads(response_text)

        if isinstance(data, dict) and "selected_indices" in data:
            indices = data["selected_indices"]
            if isinstance(indices, list) and all(isinstance(i, int) for i in indices):
                return indices

        logging.warning(f"Invalid JSON structure in response: {response_text}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON from response: {response_text}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during parsing: {e} | Response: {response_text}")
        return None


class LLMFilter:
    """A class to handle API interactions for a specific LLM."""
    def __init__(self, model_key: str):
        if model_key not in LLM_MODELS_SETTINGS:
            raise ValueError(f"Model key '{model_key}' not found in LLM_MODELS_SETTINGS.")

        settings = LLM_MODELS_SETTINGS[model_key]
        if not settings.get("api_key"):
            raise ValueError(f"API key for '{model_key}' is not set. Check your .env file or environment variables.")

        self.model_name = settings["model_name"]
        self.client = AsyncOpenAI(
            api_key=settings["api_key"],
            base_url=settings["base_url"],
            timeout=API_TIMEOUT_SECONDS,
            max_retries=0  # We handle retries manually for better control
        )
        logging.info(f"Initialized LLMFilter for model: {self.model_name}")

    async def call_api(self, titles_batch: List[str]) -> Optional[List[int]]:
        """Calls the LLM API for a single batch of titles with retry logic."""
        prompt = generate_prompt(titles_batch)
        messages = [{"role": "user", "content": prompt}]

        for attempt in range(MAX_RETRIES):
            try:
                completion = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )

                if completion.choices and completion.choices[0].message:
                    response_content = completion.choices[0].message.content
                    if response_content:
                        return parse_llm_response(response_content)
                    else:
                        logging.warning("API returned an empty response content.")
                        return None # No content, no need to retry
                else:
                    logging.warning("API response is valid but contains no choices or message.")
                    return None

            except (RateLimitError, APIConnectionError) as e:
                wait_time = 2 ** (attempt + 1)
                logging.warning(f"API connection/rate limit error on attempt {attempt + 1}/{MAX_RETRIES}: {e}. Retrying in {wait_time}s...")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(wait_time) # Exponential backoff
                else:
                    logging.error(f"API error after {MAX_RETRIES} retries. Giving up on this batch.")
            except OpenAIError as e:
                logging.error(f"OpenAI API error on attempt {attempt + 1}/{MAX_RETRIES}: {e}")
                break # Don't retry on other OpenAI errors (e.g., auth)
            except Exception as e:
                logging.error(f"An unexpected error in API call on attempt {attempt + 1}/{MAX_RETRIES}: {e}")

        return None # Return None if all retries fail

async def run_concurrent_batches(
    batches: List[Dict[str, Any]],
    llm_filter: LLMFilter,
    concurrency: int,
    pbar: tqdm
) -> List[Dict[str, Any]]:
    """Manages concurrent processing of all batches for a given LLM."""
    semaphore = asyncio.Semaphore(concurrency)
    tasks = []

    async def task_wrapper(batch_info):
        async with semaphore:
            selected_indices = await llm_filter.call_api(batch_info['titles'])
            pbar.update(1)
            # Return the full result object for processing
            return {
                "original_indices": batch_info['original_indices'],
                "selected_indices_1_based": selected_indices
            }

    for batch_data in batches:
        tasks.append(task_wrapper(batch_data))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out and log exceptions
    final_results = []
    for res in results:
        if isinstance(res, Exception):
            logging.error(f"A concurrent task failed with an exception: {res}")
        else:
            final_results.append(res)

    return final_results


def process_file_for_llm(df: pd.DataFrame, llm_name: str, batch_size: int, concurrency: int) -> pd.DataFrame:
    """
    Processes a DataFrame for a single LLM, handling checkpointing, batching, and concurrency.
    Returns the updated DataFrame.
    """
    col_name = f"{llm_name}_select"
    if col_name not in df.columns:
        df[col_name] = pd.NA # Use pandas NA for nullable integer type

    # --- Checkpointing Logic ---
    # Filter for rows that have not been processed yet for this specific LLM
    df_to_process = df[df[col_name].isnull()]

    if df_to_process.empty:
        logging.info(f"All titles already processed for {llm_name}. Skipping.")
        return df

    logging.info(f"Found {len(df_to_process)} unprocessed titles to filter for {llm_name}.")

    try:
        llm_filter = LLMFilter(model_key=llm_name)
    except ValueError as e:
        logging.error(f"Could not initialize LLMFilter for '{llm_name}': {e}. Skipping this LLM.")
        return df

    # Create batches from the dataframe that needs processing
    batches = [{
        "titles": batch_df["Title"].tolist(),
        "original_indices": batch_df.index.tolist() # Store original df index for mapping back
    } for i in range(0, len(df_to_process), batch_size) if not (batch_df := df_to_process.iloc[i:i + batch_size]).empty]

    with tqdm(total=len(batches), desc=f"Processing batches for {llm_filter.model_name}", unit="batch") as pbar:
        # Run all batches concurrently
        batch_results = asyncio.run(run_concurrent_batches(batches, llm_filter, concurrency, pbar))

    # Update the main DataFrame with the results
    updates_made = 0
    for result in batch_results:
        # Check if the API call was successful and returned valid data
        if result and result["selected_indices_1_based"] is not None:
            # Convert 1-based indices from LLM to a set of 0-based indices for efficient lookup
            selected_indices_0_based = {idx - 1 for idx in result["selected_indices_1_based"]}

            for i, original_df_index in enumerate(result["original_indices"]):
                # Mark as 1 if selected (index is in the set), 0 if not selected
                df.loc[original_df_index, col_name] = 1 if i in selected_indices_0_based else 0
                updates_made += 1
        elif result:
            logging.warning(f"Failed to get results for a batch. Original indices start at {result['original_indices']}. These will remain as NA.")

    logging.info(f"Updated {updates_made} rows for {llm_name}.")

    # Convert the column to a nullable integer type for consistency
    df[col_name] = df[col_name].astype('Int64')
    return df

def main():
    """Main function to parse arguments and run the filtering process."""
    parser = argparse.ArgumentParser(description="Filter research papers using LLMs.")
    parser.add_argument("--venues", nargs='+', required=True, help="Space-separated list of venue names (e.g., 'aaai' 'iclr').")
    parser.add_argument("--years", nargs='+', required=True, help="Space-separated list of years (e.g., '2020' '2021').")
    parser.add_argument("--llms", nargs='+', required=True, help=f"Space-separated list of LLM names to run. Available: {list(LLM_MODELS_SETTINGS.keys())}")
    parser.add_argument("--batch_size", type=int, default=20, help="Number of titles per API call.")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of concurrent API calls.")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    for venue in args.venues:
        for year in args.years:
            logging.info(f"\n{'='*60}\nProcessing: Venue={venue}, Year={year}\n{'='*60}")

            input_path = project_root / "title_extract" / "results" / venue / f"{year}.csv"
            output_dir = project_root / "filter_paper" / "results" / venue
            output_path = output_dir / f"{year}.csv"

            output_dir.mkdir(parents=True, exist_ok=True)

            if not input_path.is_file():
                logging.error(f"Input file not found: {input_path}. Skipping this combination.")
                continue

            # --- Resuming/Loading Logic ---
            # Load existing output file or create a new DataFrame from the input
            if output_path.is_file():
                logging.info(f"Resuming from existing output file: {output_path}")
                # keep_default_na=False and na_values=[''] prevent "NA" string from being read as NaN
                df = pd.read_csv(output_path, keep_default_na=False, na_values=[''])
            else:
                logging.info(f"Starting from input file: {input_path}")
                df = pd.read_csv(input_path)
                if 'ID' not in df.columns:
                    # Insert ID column at the beginning
                    df.insert(0, 'ID', range(len(df)))

            # Process with each specified LLM
            for llm_name in args.llms:
                logging.info(f"--- Starting processing for LLM: {llm_name} ---")
                df = process_file_for_llm(df, llm_name, args.batch_size, args.concurrency)

                # --- Save progress after each LLM is done for a given file ---
                logging.info(f"Saving intermediate results to {output_path}")
                df.to_csv(output_path, index=False)
                logging.info(f"--- Finished processing for LLM: {llm_name} ---")

            logging.info(f"Processing for {venue} {year} complete. Final results saved to {output_path}")

    logging.info("\nAll processing jobs are complete.")

if __name__ == "__main__":
    main()