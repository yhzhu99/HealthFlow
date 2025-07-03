# filter_paper/filter.py

import os
import sys
import argparse
import json
import logging
import asyncio
import pandas as pd
import aiohttp
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

# --- Load Environment Variables ---
# Assumes .env file is in the project root, two levels up from this script.
dotenv_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

# --- Static Configuration ---
LLM_MODELS_SETTINGS = {
    "deepseek-v3-official": {
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": "https://api.deepseek.com/chat/completions",
        "model_name": "deepseek-chat",
    },
    "deepseek-r1-official": {
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": "https://api.deepseek.com/chat/completions",
        "model_name": "deepseek-reasoner",
    },
    # You can add more LLMs here following the same structure
}

API_TIMEOUT_SECONDS = 120
MAX_RETRIES = 3

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
)
# Suppress noisy aiohttp connection logging
logging.getLogger("aiohttp.internal").setLevel(logging.WARNING)


# --- Prompt Generation (as provided by user) ---
def generate_prompt(titles_batch: List[str]) -> str:
    """
    Generates a high-quality, robust prompt for classifying titles.
    """
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

Do not include any other text, explanations, or markdown formatting.
"""
    return prompt.strip()

def parse_llm_response(response_text: str) -> Optional[List[int]]:
    """
    Safely parses the LLM's JSON response to extract selected indices.
    """
    try:
        # The response might be wrapped in markdown ```json ... ```
        if "```json" in response_text:
            # Extract content between the first ```json and the final ```
            response_text = response_text.split("```json")[1].split("```")

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

async def call_llm_api_async(
    session: aiohttp.ClientSession,
    titles_batch: List[str],
    llm_config: Dict[str, Any],
    pbar: tqdm
) -> Optional[List[int]]:
    """
    Asynchronously calls the LLM API for a single batch of titles.
    """
    prompt = generate_prompt(titles_batch)
    headers = {
        "Authorization": f"Bearer {llm_config['api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": llm_config['model_name'],
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
    }
    timeout = aiohttp.ClientTimeout(total=API_TIMEOUT_SECONDS)

    for attempt in range(MAX_RETRIES):
        try:
            async with session.post(
                llm_config['base_url'],
                headers=headers,
                json=payload,
                timeout=timeout
            ) as response:
                if response.status == 200:
                    response_json = await response.json()
                    # BUG FIX: 'choices' is a list. Access the first element.
                    if response_json and response_json.get('choices'):
                        message_content = response_json['choices']['message']['content']
                        pbar.update(1)
                        return parse_llm_response(message_content)
                    else:
                        logging.error(f"API response OK but no 'choices' field. Response: {response_json}")
                        break # Don't retry if response structure is wrong
                else:
                    error_text = await response.text()
                    logging.error(
                        f"API Error with status {response.status}: {error_text}. "
                        f"Attempt {attempt + 1}/{MAX_RETRIES}"
                    )
        except asyncio.TimeoutError:
            logging.warning(f"API call timed out. Attempt {attempt + 1}/{MAX_RETRIES}")
        except aiohttp.ClientError as e:
            logging.error(f"AIOHTTP client error: {e}. Attempt {attempt + 1}/{MAX_RETRIES}")
        except Exception as e:
            logging.error(f"An unexpected error in API call: {e}. Attempt {attempt + 1}/{MAX_RETRIES}")

        if attempt < MAX_RETRIES - 1:
            await asyncio.sleep(2 ** attempt)  # Exponential backoff

    pbar.update(1) # Still update progress bar on failure
    return None # Return None if all retries fail

async def run_concurrent_batches(
    batches: List[Dict[str, Any]],
    llm_config: Dict[str, Any],
    concurrency: int
) -> List[Dict[str, Any]]:
    """
    Manages concurrent processing of all batches for a given LLM.
    """
    semaphore = asyncio.Semaphore(concurrency)
    tasks = []

    async with aiohttp.ClientSession() as session:
        with tqdm(total=len(batches), desc=f"Processing batches for {llm_config['model_name']}") as pbar:
            for batch_data in batches:
                async def task_wrapper(batch_info):
                    async with semaphore:
                        selected_indices = await call_llm_api_async(session, batch_info['titles'], llm_config, pbar)
                        return {
                            "original_indices": batch_info['original_indices'],
                            "selected_indices_1_based": selected_indices
                        }
                tasks.append(task_wrapper(batch_data))

            results = await asyncio.gather(*tasks)

    return results

def process_llm(df: pd.DataFrame, llm_name: str, llm_config: Dict[str, Any], batch_size: int, concurrency: int) -> pd.DataFrame:
    """
    Processes all titles for a single LLM, handling batching and checkpointing.
    """
    col_name = f"{llm_name}_select"
    if col_name not in df.columns:
        df[col_name] = pd.NA

    df_to_process = df[df[col_name].isnull()]

    if df_to_process.empty:
        logging.info(f"All titles already processed for {llm_name}. Skipping.")
        return df

    logging.info(f"Found {len(df_to_process)} titles to process for {llm_name}.")

    batches = []
    for i in range(0, len(df_to_process), batch_size):
        batch_df = df_to_process.iloc[i:i + batch_size]
        batches.append({
            "titles": batch_df["Title"].tolist(),
            "original_indices": batch_df.index.tolist()
        })

    batch_results = asyncio.run(run_concurrent_batches(batches, llm_config, concurrency))

    for result in batch_results:
        if result["selected_indices_1_based"] is not None:
            selected_indices_0_based = {idx - 1 for idx in result["selected_indices_1_based"]}
            for i, original_df_index in enumerate(result["original_indices"]):
                df.loc[original_df_index, col_name] = 1 if i in selected_indices_0_based else 0

    df[col_name] = df[col_name].astype('Int64')
    return df

def main():
    """
    Main function to orchestrate the paper filtering process.
    """
    parser = argparse.ArgumentParser(description="Filter research papers using LLMs.")
    parser.add_argument("--venue", type=str, required=True, help="Conference or venue name (e.g., 'aaai').")
    parser.add_argument("--year", type=str, required=True, help="Year of the conference (e.g., '2020').")
    parser.add_argument("--llms", nargs='+', required=True, help="Space-separated list of LLM names to run (e.g., 'deepseek-v3-official').")
    parser.add_argument("--batch_size", type=int, default=20, help="Number of titles to process in each API call.")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of concurrent API calls.")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    input_path = project_root / "title_extract" / "results" / args.venue / f"{args.year}.csv"
    output_dir = project_root / "filter_paper" / "results" / args.venue
    output_path = output_dir / f"{args.year}.csv"

    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.is_file():
        logging.error(f"Input file not found: {input_path}. Skipping this run.")
        return

    if output_path.is_file():
        logging.info(f"Resuming from existing output file: {output_path}")
        df = pd.read_csv(output_path)
    else:
        logging.info(f"Starting from input file: {input_path}")
        df = pd.read_csv(input_path)
        if 'ID' not in df.columns:
            df.insert(0, 'ID', range(len(df)))

    for llm_name in args.llms:
        if llm_name not in LLM_MODELS_SETTINGS:
            logging.warning(f"LLM '{llm_name}' not found in settings. Skipping.")
            continue

        llm_config = LLM_MODELS_SETTINGS[llm_name]
        logging.info(f"--- Starting processing for LLM: {llm_name} ---")

        if not llm_config.get("api_key"):
            logging.error(f"API key for {llm_name} is not set in the .env file. Skipping.")
            continue

        df = process_llm(df, llm_name, llm_config, args.batch_size, args.concurrency)

        logging.info(f"Saving intermediate results to {output_path}")
        df.to_csv(output_path, index=False)
        logging.info(f"--- Finished processing for LLM: {llm_name} ---")

    logging.info(f"Processing for {args.venue} {args.year} complete. Final results saved.")

if __name__ == "__main__":
    main()