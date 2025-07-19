# extract_task/extract.py
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError, RateLimitError, APIConnectionError

# --- Environment and Configuration ---
# Load environment variables from a .env file in the project root
load_dotenv(dotenv_path='.env')

# --- Centralized LLM Settings ---
# This dictionary holds the configurations for different LLMs.
# The script selects one based on the --llm command-line argument.
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

# API call settings
API_TIMEOUT_SECONDS = 180
MAX_RETRIES = 3

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- LLM Prompt Template (Optimized for Diverse Task Extraction) ---
PROMPT_TEMPLATE = """
### OBJECTIVE ###
Your objective is to deconstruct a research paper into a series of self-contained, actionable tasks. Each task must represent a distinct, verifiable step in the research process—spanning from data preparation and modeling to creating tables, and interpreting results. The final output will be a benchmark of solvable challenges for an advanced AI agent.

### CORE PRINCIPLE: TRANSFORM DESCRIPTION INTO ACTION ###
Your primary function is to convert the paper's *descriptive* text (e.g., "we did X") into an *imperative*, actionable instruction (e.g., "Do X"). You are not summarizing the paper; you are creating a list of executable tests based on it.

-   **Original Paper Text (Descriptive):** "We selected patients older than 18 and calculated the SOFA score for the cohort."
-   **Your Formulated Task (Imperative):** "Given a dataset of patient admissions, filter it to include only patients aged 18 or older. Then, for each patient in the resulting cohort, calculate their SOFA score."

### CONTEXT OF THE RESEARCH ###
The source material is a research paper about applying data science to Electronic Health Records (EHR). The data involved can be:
- Structured time-series data (e.g., vital signs, lab results)
- Clinical notes (unstructured text)
- Medical codes (e.g., ICD codes)
Tasks should be formulated for any of these data types as found in the paper.

### REQUIRED JSON OUTPUT FORMAT ###
You MUST provide the output as a single, valid JSON array of objects. Do not include any text, notes, or explanations outside of this JSON structure. Each object represents a single task and MUST have the following three keys:

1.  `category` (string): A concise, high-level classification of the task's domain. Be consistent. These categories should cover the entire research lifecycle. Examples:
    - "Cohort Definition"
    - "Data Preprocessing"
    - "Feature Engineering"
    - "Predictive Modeling"
    - "Model Evaluation"
    - "Performance Table Generation"
    - "Statistical Analysis"
    - "Result Interpretation"
    *Do not be limited by these examples. Create new, appropriate categories as needed.*

2.  `task` (string): An actionable and self-contained challenge for an AI agent. This description MUST:
    - Be an **imperative command** (e.g., "Generate...", "Calculate...", "Filter...", "Implement...").
    - Clearly specify the **input data** and its assumed structure (e.g., "Given a pandas DataFrame of patient demographics...").
    - State the **precise instructions and methodology** to be followed, as described in the paper.
    - Be **solvable without reading the original paper**.

3.  `answer` (string): The ground truth result, definition, or finding from the paper that serves as the "solution" to the `task`. This will be used for evaluation. It must:
    - Directly correspond to its `task`.
    - Be a direct quote or a very faithful, close paraphrase from the paper.

### GUIDING EXAMPLES ###
```json
[
  {
    "category": "Cohort Definition",
    "task": "From a raw hospital admissions dataset (e.g., MIMIC-IV), define a study cohort by applying the following inclusion criteria: 1. Patient age at admission is 18 years or older. 2. It is the patient's first ICU admission. 3. The ICU length of stay is at least 24 hours.",
    "answer": "The final cohort consisted of 34,490 ICU stays from adult patients (age >= 18) with a length of stay of at least 24 hours, considering only the first ICU admission for each patient."
  },
  {
    "category": "Predictive Modeling",
    "task": "Implement and train a LightGBM model to predict in-hospital mortality. The model should be trained using features derived from the first 24 hours of the ICU stay. Evaluate the model using the Area Under the Receiver Operating Characteristic curve (AUROC) on the held-out test set.",
    "answer": "On the test set, our LightGBM model achieved an AUROC of 0.892 for in-hospital mortality prediction."
  },
  {
    "category": "Performance Table Generation",
    "task": "Create a performance comparison table for the mortality prediction task. The table must include the following models as rows: 'Logistic Regression', 'LightGBM', and 'LSTM'. The columns must be the evaluation metrics: 'AUROC', 'AUPRC', and 'F1-Score'.",
    "answer": "Table 3 summarizes model performances. LightGBM achieved the highest AUROC (0.892). The LSTM model yielded the best AUPRC (0.78), and Logistic Regression recorded an AUROC of 0.821."
  }
]
```

### INSTRUCTIONS ###
Now, analyze the following research paper text. Deconstruct it into a diverse set of solvable tasks covering the entire research process. Follow all rules and the JSON format specified above. Ensure every task is self-contained and grounded in the provided text.

--- RESEARCH PAPER TEXT ---
{paper_text}
"""


def find_markdown_path(markdowns_dir: Path, paper_id: str) -> Optional[Path]:
    """
    Finds the full.md file for a given paper ID within the markdowns directory.

    Args:
        markdowns_dir: The base directory containing paper markdown folders.
        paper_id: The ID of the paper to find.

    Returns:
        The Path to the full.md file, or None if not found.
    """
    if not markdowns_dir.is_dir():
        logger.error(f"Markdowns directory not found: {markdowns_dir}")
        return None

    prefix = f"{paper_id}_"
    for item in markdowns_dir.iterdir():
        if item.is_dir() and item.name.startswith(prefix):
            markdown_file = item / "full.md"
            if markdown_file.is_file():
                return markdown_file
            else:
                logger.warning(f"Directory found for {paper_id} but full.md is missing in: {item}")
                return None

    logger.warning(f"No directory found with prefix '{prefix}' in {markdowns_dir}")
    return None


def extract_json_from_response(response_text: str) -> Optional[List[Dict[str, Any]]]:
    """
    Robustly extracts a JSON array from a string that might contain markdown fences.

    Args:
        response_text: The full text response from the LLM.

    Returns:
        A list of dictionaries if a valid JSON array is found, otherwise None.
    """
    try:
        # Handle case where the JSON is wrapped in ```json ... ```
        if '```json' in response_text:
            json_str = response_text.split('```json')[1].split('```')[0].strip()
        else:
            # Find the start of the JSON array and its corresponding closing bracket
            start_index = response_text.find('[')
            if start_index == -1:
                logger.error("No JSON array start token '[' found in response.")
                return None

            open_brackets = 0
            end_index = -1
            for i, char in enumerate(response_text[start_index:]):
                if char == '[':
                    open_brackets += 1
                elif char == ']':
                    open_brackets -= 1
                if open_brackets == 0:
                    end_index = start_index + i
                    break

            if end_index == -1:
                logger.error("Could not find matching closing bracket ']' for JSON array.")
                return None

            json_str = response_text[start_index : end_index + 1]

        parsed_json = json.loads(json_str)
        if isinstance(parsed_json, list):
            return parsed_json
        else:
            logger.error(f"Parsed JSON is not a list. Type: {type(parsed_json)}")
            return None

    except (json.JSONDecodeError, IndexError) as e:
        logger.error(f"Failed to decode JSON from response. Error: {e}. Snippet: {response_text[:300]}...")
        return None


def call_llm(client: OpenAI, paper_text: str, paper_id: str, model_name: str) -> Optional[List[Dict[str, Any]]]:
    """
    Calls the LLM with the paper text to extract tasks, using streaming and retries.

    Args:
        client: An initialized OpenAI client.
        paper_text: The full text of the research paper.
        paper_id: The ID of the paper, for logging purposes.
        model_name: The name of the model to use for the API call.

    Returns:
        A list of extracted task dictionaries, or None on failure.
    """
    prompt = PROMPT_TEMPLATE.format(paper_text=paper_text)
    messages = [{"role": "user", "content": prompt}]

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"[{paper_id}] Calling LLM '{model_name}' (Attempt {attempt + 1}/{MAX_RETRIES})...")
            stream = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                stream=True,
                timeout=API_TIMEOUT_SECONDS,
            )

            full_response = "".join(
                chunk.choices[0].delta.content or "" for chunk in stream
            )

            if not full_response.strip():
                logger.warning(f"[{paper_id}] LLM returned an empty response.")
                continue

            tasks = extract_json_from_response(full_response)
            return tasks

        except (RateLimitError, APIConnectionError) as e:
            wait_time = 2 ** (attempt + 2)  # Exponential backoff (4s, 8s, 16s)
            logger.warning(
                f"[{paper_id}] API Error (Attempt {attempt+1}): {e}. Retrying in {wait_time}s..."
            )
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait_time)
            else:
                logger.error(f"[{paper_id}] API error after {MAX_RETRIES} retries. Giving up.")
                return None
        except OpenAIError as e:
            logger.error(f"[{paper_id}] Non-retriable OpenAI API error: {e}")
            return None
        except Exception as e:
            logger.error(f"[{paper_id}] An unexpected error occurred during API call: {e}")
            return None

    return None


def process_paper(paper_id: str, markdowns_dir: Path, output_dir: Path, client: OpenAI, model_name: str) -> None:
    """
    Main processing logic for a single paper ID.

    Args:
        paper_id: The ID of the paper to process.
        markdowns_dir: The directory containing markdown folders.
        output_dir: The directory to save the output JSON.
        client: An initialized OpenAI client.
        model_name: The specific model name to use for the API call.
    """
    logger.info(f"--- Starting processing for Paper ID: {paper_id} ---")

    # 1. Find and read the markdown file
    markdown_path = find_markdown_path(markdowns_dir, paper_id)
    if not markdown_path:
        logger.error(f"[{paper_id}] Could not find markdown file. Skipping.")
        return

    try:
        paper_text = markdown_path.read_text(encoding='utf-8')
    except Exception as e:
        logger.error(f"[{paper_id}] Failed to read markdown file {markdown_path}: {e}")
        return

    # 2. Call LLM to extract tasks
    tasks = call_llm(client, paper_text, paper_id, model_name)

    # 3. Validate and save results
    if tasks is None:
        logger.error(f"[{paper_id}] Failed to extract tasks from LLM. No output will be saved.")
        return

    # Basic validation of the returned structure
    valid_tasks = []
    for i, task in enumerate(tasks):
        if isinstance(task, dict) and all(k in task for k in ["category", "task", "answer"]):
            valid_tasks.append(task)
        else:
            logger.warning(f"[{paper_id}] Skipping malformed task item #{i}: {task}")

    if not valid_tasks:
        logger.warning(f"[{paper_id}] No valid tasks were extracted. Not saving file.")
        return

    output_path = output_dir / f"{paper_id}_tasks.json"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(valid_tasks, f, indent=2, ensure_ascii=False)
        logger.info(f"[{paper_id}] Successfully extracted {len(valid_tasks)} tasks. Saved to {output_path}")
    except Exception as e:
        logger.error(f"[{paper_id}] Failed to write tasks to {output_path}: {e}")

    logger.info(f"--- Finished processing for Paper ID: {paper_id} ---")


def main():
    """Parses command-line arguments and orchestrates the extraction process for one paper."""
    parser = argparse.ArgumentParser(
        description="Extracts EHR-related tasks from a research paper's markdown file using an LLM."
    )
    parser.add_argument(
        "--paper_id",
        type=str,
        required=True,
        help="The ID of the paper to process."
    )
    parser.add_argument(
        "--markdowns-dir",
        type=Path,
        default=Path("extract_task/assets/markdowns"),
        help="Directory containing paper markdown folders."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("extract_task/tasks"),
        help="Directory to save the output JSON files."
    )
    parser.add_argument(
        "--llm",
        type=str,
        default="deepseek-v3-official",
        choices=list(LLM_MODELS_SETTINGS.keys()),
        help=f"The LLM to use for extraction. Available: {list(LLM_MODELS_SETTINGS.keys())}"
    )
    args = parser.parse_args()

    # --- LLM Client Initialization ---
    model_key = args.llm
    settings = LLM_MODELS_SETTINGS[model_key]
    api_key = settings.get("api_key")
    base_url = settings.get("base_url")
    model_name = settings.get("model_name")

    if not api_key:
        # Determine which environment variable is missing for the error message.
        required_env_var = "DEEPSEEK_API_KEY" if "deepseek" in model_key else "DASHSCOPE_API_KEY" if "qwen" in model_key else "the required API key"
        logger.error(f"API key for '{model_key}' is not set. Please set {required_env_var} in your .env file or environment.")
        sys.exit(1)

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    process_paper(args.paper_id, args.markdowns_dir, args.output_dir, client, model_name)


if __name__ == "__main__":
    main()
