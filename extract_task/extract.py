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
try:
    project_root = Path(__file__).resolve().parent.parent
    dotenv_path = project_root / '.env'
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
    else:
        print(f"Warning: .env file not found at {dotenv_path}. Relying on system environment variables.")
except Exception as e:
    print(f"Error loading .env file: {e}")

# Use DeepSeek as the default LLM, which is compatible with the OpenAI SDK
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY")
LLM_BASE_URL = "https://api.deepseek.com"
LLM_MODEL_NAME = "deepseek-chat"

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

# --- LLM Prompt Template ---
PROMPT_TEMPLATE = """
### ROLE & GOAL ###
You are a meticulous research assistant. Your goal is to read the provided text from a biomedical research paper and extract all tasks related to the use of Electronic Health Records (EHR). A "task" is a distinct step or procedure performed by the researchers, such as defining a patient group, preprocessing data, or building a model.

### CONTEXT ###
The research papers are about applying AI and data science to EHR data, which can include:
- Structured time-series data (e.g., vital signs, lab results)
- Clinical notes (unstructured text)
- Medical codes (e.g., ICD, CPT codes)
Your extraction should cover tasks related to any of these data types.

### OUTPUT FORMAT ###
You MUST provide the output as a single, valid JSON array of objects. Do not include any text or explanations outside of this JSON structure. Each object in the array represents a single extracted task and MUST have the following three keys:

1.  `category` (string): A concise, high-level classification of the task, in a few words. You should generate this category based on the task's nature. Examples include:
    - "Cohort Definition"
    - "Data Preprocessing"
    - "Feature Engineering"
    - "Predictive Modeling"
    - "Model Evaluation"
    - "Statistical Analysis"
    - "Causal Inference"
    *Do not be limited by these examples. Create new categories if they are more appropriate.*

2.  `task` (string): A detailed, self-contained description of the task. This description should be clear enough to be understood without reading the full paper. It should summarize what was done, what data was used, and what the objective was.

3.  `answer` (string): The evidence for the task. This MUST be a direct quote or a very faithful, close paraphrase from the paper. If the evidence is from a table or figure, you must describe its content and cite it (e.g., "As shown in Table 2, the model achieved an AUROC of 0.91...").

### EXAMPLE ###
```json
[
  {
    "category": "Cohort Definition",
    "task": "Define a patient cohort for sepsis prediction. The process involves selecting adult patients (age >= 18) from the MIMIC-IV database who were admitted to the ICU. Sepsis is identified based on the Sepsis-3 criteria, requiring evidence of suspected infection and an acute increase in the SOFA score of 2 points or more.",
    "answer": "We identified a cohort of adult patients (age ≥ 18 years) from their first ICU admission in the MIMIC-IV v2.0 database... Sepsis was defined according to the Sepsis-3 criteria, which includes suspected infection and a sequential organ failure assessment (SOFA) score increase of ≥2 points."
  },
  {
    "category": "Data Preprocessing",
    "task": "Process and clean time-series vital sign data for the defined cohort. This includes handling missing values by carrying the last observation forward (LOCF) and removing variables with more than 80% missingness. Outliers were addressed by winsorizing at the 1st and 99th percentiles.",
    "answer": "For missing data in time-series variables, we used the last observation carried forward method. Variables with over 80% missing values were excluded from the analysis. To handle extreme values, we applied winsorization at the 1st and 99th percentiles."
  }
]
```

### INSTRUCTIONS ###
Now, analyze the following research paper text and extract all EHR-related tasks according to the format specified above.

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


def call_llm(client: OpenAI, paper_text: str, paper_id: str) -> Optional[List[Dict[str, Any]]]:
    """
    Calls the LLM with the paper text to extract tasks, using streaming and retries.

    Args:
        client: An initialized OpenAI client.
        paper_text: The full text of the research paper.
        paper_id: The ID of the paper, for logging purposes.

    Returns:
        A list of extracted task dictionaries, or None on failure.
    """
    prompt = PROMPT_TEMPLATE.format(paper_text=paper_text)
    messages = [{"role": "user", "content": prompt}]

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"[{paper_id}] Calling LLM (Attempt {attempt + 1}/{MAX_RETRIES})...")
            stream = client.chat.completions.create(
                model=LLM_MODEL_NAME,
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


def process_paper(paper_id: str, markdowns_dir: Path, output_dir: Path, client: OpenAI) -> None:
    """
    Main processing logic for a single paper ID.

    Args:
        paper_id: The ID of the paper to process.
        markdowns_dir: The directory containing markdown folders.
        output_dir: The directory to save the output JSON.
        client: An initialized OpenAI client.
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
    tasks = call_llm(client, paper_text, paper_id)

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
    args = parser.parse_args()

    if not LLM_API_KEY:
        logger.error("DEEPSEEK_API_KEY environment variable not set. Please set it in your .env file or environment.")
        sys.exit(1)

    client = OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
    )

    process_paper(args.paper_id, args.markdowns_dir, args.output_dir, client)


if __name__ == "__main__":
    main()