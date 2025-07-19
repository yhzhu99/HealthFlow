# extract_task/extract.py
import argparse
import json
import logging
import os
import re
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

PROMPT_TEMPLATE = """
### CORE MISSION ###
Please dissect the provided research paper and generate a series of self-contained, complex "mini-projects." These projects will be used to evaluate an advanced AI agent's ability to implement algorithms and reproduce scientific findings. You will generate approximately 5 such projects.

### ABSOLUTE RULES FOR TASK GENERATION (NON-NEGOTIABLE) ###

1.  **ZERO-REFERENCE MANDATE:** The `task` description MUST be entirely self-contained. It must NOT reference the source paper in any way (e.g., "as described in Section 3," "using Equation (5)," "from Table 2"). The AI agent performing the task will NOT have access to the paper. All necessary information, formulas, parameters, and constants must be explicitly defined within the `task` string.

    *   **FAILURE EXAMPLE (DO NOT DO THIS):** "Implement the time-aware graph attention mechanism from equations (1)-(4)."
    *   **SUCCESS EXAMPLE (DO THIS):** "Implement a time-aware graph attention mechanism. Given a head entity \\(c_h\\), its neighbors \\(N_h\\), and a time interval \\(\\tau\\), compute the aggregated neighbor representation \\(e_{{N_h}}\\). First, compute the time embedding \\(f_\\tau = \\tanh(W_f \\tau + b_f)\\). Then, for each neighbor \\(c_u \\in N_h\\), calculate the attention score \\(\\pi(c_h, r, c_u, \\tau)\\) using a feed-forward network: \\(\\text{{FFN}}(M_r e_h \\| M_r e_u \\| f_\\tau)\\), where \\(\\|\\) denotes concatenation. Normalize these scores using softmax to get \\(\\tilde{{\\pi}}\\). Finally, compute \\(e_{{N_h}} = \\sum_{{c_u \\in N_h}} \\tilde{{\\pi}}(c_h, r, c_u, \\tau) e_u\\). Use parameter dimensions: \\(e_h, e_u \\in \\mathbb{{R}}^{{100}}\\), \\(M_r \\in \\mathbb{{R}}^{{100 \\times 100}}\\), \\(W_f \\in \\mathbb{{R}}^{{64}}\\), \\(b_f \\in \\mathbb{{R}}^{{64}}\\)."

2.  **MANDATORY LATEX FOR ALL MATH:** All mathematical variables, formulas, and expressions MUST be formatted using LaTeX syntax.
    *   For inline math, use `\\( ... \\)`. Example: The loss is calculated for each sample \\(i\\).
    *   For block/display math, use `\\[ ... \\]`. Example: \\[ L_{{\\text{{total}}}} = \\sum_{{i=1}}^{{N}} (y_i - \\hat{{y}}_i)^2 \\]

3.  **DIVERSE & SUBSTANTIAL TASKS:** Generate a variety of tasks covering different research stages (e.g., Cohort Definition, Feature Engineering, Model Implementation, Evaluation). Each task should be a meaningful unit of work, not a trivial query.

4.  **VERIFIABLE ANSWER:** The `answer` field must contain the specific, verifiable result from the paper that directly corresponds to the completion of the `task`. The answer must also include a brief interpretation of the result's significance within the context of the study. Use LaTeX for any math in the answer.

### JSON OUTPUT SPECIFICATION (STRICTLY ENFORCED) ###
Your entire output must be a single, valid JSON array `[[...]]`. Do not include any text, explanations, or markdown fences before or after the JSON. Each object within the array must have exactly these three keys:

1.  `category` (string): A descriptive category for the task (e.g., "Cohort Definition", "Feature Engineering", "Model Implementation", "Model Evaluation").
2.  `task` (string): The detailed, self-contained, imperative instructions for the AI agent, following all rules above. **All math must be in LaTeX.**
3.  `answer` (string): The verifiable result from the paper. This should contain the specific value/outcome and a brief sentence explaining its context. **Any math must be in LaTeX.**

--- BEGIN RESEARCH PAPER TEXT ---
{paper_text}
--- END RESEARCH PAPER TEXT ---
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
    json_str = ""
    try:
        # Handle case where the JSON is wrapped in ```json ... ```
        if '```json' in response_text:
            json_str = response_text.split('```json')[1].split('```')[0].strip()
        # Handle case where the JSON starts immediately
        elif response_text.strip().startswith('['):
            json_str = response_text.strip()
        else:
            # Find the start of the JSON array if it's embedded
            start_index = response_text.find('[')
            end_index = response_text.rfind(']')
            if start_index != -1 and end_index != -1 and end_index > start_index:
                json_str = response_text[start_index : end_index + 1]
            else:
                logger.error("No JSON array start/end tokens '[' or ']' found in response.")
                return None

        # Added check for empty string before attempting to load JSON
        if not json_str:
            logger.error("Extracted JSON string is empty.")
            return None

        json_str = re.sub(r'\\(?![/"\\bfnrtu])', r'\\\\', json_str)

        parsed_json = json.loads(json_str)

        if isinstance(parsed_json, list):
            return parsed_json
        else:
            logger.error(f"Parsed JSON is not a list. Type: {type(parsed_json)}")
            return None

    except (json.JSONDecodeError, IndexError) as e:
        logger.error(f"Failed to decode JSON from response. Error: {e}. Snippet: {response_text[:500]}...")
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
    Main processing logic for a single paper ID: reads file, calls LLM, saves results.

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
        if not paper_text.strip():
            logger.error(f"[{paper_id}] Markdown file is empty. Skipping.")
            return
    except Exception as e:
        logger.error(f"[{paper_id}] Failed to read markdown file {markdown_path}: {e}")
        return

    # 2. Call LLM to extract tasks
    tasks = call_llm(client, paper_text, paper_id, model_name)

    # 3. Validate and save results
    if tasks is None:
        logger.error(f"[{paper_id}] Failed to extract tasks from LLM. No output will be saved.")
        return

    valid_tasks = []
    for i, task in enumerate(tasks):
        if isinstance(task, dict) and all(k in task for k in ["category", "task", "answer"]):
            valid_tasks.append(task)
        else:
            logger.warning(f"[{paper_id}] Skipping malformed task item #{i+1}: {str(task)[:100]}...")

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
        description="Extracts complex, self-contained tasks from a research paper's markdown file using an LLM."
    )
    parser.add_argument(
        "--paper_id",
        type=str,
        required=True,
        help="The ID of the paper to process (e.g., '2203.01077')."
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
        default="deepseek-r1-official",
        choices=list(LLM_MODELS_SETTINGS.keys()),
        help=f"The LLM to use for extraction. Available: {list(LLM_MODELS_SETTINGS.keys())}"
    )
    args = parser.parse_args()

    # --- Check if output file already exists before any processing ---
    output_file_path = args.output_dir / f"{args.paper_id}_tasks.json"
    if output_file_path.is_file():
        logger.info(f"Output file for paper ID '{args.paper_id}' already exists at {output_file_path}. Skipping.")
        return  # Exit gracefully

    # --- LLM Client Initialization ---
    model_key = args.llm
    settings = LLM_MODELS_SETTINGS.get(model_key)
    if not settings:
        logger.error(f"LLM settings for '{model_key}' not found.")
        sys.exit(1)

    api_key = settings.get("api_key")
    base_url = settings.get("base_url")
    model_name = settings.get("model_name")

    if not api_key:
        # Determine which environment variable is missing for the error message.
        required_env_var_map = {
            "deepseek": "DEEPSEEK_API_KEY",
            "qwen": "DASHSCOPE_API_KEY"
        }
        env_var_name = next((v for k, v in required_env_var_map.items() if k in model_key), "the required API key")
        logger.error(f"API key for '{model_key}' is not set. Please set {env_var_name} in your .env file or environment.")
        sys.exit(1)

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client for '{model_key}'. Error: {e}")
        sys.exit(1)

    # --- Process the Paper ---
    process_paper(args.paper_id, args.markdowns_dir, args.output_dir, client, model_name)


if __name__ == "__main__":
    main()