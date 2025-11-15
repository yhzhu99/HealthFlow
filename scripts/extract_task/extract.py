# extract_task/extract.py
import argparse
import json
import logging
import os
import re
import sys
import time
import atexit
from pathlib import Path
from typing import List, Dict, Any, Optional
import xml.etree.ElementTree as ET

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError, RateLimitError, APIConnectionError

# --- Environment and Configuration ---
# Load environment variables from a .env file in the project root
load_dotenv(dotenv_path='.env')

# --- Centralized LLM Settings ---
LLM_MODELS_SETTINGS = {
    "deepseek-chat-official": {
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": "https://api.deepseek.com",
        "model_name": "deepseek-chat",
    },
    "deepseek-reasoner-official": {
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": "https://api.deepseek.com",
        "model_name": "deepseek-reasoner",
    },
    "qwen3-235b-a22b": {
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_name": "qwen3-235b-a22b",
    },
    "gemini-2.5-pro": {
        "api_key": os.getenv("GEMINI_API_KEY"),
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        # "base_url": "https://api.laozhang.ai/v1",
        "model_name": "gemini-2.5-pro",
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

# --- Concurrency Lock File ---
# Global variable to hold the path to the lock file for cleanup
LOCK_FILE_PATH: Optional[Path] = None

def cleanup_lock():
    """Remove the lock file on exit."""
    if LOCK_FILE_PATH and LOCK_FILE_PATH.exists():
        try:
            LOCK_FILE_PATH.unlink()
            logger.info(f"Successfully removed lock file: {LOCK_FILE_PATH}")
        except OSError as e:
            logger.error(f"Error removing lock file {LOCK_FILE_PATH}: {e}")

# Register the cleanup function to be called on script exit
atexit.register(cleanup_lock)


PROMPT_TEMPLATE = """
### CORE MISSION ###
Please dissect the provided research paper and generate a series of self-contained, complex "mini-projects." These projects will be used to evaluate an advanced AI agent's ability to implement algorithms and reproduce scientific findings. You will generate approximately 5 such projects.

### ABSOLUTE RULES FOR TASK GENERATION (NON-NEGOTIABLE) ###

1.  **ZERO-REFERENCE MANDATE:** The task description MUST be entirely self-contained. It must NOT reference the source paper in any way (e.g., "as described in Section 3," "using Equation (5)," "from Table 2"). The AI agent performing the task will NOT have access to the paper. All necessary information, formulas, parameters, and constants must be explicitly defined within the `<task>`.

    *   **FAILURE EXAMPLE (DO NOT DO THIS):** "Implement the time-aware graph attention mechanism from equations (1)-(4)."
    *   **SUCCESS EXAMPLE (DO THIS):** "Implement a time-aware graph attention mechanism. Given a head entity \\(c_h\\), its neighbors \\(N_h\\), and a time interval \\(\\tau\\), compute the aggregated neighbor representation \\(e_{{N_h}}\\). First, compute the time embedding \\(f_\\tau = \\tanh(W_f \\tau + b_f)\\). Then, for each neighbor \\(c_u \\in N_h\\), calculate the attention score \\(\\pi(c_h, r, c_u, \\tau)\\) using a feed-forward network: \\(\\text{{FFN}}(M_r e_h \\| M_r e_u \\| f_\\tau)\\), where \\(\\|\\) denotes concatenation. Normalize these scores using softmax to get \\(\\tilde{{\\pi}}\\). Finally, compute \\(e_{{N_h}} = \\sum_{{c_u \\in N_h}} \\tilde{{\\pi}}(c_h, r, c_u, \\tau) e_u\\). Use parameter dimensions: \\(e_h, e_u \\in \\mathbb{{R}}^{{100}}\\), \\(M_r \\in \\mathbb{{R}}^{{100 \\times 100}}\\), \\(W_f \\in \\mathbb{{R}}^{{64}}\\), \\(b_f \\in \\mathbb{{R}}^{{64}}\\)."

2.  **MANDATORY LATEX FOR ALL MATH:** All mathematical variables, formulas, and expressions MUST be formatted using LaTeX syntax.
    *   For inline math, use `\\( ... \\)`. Example: The loss is calculated for each sample \\(i\\).
    *   For block/display math, use `\\[ ... \\]`. Example: \\[ L_{{\\text{{total}}}} = \\sum_{{i=1}}^{{N}} (y_i - \\hat{{y}}_i)^2 \\]

3.  **DIVERSE & SUBSTANTIAL TASKS:** Generate a variety of tasks covering different research stages (e.g., Cohort Definition, Feature Engineering, Model Implementation, etc.). Each task should be a meaningful unit of work, not a trivial query.

4.  **VERIFIABLE ANSWER:** The `<answer>` field must contain the specific, verifiable result from the paper that directly corresponds to the completion of the task. The answer must also include a brief interpretation of the result's significance within the context of the study. Use LaTeX for any math in the answer.

### XML OUTPUT SPECIFICATION (STRICTLY ENFORCED) ###
Your entire output must be a single, valid XML block. Do not include any text, explanations, or markdown fences before or after the XML. The root element must be `<response>`. Each task must be enclosed in an `<item>` tag with exactly these three child tags: `<category>`, `<task>`, and `<answer>`:

1. `<category>`: A descriptive category for the task.
2. `<task>`: The detailed, self-contained, imperative instructions for the AI agent, following all rules above. **All math must be in LaTeX.**
3. `<answer>`: The verifiable result from the paper. This should contain the specific value/outcome and a brief sentence explaining its context. **Any math must be in LaTeX.**

--- BEGIN RESEARCH PAPER TEXT ---
{paper_text}
--- END RESEARCH PAPER TEXT ---
"""


def find_markdown_path(markdowns_dir: Path, paper_id: str) -> Optional[Path]:
    """Finds the full.md file for a given paper ID."""
    if not markdowns_dir.is_dir():
        logger.error(f"Markdowns directory not found: {markdowns_dir}")
        return None
    prefix = f"{paper_id}_"
    for item in markdowns_dir.iterdir():
        if item.is_dir() and item.name.startswith(prefix):
            markdown_file = item / "full.md"
            if markdown_file.is_file():
                return markdown_file
    logger.warning(f"No directory found with prefix '{prefix}' in {markdowns_dir}")
    return None


def extract_tasks_from_xml(response_text: str, paper_id: str) -> Optional[List[Dict[str, Any]]]:
    """
    Extracts a list of tasks from an XML-formatted string.

    Args:
        response_text: The full text response from the LLM, expected to be XML.
        paper_id: The ID of the paper for which the tasks are being extracted.

    Returns:
        A list of task dictionaries, or None if parsing fails.
    """
    tasks = []
    try:
        # Clean up the response to remove potential markdown fences
        clean_text = re.sub(r'```xml\s*|\s*```', '', response_text).strip()
        root = ET.fromstring(clean_text)

        for item_node in root.findall('item'):
            category = item_node.find('category')
            task_desc = item_node.find('task')
            answer = item_node.find('answer')

            if all(tag is not None and tag.text for tag in [category, task_desc, answer]):
                tasks.append({
                    'category': category.text.strip(),
                    'task': task_desc.text.strip(),
                    'answer': answer.text.strip()
                })
            else:
                logger.warning("Skipping malformed XML item with missing tags or empty text.")

        return tasks if tasks else None
    except ET.ParseError as e:
        # save the response_text to a file for debugging
        with open(f'{paper_id}_failed_response.xml', 'w') as f:
            f.write(response_text)
        logger.error(f"Failed to parse XML from LLM response. Error: {e}. Snippet: {response_text[:500]}...")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during XML parsing: {e}")
        return None


def call_llm(client: OpenAI, paper_text: str, paper_id: str, model_name: str) -> Optional[List[Dict[str, Any]]]:
    """Calls the LLM to extract tasks, using streaming and retries."""
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

            full_response = "".join(chunk.choices[0].delta.content or "" for chunk in stream)

            if not full_response.strip():
                logger.warning(f"[{paper_id}] LLM returned an empty response.")
                continue

            tasks = extract_tasks_from_xml(full_response, paper_id)
            return tasks

        except (RateLimitError, APIConnectionError) as e:
            wait_time = 2 ** (attempt + 2)
            logger.warning(f"[{paper_id}] API Error (Attempt {attempt+1}): {e}. Retrying in {wait_time}s...")
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
    """Main processing logic for a single paper: reads file, calls LLM, saves results to JSONL."""
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

    # 3. Validate and save results to JSONL
    if not tasks:
        logger.error(f"[{paper_id}] Failed to extract tasks from LLM. No output will be saved.")
        return

    output_path = output_dir / f"{paper_id}_tasks.jsonl"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as f:
            for task in tasks:
                # Write each task as a JSON object on a new line
                f.write(json.dumps(task, ensure_ascii=False) + '\n')
        logger.info(f"[{paper_id}] Successfully extracted {len(tasks)} tasks. Saved to {output_path}")
    except Exception as e:
        logger.error(f"[{paper_id}] Failed to write tasks to {output_path}: {e}")

    logger.info(f"--- Finished processing for Paper ID: {paper_id} ---")


def main():
    """Parses args, handles locking, and orchestrates the extraction process."""
    global LOCK_FILE_PATH

    parser = argparse.ArgumentParser(description="Extracts tasks from a paper's markdown using an LLM and saves as JSONL.")
    parser.add_argument("--paper_id", type=str, required=True, help="ID of the paper to process (e.g., '2203.01077').")
    parser.add_argument("--markdowns-dir", type=Path, default=Path("extract_task/assets/markdowns"), help="Directory of markdown folders.")
    parser.add_argument("--output-dir", type=Path, default=Path("extract_task/tasks"), help="Directory to save output JSONL files.")
    parser.add_argument("--llm", type=str, default="deepseek-reasoner-official", choices=list(LLM_MODELS_SETTINGS.keys()), help="LLM to use for extraction.")
    args = parser.parse_args()

    # --- Concurrency & Pre-flight Checks ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = args.output_dir / f"{args.paper_id}_tasks.jsonl"
    LOCK_FILE_PATH = args.output_dir / f"{args.paper_id}.lock"

    if output_file_path.is_file():
        logger.info(f"Output file for '{args.paper_id}' already exists. Skipping.")
        return

    if LOCK_FILE_PATH.is_file():
        logger.warning(f"Lock file for '{args.paper_id}' exists. Another process is likely working on it. Skipping.")
        return

    # Create lock file
    try:
        LOCK_FILE_PATH.touch()
    except OSError as e:
        logger.error(f"Failed to create lock file {LOCK_FILE_PATH}: {e}. Aborting.")
        return


    # --- LLM Client Initialization ---
    settings = LLM_MODELS_SETTINGS.get(args.llm)
    if not settings:
        logger.error(f"LLM settings for '{args.llm}' not found.")
        sys.exit(1)

    api_key = settings.get("api_key")
    if not api_key:
        env_var_map = {"deepseek": "DEEPSEEK_API_KEY", "qwen": "DASHSCOPE_API_KEY"}
        env_var = next((v for k, v in env_var_map.items() if k in args.llm), "the required API key")
        logger.error(f"API key for '{args.llm}' is not set. Please set {env_var} in your .env file.")
        sys.exit(1)

    try:
        client = OpenAI(api_key=api_key, base_url=settings.get("base_url"))
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client for '{args.llm}'. Error: {e}")
        sys.exit(1)

    # --- Process the Paper ---
    process_paper(args.paper_id, args.markdowns_dir, args.output_dir, client, settings.get("model_name"))


if __name__ == "__main__":
    main()
