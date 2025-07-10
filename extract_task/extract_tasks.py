# extract_task/extract_tasks.py

import os
import json
import fitz  # PyMuPDF
from openai import OpenAI, APIError, AuthenticationError, BadRequestError
from tqdm import tqdm
import logging
import traceback

from config import API_KEY, BASE_URL, MODEL_NAME, PDF_DIR, OUTPUT_DIR

# --- Configuration ---
TEST_MODE = False
MAX_FILES_IN_TEST_MODE = 3
MAX_CHARS_PER_CHUNK = 15000  # ~4000 tokens

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initialize API Client ---
if not API_KEY:
    raise ValueError("API key not found. Please set DEEPSEEK_API_KEY or OPENAI_API_KEY in your environment.")

try:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
except Exception as e:
    logging.error(f"Failed to initialize OpenAI client: {e}")
    exit(1)

# --- The Golden Prompt (No Changes) ---
PROMPT_TEMPLATE = """
### ROLE ###
You are an expert **AI Benchmark Designer**. Your specialty is analyzing biomedical research papers and transforming their methodologies, analyses, and findings into a set of standardized, challenging tasks. These tasks are designed to rigorously evaluate the capabilities of new, powerful foundation models or AI frameworks in the biomedical domain.

### PRIMARY GOAL ###
I will provide you with the text of a biomedical AI research paper. Your primary goal is to extract a structured list of **generalizable benchmark tasks**. These tasks must be challenging enough to test an advanced AI system, not just simple information retrieval that any model can perform. The focus should be on tasks that require reasoning, code generation, data analysis, visualization, and complex instruction-following.

### KEY SHIFT IN PERSPECTIVE ###
This is crucial: you are **not** merely summarizing the paper. You are creating self-contained challenges inspired by the paper. The key shift is from *reporting what the paper did* to *formulating a challenge that another model could attempt*. Each task should be framed as a question or a command.

### OUTPUT FORMAT ###
You MUST provide the output as a single, valid JSON array of objects. Do not include any text or explanations outside of this JSON structure. Each object in the array represents a single benchmark task and MUST have the following three keys:

- "category": A string matching one of the 7 categories defined below. These categories reflect the *type of task* a large model would perform.
- "task_prompt": A clear, concise, and generalizable instruction for an AI model. It MUST be phrased as a question or an imperative command (e.g., "Generate Python code to...", "Based on the provided context, what is the primary limitation of...").
- "context_and_expected_output": The necessary information for a model to attempt the task and for a human to evaluate the response. This includes input data descriptions, methodologies, and—most importantly—the "ground truth" or expected outcome derived from the paper (e.g., an example code snippet, a specific performance metric to achieve, a key insight to identify).

### CATEGORY DEFINITIONS ###
1. **Code Generation**
2. **Visualization Task**
3. **Prediction Task Formulation**
4. **Data Analysis & Processing**
5. **Complex Reasoning & QA**
6. **Results Interpretation**
7. **Hypothesis Generation & Experimental Design**

### GUIDANCE FOR `context_and_expected_output` FIELD ###

- **For "Code Generation"**:
    - **Context**: Describe the goal of the code (e.g., "To normalize a Pandas DataFrame using z-scores," "To implement a specific layer in PyTorch").
    - **Expected Output**: Provide the **exact code snippet** from the paper if available. If not, provide a detailed, step-by-step description of the required logic and functions that must be used.

- **For "Visualization Task"**:
    - **Context**: Describe the type of data being visualized (e.g., "High-dimensional patient embeddings").
    - **Expected Output**: Provide either the code used to generate the plot, or a detailed description of what the plot should contain (axes, labels, colors, insights, etc.).

- **For "Prediction Task Formulation"**:
    - **Context**: Define the clinical prediction goal and the model inputs.
    - **Expected Output**: Specify the target variable, model type used in the paper, evaluation metric, and expected performance benchmark.

- **For "Data Analysis & Processing"**:
    - **Context**: Describe input data issues (e.g., missing values).
    - **Expected Output**: Provide the exact method and settings (e.g., "used sklearn's IterativeImputer with default parameters").

- **For "Complex Reasoning & QA"**:
    - **Context**: Include a methodology excerpt or result to analyze.
    - **Expected Output**: Provide the correct interpretation or explanation from the paper.

- **For "Results Interpretation"**:
    - **Context**: Show a comparison between models or results.
    - **Expected Output**: Summarize the paper's reasoning for why one model outperformed another.

- **For "Hypothesis Generation & Experimental Design"**:
    - **Context**: State a limitation or future direction.
    - **Expected Output**: Propose a novel, testable follow-up experiment inspired by the paper.

### EXAMPLE ###
**Input Text Snippet:**
"...To visualize the patient representations, we first projected the 256-dimensional embeddings from our model's final hidden layer into a 2-dimensional space using UMAP with `n_neighbors=15` and `min_dist=0.1`. The resulting plot was colored by the patient's ground-truth mortality status. This revealed distinct clusters for survivors and non-survivors. The code used was:
```python
import umap
import matplotlib.pyplot as plt

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding = reducer.fit_transform(patient_embeddings)
plt.scatter(embedding[:, 0], embedding[:, 1], c=mortality_labels, cmap='Spectral')
plt.title('UMAP projection of patient embeddings')
plt.show()

### RESEARCH PAPER TEXT ###
{}
"""

def extract_text_from_pdf(pdf_path):
    """Extracts plain text from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = " ".join([page.get_text() for page in doc])
        doc.close()
        return text
    except Exception as e:
        logging.error(f"Error extracting text from {os.path.basename(pdf_path)}: {e}")
        return None

def extract_json_from_response(response_text):
    """Robustly extracts a JSON array from a string that might contain other text."""
    start_index = response_text.find('[')
    if start_index == -1:
        return None

    end_index = -1
    open_brackets = 0
    for i in range(start_index, len(response_text)):
        if response_text[i] == '[':
            open_brackets += 1
        elif response_text[i] == ']':
            open_brackets -= 1
            if open_brackets == 0:
                end_index = i
                break

    if end_index == -1:
        return None

    json_str = response_text[start_index : end_index + 1]
    return json.loads(json_str)

def get_tasks_from_text(text_chunk, paper_name):
    """Sends a text chunk to the LLM API and gets a structured task list."""
    if not text_chunk.strip():
        return []

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(text_chunk)}],
            temperature=0.0,
        )
        content = response.choices[0].message.content
        result = extract_json_from_response(content)

        if result is None:
            logging.warning(f"Could not find a valid JSON array in the response for {paper_name}. Response: {content[:500]}...")
            return []

        if isinstance(result, list):
            return result

        logging.warning(f"Unexpected JSON structure returned for {paper_name}: {result}")
        return []

    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON from API response for {paper_name}: {e}\nResponse: {content}")
        return []
    except (AuthenticationError, BadRequestError) as e:
        logging.error(f"CRITICAL API ERROR for {paper_name}. Type: {type(e).__name__}. This often means your API Key, Base URL, or Model Name is wrong. Please check your .env and config.py files.")
        logging.error(f"Error Details: {e}")
        return None
    except APIError as e:
        logging.error(f"API Error for {paper_name} (e.g., rate limit, server error): {e}")
        return []
    except Exception as e:
        logging.error(f"An unexpected error occurred for {paper_name}: {e}")
        logging.error(f"Full Traceback: {traceback.format_exc()}")
        return []

def process_single_paper(pdf_path):
    """Processes a single paper: extract text, chunk, call API, merge, and save."""
    paper_name = os.path.basename(pdf_path)
    logging.info(f"Processing paper: {paper_name}")

    full_text = extract_text_from_pdf(pdf_path)
    if not full_text:
        return

    text_chunks = [full_text[i:i + MAX_CHARS_PER_CHUNK] for i in range(0, len(full_text), MAX_CHARS_PER_CHUNK)]
    logging.info(f"Split text into {len(text_chunks)} chunk(s) for {paper_name}.")

    all_tasks = []
    for i, chunk in enumerate(text_chunks):
        logging.info(f"Processing chunk {i+1}/{len(text_chunks)}...")
        tasks = get_tasks_from_text(chunk, paper_name)

        if tasks is None:
            logging.error(f"Stopping processing for {paper_name} due to a critical API error.")
            all_tasks = []
            break

        all_tasks.extend(tasks)

    unique_tasks = []
    seen = set()
    for task in all_tasks:
        if (
            isinstance(task, dict)
            and 'category' in task
            and 'task_prompt' in task
            and 'context_and_expected_output' in task
        ):
            identifier = (task['category'], task['task_prompt'].strip())
            if identifier not in seen:
                unique_tasks.append(task)
                seen.add(identifier)
        else:
            logging.warning(f"Skipping malformed task item from API: {task}")


    output_filename = paper_name.replace('.pdf', '.json')
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unique_tasks, f, indent=2, ensure_ascii=False)

    logging.info(f"Successfully extracted {len(unique_tasks)} unique tasks. Saved to {output_path}")

def main():
    pdf_files = sorted([f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")])
    if TEST_MODE:
        pdf_files = pdf_files[:MAX_FILES_IN_TEST_MODE]
        logging.info(f"Running in TEST_MODE. Will process {len(pdf_files)} file(s).")
    else:
        logging.info(f"Found {len(pdf_files)} PDF file(s) to process.")

    for filename in tqdm(pdf_files, desc="Processing Papers"):
        pdf_path = os.path.join(PDF_DIR, filename)
        process_single_paper(pdf_path)

if __name__ == "__main__":
    main()
