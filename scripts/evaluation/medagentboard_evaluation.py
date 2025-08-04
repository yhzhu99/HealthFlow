import os
import json
import glob
import traceback
import toml
import concurrent.futures
import threading
import argparse  # Import argparse module
from tqdm import tqdm
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# --- Global constants and thread locks ---
# Create thread lock to safely write tqdm logs
tqdm_lock = threading.Lock()

# Define file types
CODE_EXTENSIONS = ['.py']
TEXT_EXTENSIONS = ['.txt', '.json', 'md', '.log']
# Note: We handle CSV specially, only record existence of binary files like PNG/PKL

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run evaluation and aggregation for AI agent submissions using command-line arguments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Display default values in a friendly way
    )
    
    parser.add_argument(
        '--config-file', 
        type=str,
        default='config.toml',
        help="Path to the TOML configuration file for LLMs."
    )
    parser.add_argument(
        '--dataset-path', 
        type=str,
        default='mab_dataset/alita',
        help="Path to the agent's output dataset directory (contains QID subfolders)."
    )
    parser.add_argument(
        '--answer-path', 
        type=str,
        default='mab_answer',
        help="Path to the standard answers directory."
    )
    parser.add_argument(
        '--output-dir', 
        type=str,
        default='medagentboard/evaluation/alita',
        help="Directory to save all evaluation results."
    )
    parser.add_argument(
        '--qid-range', 
        type=str,
        default=None, # Process all found QIDs by default
        help="Specify a range or list of QIDs to process. Examples: '1-10', '5', '1,3,8'. If not set, all QIDs will be processed."
    )
    
    return parser.parse_args()


# --- 1. Configuration loading and model initialization (this part of the code remains unchanged) ---
def load_llm_configs(config_path):
    """Load all LLM configurations from TOML configuration file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = toml.load(f)
        llm_configs = []
        for key, value in config.get('llm', {}).items():
            if all(k in value for k in ['base_url', 'api_key', 'model_name']):
                value['evaluator_name'] = key
                value['max_workers'] = value.get('max_workers', 1)
                llm_configs.append(value)
            else:
                print(f"[Warning] Skipping invalid LLM config '{key}' due to missing keys.")
        if not llm_configs:
            raise ValueError("No valid LLM configurations found in config.toml.")
        return llm_configs
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at '{config_path}'.")

def initialize_llms(configs):
    """Initialize all LLM instances based on configuration list."""
    models = {}
    for config in configs:
        try:
            model_instance = ChatOpenAI(
                model_name=config['model_name'],
                openai_api_key=config['api_key'],
                openai_api_base=config['base_url'],
                temperature=0.0, # Set temperature to 0 for strict evaluation to ensure result stability
                timeout=300, # Increase timeout to handle complex tasks
            )
            models[config['evaluator_name']] = {
                "instance": model_instance,
                "max_workers": config['max_workers']
            }
            print(f"Successfully initialized model '{config['evaluator_name']}' with max_workers={config['max_workers']}")
        except Exception as e:
            print(f"[ERROR] Failed to initialize model '{config['evaluator_name']}': {e}")
    if not models:
        raise ValueError("No LLM models could be initialized.")
    return models

# --- 2. Core working functions (this part of the code remains unchanged) ---

def read_file_content(file_path, max_chars=15000):
    """
    Read file content with intelligent processing for different file types.
    Modified version: Treat CSV files as regular text files, reading their full content (limited by max_chars).
    """
    try:
        ext = os.path.splitext(file_path)[1].lower()

        # Add .csv to the processing logic for text/code files
        if ext in CODE_EXTENSIONS or ext in TEXT_EXTENSIONS or ext == '.csv':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Read at most max_chars + 1 characters to determine if truncation is needed
                content = f.read(max_chars + 1)
                if len(content) > max_chars:
                    return content[:max_chars] + "\n... [File content truncated] ..."
                return content
        
        # Processing logic for binary files remains unchanged
        elif ext in ['.png', '.pkl', '.jpg', '.jpeg', '.bin']:
            size_kb = os.path.getsize(file_path) / 1024
            return f"[Binary File Exists: {os.path.basename(file_path)}, Size: {size_kb:.2f} KB]"
        
        # Processing logic for unsupported file types remains unchanged
        else:
            return f"[Unsupported File Type: {os.path.basename(file_path)}]"
            
    except Exception as e:
        return f"[Error reading file {os.path.basename(file_path)}: {e}]"


def get_files_summary(directory_or_prefix, is_answer=False):
    """Get a summary of files in a directory or with a specific prefix."""
    summary = ""
    if is_answer:
        # [Fix] Answer files are in subdirectories named after QID, so search within the directory
        files = glob.glob(os.path.join(directory_or_prefix, '*'))
    else:
        # Agent-generated files are in the directory
        files = glob.glob(os.path.join(directory_or_prefix, '*'))
        # Exclude result.json
        files = [f for f in files if os.path.basename(f) != 'result.json']

    if not files:
        # Return more informative hints
        if is_answer:
            return f"No standard answer files found in directory: {directory_or_prefix}"
        else:
            return f"No agent-generated files found in directory: {directory_or_prefix}"

    file_list = sorted([os.path.basename(f) for f in files])
    summary += f"File List: {', '.join(file_list) or 'None'}\n\n"

    for file_path in sorted(files):
        if os.path.isfile(file_path):
            summary += f"--- Content of {os.path.basename(file_path)} ---\n"
            summary += read_file_content(file_path)
            summary += f"--- End of {os.path.basename(file_path)} ---\n\n"

    return summary


def build_evaluation_prompt(qid_path, answer_path):
    """Build a complete, highly structured and task-adaptive Prompt for evaluation."""
    qid = os.path.basename(qid_path)
    result_json_path = os.path.join(qid_path, 'result.json')
    if not os.path.exists(result_json_path):
        return None, f"result.json not found in {qid_path}"

    with open(result_json_path, 'r', encoding='utf-8') as f:
        result_data = json.load(f)

    task_brief = result_data.get('task', 'N/A')
    # Use json.dumps to ensure correct format of generated_answer, especially when it's already a json string
    agent_reasoning = json.dumps(result_data.get('generated_answer', 'N/A'), indent=2, ensure_ascii=False)
    task_type = result_data.get('task_type', 'unknown').lower()

    # 获取Agent生成的文件摘要
    agent_files_summary = get_files_summary(qid_path, is_answer=False)

    # 获取标准答案文件摘要
    answer_file_prefix = os.path.join(answer_path, qid)
    standard_answer_summary = get_files_summary(answer_file_prefix, is_answer=True)

    # --- Dynamically generate evaluation dimensions and scoring criteria ---
    # 1. Map specific task_type to four major evaluation dimensions
    if 'preprocessing' in task_type or 'wrangling' in task_type or 'statistics' in task_type or 'querying' in task_type:
        dimension_name = "Data Extraction and Statistical Analysis"
        dimension_indicators = {
            "correctness_of_data_selection": "Correctness of data selection",
            "transformation_logic": "Transformation logic",
            "handling_of_missing_values": "Handling of missing values",
            "appropriateness_of_statistical_methods": "Appropriateness of statistical methods"
        }
        dimension_rubrics = """
### 1. Correctness of Data Selection (1-5 points)
- **5 (Perfect):** The selected data subset (rows and columns) is **exactly** what is required by the task and perfectly matches the standard answer.
- **4 (Minor Deviation):** The core data is correct, but there are minor discrepancies, such as an extra non-essential column or a slightly different but still valid filter condition.
- **3 (Partial Match):** The selected data has some correct elements but misses significant portions of required data or includes a large amount of incorrect data. E.g., correct rows but wrong columns.
- **2 (Incorrect):** The data selection logic is fundamentally flawed. The resulting dataset is largely irrelevant to the task.
- **1 (Critically Flawed):** No data is selected, or the selected data is completely wrong.

### 2. Transformation Logic (1-5 points)
- **5 (Perfect):** All data transformations (e.g., calculations, aggregations, reshaping) are implemented correctly and efficiently, yielding results identical to the standard answer.
- **4 (Mostly Correct):** The logic is sound and achieves the correct outcome, but with minor inefficiencies or stylistic differences from the ideal implementation. The final numbers are correct.
- **3 (Partially Correct):** The transformation logic contains errors that lead to partially incorrect results. For example, a calculation is wrong, or an aggregation is performed on the wrong group.
- **2 (Incorrect):** The transformation logic is fundamentally incorrect and does not produce the required output format or values.
- **1 (Critically Flawed):** The code for transformation is non-functional, absent, or completely irrelevant.

### 3. Handling of Missing Values (1-5 points)
- **5 (Perfect):** Missing values are handled appropriately as dictated by the task or best practices (e.g., imputation, removal) and aligns perfectly with the standard answer's approach.
- **4 (Acceptable Alternative):** An alternative, valid method for handling missing values was used that still leads to a correct or very similar outcome.
- **3 (Suboptimal Handling):** Missing values were handled, but in a way that negatively impacts the final result or is not appropriate for the data type (e.g., filling categorical data with 0).
- **2 (Incorrect Handling):** The method for handling missing values is wrong or leads to significant errors in the analysis.
- **1 (Critically Flawed):** Missing values were completely ignored when they should have been handled, or the handling method caused the process to fail.

### 4. Appropriateness of Statistical Methods (1-5 points)
- **5 (Perfect):** The statistical methods used (e.g., mean, median, standard deviation, t-test) are perfectly appropriate for the data and the question, matching the standard answer.
- **4 (Mostly Appropriate):** The chosen statistical method is valid and yields a correct conclusion, though a slightly more optimal method might exist.
- **3 (Partially Appropriate):** A statistical method was used, but it was not the right choice for the data distribution or task, leading to potentially misleading results.
- **2 (Inappropriate):** The statistical method is clearly wrong (e.g., using mean on ordinal data, correlation on non-linear data).
- **1 (Critically Flawed):** No statistical analysis was performed, or a completely nonsensical method was applied.
"""
    elif 'modeling' in task_type:
        dimension_name = "Predictive Modeling"
        dimension_indicators = {
            "model_selection": "Appropriateness of model selection",
            "training_procedure": "Implementation of training procedures",
            "evaluation_metrics": "Inclusion of necessary evaluation metrics",
            "validation_practices": "Adherence to proper validation practices"
        }
        dimension_rubrics = """
### 1. Appropriateness of Model Selection (1-5 points)
- **5 (Perfect):** The chosen model is highly appropriate for the data type, task (e.g., classification, regression), and complexity, aligning with the standard answer.
- **4 (Acceptable):** The model is a reasonable choice and works, but a more standard or higher-performing model is available and used in the standard answer.
- **3 (Suboptimal):** The chosen model is unconventional or ill-suited for the task, leading to poor performance or unnecessarily complex implementation.
- **2 (Incorrect):** The model type is wrong for the task (e.g., using a regression model for a classification task).
- **1 (Critically Flawed):** No model was selected, or a completely nonsensical choice was made.

### 2. Implementation of Training Procedures (1-5 points)
- **5 (Perfect):** The model training code is bug-free, efficient, and correctly implemented, including data splitting, feature preparation, and fitting. It matches the standard answer's implementation.
- **4 (Mostly Correct):** The training procedure is correct, but with minor issues like suboptimal hyperparameter defaults or inefficient data handling. The model trains successfully.
- **3 (Partially Correct):** The training code has errors that allow it to run but produce a poorly trained model (e.g., data leakage, wrong feature scaling).
- **2 (Incorrect):** The training code has significant bugs that prevent the model from training correctly or at all.
- **1 (Critically Flawed):** The training code is completely non-functional or absent.

### 3. Inclusion of Necessary Evaluation Metrics (1-5 points)
- **5 (Perfect):** All relevant evaluation metrics for the task (e.g., AUC, F1-score for classification; R-squared, MSE for regression) are correctly calculated and reported, matching the standard answer.
- **4 (Mostly Complete):** The primary metrics are reported, but some useful secondary metrics are missing.
- **3 (Partially Complete):** Some key metrics are missing, or the reported metrics are not the most appropriate for the task.
- **2 (Incorrect):** The wrong metrics are calculated (e.g., accuracy on a highly imbalanced dataset without other metrics), or they are calculated incorrectly.
- **1 (Critically Flawed):** No evaluation metrics are provided.

### 4. Adherence to Proper Validation Practices (1-5 points)
- **5 (Perfect):** A proper validation strategy (e.g., train-test split, cross-validation) is used correctly, ensuring no data leakage and providing an unbiased estimate of performance.
- **4 (Acceptable):** A simple train-test split is used correctly, where cross-validation might have been more robust. The validation is still sound.
- **3 (Flawed):** The validation practice has flaws, such as not stratifying a split on an imbalanced dataset or testing on data that was used in training (leakage).
- **2 (Incorrect):** The concept of validation is misunderstood. For example, the model is evaluated on the training set.
- **1 (Critically Flawed):** There is no validation procedure at all.
"""
    elif 'visualization' in task_type:
        dimension_name = "Data Visualization"
        dimension_indicators = {
            "visualization_correctness": "Correctness of visualization techniques",
            "alignment_with_objective": "Alignment with analytical objectives",
            "aesthetic_quality_readability": "Aesthetic quality and readability",
            "file_generation": "Correctness of File Generation"
        }
        dimension_rubrics = """
IF {agent_files_summary} has no content, the visualization task fails, and the overall answer is "no".

**SPECIAL INSTRUCTION FOR VISUALIZATION:** You cannot see the generated image. Your judgment MUST be based on a combination of: 1) The existence and format of the output file (e.g., a `.png` was created). 2) A meticulous comparison of the Agent's plotting **code** against the Standard Answer's plotting **code**. 3) The agent's own text description of the plot. A correct plot is one generated by code that is logically identical to the standard answer's code.

### 1. Correctness of Visualization Techniques (Code-based) (1-5 points)
- **5 (Perfect):** The agent's plotting code is logically identical to the standard answer's code. It selects the correct data, uses the right plot type (e.g., bar, scatter), and correctly maps variables to axes.
- **4 (Mostly Correct):** The plotting code produces the same type of plot with the correct data, but may have minor differences in implementation (e.g., using a different library but achieving the same result).
- **3 (Partially Correct):** The code attempts to create the right kind of plot but makes significant errors, such as plotting the wrong variables, using incorrect data aggregations, or choosing a plot type that obscures the insight.
- **2 (Incorrect):** The plotting code is fundamentally flawed and would produce a visualization that is misleading or completely different from the required one.
- **1 (Critically Flawed):** The plotting code is non-functional, absent, or completely irrelevant.

### 2. Alignment with Analytical Objectives (1-5 points)
- **5 (Perfect):** The visualization (as inferred from the code) directly and clearly answers the question posed in the task brief.
- **4 (Mostly Aligned):** The visualization is relevant but might not be the most effective way to show the specific insight required.
- **3 (Partially Aligned):** The visualization shows related data but does not directly address the core analytical objective of the task.
- **2 (Poorly Aligned):** The visualization is only tangentially related to the task.
- **1 (Critically Flawed):** The visualization is completely irrelevant to the analytical objective.

### 3. Aesthetic Quality and Readability (Inferred) (1-5 points)
- **5 (Perfect):** The code includes clear labels for the title, x-axis, and y-axis, and a legend if necessary. The implementation suggests a clean, professional, and easy-to-read plot.
- **4 (Good):** The plot is mostly readable, but is missing one minor element like a title or a clear axis label.
- **3 (Acceptable):** The plot is generated, but the code lacks any labels, titles, or other elements that aid interpretation.
- **2 (Poor):** The code suggests a messy or confusing plot (e.g., overlapping labels, no clear differentiation of elements).
- **1 (Critically Flawed):** No effort is made to make the plot interpretable.

### 4. Correctness of File Generation (1-5 points)
- **5 (Perfect):** The agent correctly generated the required image file in the correct format (e.g., `output.png`).
- **4 (Minor Issue):** The file was generated in a different but still acceptable image format (e.g., a `.jpg` instead of a `.png`).
- **3 (Incorrect Format):** The output was saved in a non-image format (e.g., `.txt`, `.csv`).
- **2 (Failed Generation):** The code includes a save command, but it is incorrect and would fail, or the generated answer indicates a failure to save.
- **1 (Critically Flawed):** The required image file was **not generated at all**. This is a critical failure.
"""
    elif 'report' in task_type:
        dimension_name = "Report Generation"
        dimension_indicators = {
            "completeness": "Completeness",
            "accuracy": "Accuracy",
            "coherence": "Coherence of synthesized findings",
            "clinical_relevance": "Clinical relevance of conclusions"
        }
        dimension_rubrics = """
### 1. Completeness (1-5 points)
- **5 (Perfect):** The report addresses all parts of the task prompt, synthesizing all required pieces of information from the data. All key findings from the standard answer are present.
- **4 (Mostly Complete):** The report covers the main findings but omits a minor detail or a secondary point.
- **3 (Partially Complete):** The report addresses some of the task requirements but misses major findings or sections. If no markdown file (MD file) is generated, the maximum score is 3.
- **2 (Incomplete):** The report only touches on one aspect of the task and is largely incomplete.
- **1 (Critically Flawed):** The report is empty or does not attempt to address the task.

### 2. Accuracy (1-5 points)
- **5 (Perfect):** All statements, numbers, and conclusions in the report are factually correct and perfectly match the data and the standard answer.
- **4 (Mostly Accurate):** The report contains very minor inaccuracies that do not affect the overall conclusion (e.g., a slightly rounded number, a trivial misstatement).
- **3 (Partially Accurate):** The report contains a mix of correct and incorrect information. Some of the stated facts or numbers are wrong, affecting the validity of the conclusions.
- **2 (Inaccurate):** The majority of the report is factually incorrect. The numbers or statements fundamentally misrepresent the data.
- **1 (Critically Flawed):** The report is entirely fictional, hallucinatory, or contradictory to the data.

### 3. Coherence (1-5 points)
- **5 (Perfect):** The report is well-structured, logical, and easy to follow. It tells a clear story and connects findings together seamlessly.
- **4 (Good):** The report is coherent and understandable, but the structure or flow could be improved.
- **3 (Acceptable):** The report presents a series of facts but fails to synthesize them into a coherent narrative. The points are disconnected.
- **2 (Poor):** The report is rambling, disorganized, and difficult to understand.
- **1 (Critically Flawed):** The report is a jumble of incoherent sentences or bullet points.

### 4. Clinical Relevance of Conclusions (1-5 points)
- **5 (Perfect):** The conclusions drawn are not only accurate but also clinically relevant and insightful, directly aligning with the context of the problem.
- **4 (Relevant):** The conclusions are relevant but may lack depth or fail to highlight the most critical clinical insight.
- **3 (Superficial):** The conclusions are factually correct but superficial, stating the obvious without providing any deeper interpretation or clinical context.
- **2 (Irrelevant):** The conclusions drawn are not relevant to the clinical question at hand.
- **1 (Critically Flawed):** No conclusions are drawn, or they are nonsensical.
"""
    else: # Fallback for unknown task types
        dimension_name = "General Task Execution"
        dimension_indicators = { "overall_correctness": "Overall Correctness", "completeness": "Completeness" }
        dimension_rubrics = """
### 1. Overall Correctness (1-5 points)
- 5: The agent's output is perfectly correct and matches the standard answer.
- 3: The agent's output is partially correct but has significant errors.
- 1: The agent's output is completely incorrect.
### 2. Completeness (1-5 points)
- 5: The agent completed all parts of the task.
- 3: The agent completed some parts of the task.
- 1: The agent did not complete the task.
"""

    # --- Construct the Final Prompt ---
    # The JSON output structure is fixed to simplify aggregation, but the rubrics are task-specific.
    json_output_structure = f"""
{{
  "dimension_analysis": "Your detailed, evidence-based analysis comparing the agent's submission to the standard answer, focusing on the specific dimension of '{dimension_name}'. Justify each of your scores below with specific examples of what was right or wrong.",
  "overall_summary": "Your final, holistic summary of the agent's performance, considering all aspects.",
"""
    for key, name in dimension_indicators.items():
        json_output_structure += f'  "{key}_score": <Integer from 1-5 for: {name}>,\n'
    json_output_structure += """
  "overall_answer":  outputting only 'yes' or 'no'
}
"""

    prompt_text = f"""
You are a hyper-critical and meticulous AI Quality Engineer specializing in clinical data science. Your task is to rigorously evaluate an AI agent's solution. Your judgment must be strict, precise, and based entirely on the comparison between the **Agent's Submission** and the **Ground Truth Standard Answer**. A perfect score is reserved for solutions that are flawless.

A perfect score is reserved only for solutions that are flawless in every aspect. Any deviation, no matter how minor, will result in a reduction of the score. The evaluation criteria must be applied with the utmost precision。

**CRITICAL OUTPUT REQUIREMENT:**
Your entire response MUST be a single, valid, flat (non-nested) JSON object. No other text, explanations, or markdown formatting outside the JSON is permitted.

**REQUIRED JSON STRUCTURE:**
{json_output_structure}

---
**EVALUATION CONTEXT**
---

### 1. Task Brief
{task_brief}

### 2. Task Type and Primary Evaluation Dimension
- **Task Type:** `{task_type}`
- **Primary Evaluation Dimension:** `{dimension_name}`. Your evaluation should focus intensely on the indicators for this dimension.

---
**EVALUATION MATERIALS**
---

### 1. Agent's Submission (To be Evaluated)

#### Agent's Final Answer:
```json
{agent_reasoning}
```

#### Agent's Generated Files & Content:
{agent_files_summary}

### 2. Ground Truth: Standard Answer (The Gold Standard for Correctness)

**The Agent's submission MUST be compared directly against this ground truth.**

#### Standard Answer Files & Content:
{standard_answer_summary}

---
**DETAILED SCORING RUBRICS (1-5 Scale, BE STRICT)**
---

You must score the agent on the following indicators based on the detailed rubrics below.

{dimension_rubrics}

### Overall Answer
The evaluation of this section depends solely on whether the answers provided by **Standard Answer Files & Content** and **Agent's Final Answer** match.
- **yes**: The task is successfully completed, and the solution meets the requirements.
- **no**: The task is not successfully completed, and the solution contains significant errors or fails to meet the task goal."

---
**FINAL INSTRUCTION**
---
Now, perform your evaluation. Compare the agent's submission to the standard answer meticulously. Fill in the required JSON object with your detailed analysis and scores. **Do not add any text before or after the JSON object.**
"""

    return prompt_text, None

# --- 3. Evaluation phase (Phase 1) - Parameter passing adapted ---
def evaluate_qid_with_single_model(qid_path, llm_name, model_instance, answer_path, output_dir):
    """Evaluate a QID with a specified LLM."""
    qid = os.path.basename(qid_path)
    # Use the passed output_dir
    qid_output_dir = os.path.join(output_dir, qid)
    os.makedirs(qid_output_dir, exist_ok=True)
    
    output_file = os.path.join(qid_output_dir, f"{llm_name}.json")
    
    try:
        # answer_path passed as parameter
        prompt_text, error = build_evaluation_prompt(qid_path, answer_path)
        if error:
            return qid, llm_name, "error", f"Prompt creation failed: {error}"

        message = HumanMessage(content=prompt_text)
        response = model_instance.invoke([message])
        llm_output = response.content.strip()

        # Enhance JSON parsing robustness
        if llm_output.startswith("```json"):
            llm_output = llm_output[7:-3].strip()
        
        parsed_json = json.loads(llm_output)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(parsed_json, f, indent=4, ensure_ascii=False)
            
        return qid, llm_name, "success", "Evaluation successful."

    except json.JSONDecodeError:
        error_file = os.path.join(qid_output_dir, f"{llm_name}_error.txt")
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write("--- PROMPT ---\n")
            f.write(prompt_text)
            f.write("\n\n--- LLM RAW OUTPUT ---\n")
            f.write(llm_output)
        return qid, llm_name, "warning", f"Invalid JSON. Raw output and prompt saved to {os.path.basename(error_file)}"
    except Exception as e:
        return qid, llm_name, "error", f"API call failed: {e}\n{traceback.format_exc()}"


def run_evaluation_phase(llm_models, qid_folders, answer_path, output_dir):
    """Create independent thread pools for each LLM and run all evaluation tasks simultaneously."""
    print("\n--- Starting Evaluation Phase (Fully Parallel) ---")
    
    executors = {
        llm_name: concurrent.futures.ThreadPoolExecutor(
            max_workers=model_info['max_workers'],
            thread_name_prefix=f"{llm_name}_worker"
        )
        for llm_name, model_info in llm_models.items()
    }
    all_futures = []
    
    try:
        print("Submitting all tasks to their respective thread pools...")
        for qid_path in qid_folders:
            qid = os.path.basename(qid_path)
            for llm_name, model_info in llm_models.items():
                # Use the passed output_dir
                output_file = os.path.join(output_dir, qid, f"{llm_name}.json")
                if os.path.exists(output_file):
                    # print(f"Skipping already completed task: QID {qid} | Model {llm_name}")
                    continue 

                model_instance = model_info["instance"]
                executor = executors[llm_name]
                # Pass answer_path and output_dir to worker function
                future = executor.submit(evaluate_qid_with_single_model, qid_path, llm_name, model_instance, answer_path, output_dir)
                all_futures.append(future)

        if not all_futures:
            print("All evaluation tasks have already been completed. Nothing to do.")
            return

        print(f"Submitted {len(all_futures)} new evaluation tasks. Awaiting completion...")
        progress_bar = tqdm(concurrent.futures.as_completed(all_futures), total=len(all_futures), desc="Overall Evaluation Progress")
        
        for future in progress_bar:
            try:
                qid, model, status, message = future.result()
                if status in ["error", "warning"]:
                    with tqdm_lock:
                        tqdm.write(f"[{status.upper()}] QID {qid} | Model {model}: {message.splitlines()[0]}")
            except Exception as e:
                with tqdm_lock:
                    tqdm.write(f"[FATAL] A task failed unexpectedly: {e}")

    finally:
        print("\nShutting down all thread pools...")
        for executor in executors.values():
            executor.shutdown(wait=True)
        print("All executors have been shut down.")

    print("\n--- Evaluation Phase Finished ---")


# --- 4. Aggregation phase (Phase 2) - Parameter passing adapted ---
def run_aggregation_phase(qid_folders, output_dir):
    """Aggregate all evaluation results and calculate average scores."""
    print("\n--- Starting Aggregation Phase ---")
    
    all_qids_summaries = []
    
    # Note: score_keys need to match your JSON output structure.
    # The keys here are dynamically generated based on your Prompt, so we need to find all keys ending with _score during aggregation
    # For simplification, we first assume a fixed key list, but a more robust implementation would dynamically discover them.
    # Temporarily use a general method to capture all scores
    
    grand_totals = {}
    grand_counts = {}
    processed_once = False # Flag to extract scoring keys only once from the first file

    for qid_path in tqdm(qid_folders, desc="Aggregating results"):
        qid = os.path.basename(qid_path)
        qid_output_dir = os.path.join(output_dir, qid)
        
        if not os.path.isdir(qid_output_dir):
            continue

        qid_score_sums = {}
        qid_score_counts = {}
        eval_files = glob.glob(os.path.join(qid_output_dir, "*.json"))
        
        successful_evals = 0
        for file_path in eval_files:
            if os.path.basename(file_path) == "_average_scores.json":
                continue
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Dynamically discover score keys
                current_score_keys = [key for key in data.keys() if key.endswith('_score')]
                if not current_score_keys:
                    with tqdm_lock:
                        tqdm.write(f"[Warning] No keys ending with '_score' found in {file_path}. Skipping.")
                    continue

                # Initialize dictionaries
                for key in current_score_keys:
                    qid_score_sums.setdefault(key, 0)
                    qid_score_counts.setdefault(key, 0)
                    grand_totals.setdefault(key, 0)
                    grand_counts.setdefault(key, 0)
                
                # Accumulate scores
                for key in current_score_keys:
                    score = data.get(key)
                    if isinstance(score, (int, float)):
                        qid_score_sums[key] += score
                        qid_score_counts[key] += 1
                
                successful_evals += 1

            except (json.JSONDecodeError, IOError) as e:
                with tqdm_lock:
                    tqdm.write(f"[Warning] Could not process file {file_path}: {e}")
        
        qid_averages = {}
        if successful_evals > 0:
            for key in qid_score_sums:
                if qid_score_counts.get(key, 0) > 0:
                    avg = qid_score_sums[key] / qid_score_counts[key]
                    qid_averages[f"average_{key}"] = round(avg, 2)
                    grand_totals[key] += qid_score_sums[key]
                    grand_counts[key] += qid_score_counts[key]
        
        qid_summary = {
            "qid": qid,
            "successful_evaluations": successful_evals,
            "average_scores": qid_averages
        }
        all_qids_summaries.append(qid_summary)
        
        summary_file_path = os.path.join(qid_output_dir, "_average_scores.json")
        with open(summary_file_path, 'w', encoding='utf-8') as f:
            json.dump(qid_summary, f, indent=4, ensure_ascii=False)

    final_averages = {}
    for key in grand_totals:
        if grand_counts.get(key, 0) > 0:
            final_averages[f"overall_average_{key}"] = round(grand_totals[key] / grand_counts[key], 2)
    
    total_successful_evals = sum(grand_counts.values()) / len(grand_counts) if grand_counts else 0

    final_summary = {
        "overall_average_scores": final_averages,
        "total_qids_processed": len(qid_folders),
        "total_successful_evaluations": int(total_successful_evals),
        "per_qid_summary": all_qids_summaries
    }
    
    final_summary_path = os.path.join(output_dir, "_final_summary.json")
    with open(final_summary_path, 'w', encoding='utf-8') as f:
        json.dump(final_summary, f, indent=4, ensure_ascii=False)
        
    print("\n--- Aggregation Phase Finished ---")
    print("\nFinal Overall Average Scores:")
    print(json.dumps(final_averages, indent=4))
    print(f"\nDetailed summary saved to: {final_summary_path}")


# --- 5. Main execution flow (parameters adapted) ---
def get_qid_folders_to_process(qid_range_str, dataset_path):
    """Return a list of folder paths to process based on the specified QID range."""
    all_potential_folders = glob.glob(os.path.join(dataset_path, '*'))

    qid_folders_to_process = []
    for folder_path in all_potential_folders:
        if os.path.isdir(folder_path):
            qid_str = os.path.basename(folder_path)
            if qid_str.isdigit():
                qid_folders_to_process.append(folder_path)

    # Sort folders by QID
    qid_folders = sorted(
        qid_folders_to_process,
        key=lambda x: int(os.path.basename(x))
    )

    # Return all folders if no range is specified
    if not qid_range_str:
        return qid_folders

    # Filter based on the provided QID range
    try:
        if ',' in qid_range_str: # Handle list format, e.g., '1,4,6,7,8'
            qid_set = set(map(int, qid_range_str.split(',')))
        elif '-' in qid_range_str: # Handle continuous range format, e.g., '1-5'
            start, end = map(int, qid_range_str.split('-'))
            qid_set = set(range(start, end + 1))
        else: # Handle single number format, e.g., '5'
            qid_set = {int(qid_range_str)}
        
        qid_folders = [folder for folder in qid_folders if int(os.path.basename(folder)) in qid_set]
    except ValueError:
        print(f"[Warning] Invalid QID range format: '{qid_range_str}'. Processing all QIDs.")

    return qid_folders


def main(args):
    """Main execution function"""
    # Ensure output directory exists (using passed parameters)
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Use passed parameters
        llm_configs = load_llm_configs(args.config_file)
        llm_models = initialize_llms(llm_configs)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"[CRITICAL ERROR] {e}")
        return

    # Retrieve QID folders using passed parameters
    qid_folders = get_qid_folders_to_process(args.qid_range, args.dataset_path)

    if not qid_folders:
        print(f"No QID folders found in '{args.dataset_path}' for the specified range. Exiting.")
        return
    
    print(f"\nFound {len(qid_folders)} QIDs to process: {[os.path.basename(p) for p in qid_folders]}")

    # Run evaluation and aggregation phases
    run_evaluation_phase(llm_models, qid_folders, args.answer_path, args.output_dir)
    run_aggregation_phase(qid_folders, args.output_dir)
    
    print("\n\nAll processes completed successfully!")


if __name__ == '__main__':
    # 1. Parse command line arguments
    args = parse_arguments()
    # 2. Pass the parsed argument object to the main function
    main(args)