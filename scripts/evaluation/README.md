# Medical AI Agent Evaluation Suite

This repository provides a comprehensive suite of Python scripts designed for the rigorous evaluation of AI agents on various medical reasoning and data analysis benchmarks. The tools automate the assessment process, leveraging Large Language Models (LLMs) as judges for complex, multi-faceted tasks and employing robust extraction methods for calculating accuracy on question-answering datasets.

The primary goal is to offer a standardized, reproducible, and scalable framework for benchmarking the capabilities of medical AI agents.

## Key Features

-   **Multi-Faceted Evaluation:** Supports both complex, open-ended task evaluation and standardized question-answering accuracy measurement.
-   **LLM-as-a-Judge:** Utilizes powerful LLMs with detailed, rubric-based prompts to score agent performance on dimensions like methodological soundness, artifact quality, and presentation clarity.
-   **Dynamic Rubrics:** For complex data science tasks (e.g., in `MedAgentBoard`), the evaluation rubric is dynamically adapted to the specific task type (e.g., Preprocessing, Modeling, Visualization, Reporting), ensuring a highly relevant assessment.
-   **Automated Accuracy Calculation:** For multiple-choice and short-answer benchmarks (`MedAgentsBench`, `HLE`), it uses a hybrid Regex + LLM approach for robust answer extraction and accuracy scoring.
-   **High-Performance Parallel Processing:** Employs concurrent execution to significantly speed up the evaluation of large datasets across multiple LLM evaluators.
-   **Configurable and Extensible:** Easily configure different LLM evaluators (e.g., GPT-4, Claude, Gemini) through a simple `config.toml` file.
-   **Detailed Reporting:** Generates comprehensive JSON and CSV outputs, providing both per-task scores and aggregated final results for easy analysis and comparison.

## Evaluation Scripts Overview

This suite contains specialized scripts for different medical benchmarks:

1.  **`ehrflowbench_evaluation.py`**:
    -   **Purpose:** Evaluates agents on the **EHRFlowBench** benchmark, which involves complex, open-ended tasks based on electronic health records.
    -   **Methodology:** Uses an LLM-as-a-judge approach with a fixed, multi-dimensional rubric assessing `method_soundness`, `presentation_quality`, and `artifact_generation`.
    -   **Input:** Agent's output directories, each identified by a `QID` (Question ID).

2.  **`medagentboard_evaluation.py`**:
    -   **Purpose:** Evaluates agents on a complex, multi-task benchmark (referred to here as **MedAgentBoard**) involving clinical data science tasks.
    -   **Methodology:** A sophisticated LLM-as-a-judge script that uses **dynamic rubrics** tailored to the task type (e.g., modeling, visualization, data wrangling). It compares the agent's generated code and files against a ground-truth solution.
    -   **Input:** Agent's output directories (`QID`), a directory with corresponding ground-truth answers.

3.  **`medagentsbench_evaluation.py`**:
    -   **Purpose:** Measures accuracy on the **MedAgentsBench** benchmark, which consists primarily of multiple-choice questions.
    -   **Methodology:** Extracts the final letter choice (e.g., 'A', 'B') from the agent's generated response using regular expressions, with an LLM fallback for ambiguous cases. Calculates final accuracy.
    -   **Input:** A directory of log files containing the agent's generated answers and reference answers.

4.  **`hle_evaluation.py`**:
    -   **Purpose:** Measures accuracy on the **HLE (Health Large-scale Evaluation)** benchmark.
    -   **Methodology:** Similar to the MedAgentsBench script, it extracts the final answer from the agent's response and compares it to the ground truth to calculate accuracy. It handles both multiple-choice and other short-answer formats.
    -   **Input:** A directory of log files from the agent.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file based on the imports in the scripts. Key libraries include `langchain-openai`, `openai`, `pandas`, `tqdm`, and `toml`.)*

3.  **Configure LLM Evaluators:**
    Create a `config.toml` file in the root directory. This file will store the API credentials and settings for the LLM(s) you want to use for evaluation. You can configure multiple evaluators.

    **`config.toml` Example:**
    ```toml
    [llm.gpt-4-evaluator]
    base_url = "https://api.openai.com/v1"
    api_key = "sk-..."
    model_name = "gpt-4-turbo-preview"
    max_workers = 10 # Number of parallel threads for this evaluator

    [llm.deepseek-evaluator]
    base_url = "https://api.deepseek.com/v1"
    api_key = "sk-..."
    model_name = "deepseek-chat"
    max_workers = 5
    ```

## Usage

All scripts are run from the command line and support various arguments for customization.

### 1. Evaluating Complex Tasks (`EHRFlowBench` and `MedAgentBoard`)

These scripts follow a two-phase process:
-   **Phase 1: Evaluation:** An LLM scores each question instance in parallel.
-   **Phase 2: Aggregation:** The script aggregates all individual scores into a final summary.

#### **EHRFlowBench Evaluation**

```bash
python ehrflowbench_evaluation.py \
    --config-file config.toml \
    --dataset-path /path/to/your/agent_outputs/agent_name \
    --output-dir /path/to/save/evaluation_results \
    --qid-range "1-50"
```
-   `--dataset-path`: Path to the directory containing agent output folders (e.g., `1/`, `2/`, ...).
-   `--output-dir`: Where to save all evaluation JSON files and summaries.
-   `--qid-range` (Optional): Specify which questions to evaluate. Examples: `1-10`, `5`, `1,3,8`. If omitted, all QIDs are processed.

#### **MedAgentBoard Evaluation**

```bash
python medagentboard_evaluation.py \
    --config-file config.toml \
    --dataset-path /path/to/agent_outputse \
    --answer-path /path/to/ground_truth \
    --output-dir /path/to/save/evaluation_results/medagentboard/agent_name \
    --qid-range "1-10"
```
-   `--dataset-path`: Path to the agent's output folders (e.g., `1/`, `2/`, ...).
-   `--answer-path`: **Crucially**, this is the path to the directory containing the ground-truth answers, which are necessary for comparison.
-   `--output-dir`: Where to save evaluation results.
-   `--qid-range` (Optional): Specify which questions to evaluate.

### 2. Evaluating QA Accuracy (`MedAgentsBench` and `HLE`)

These scripts first process all log files to extract answers and then calculate the final accuracy.

#### **MedAgentsBench (Multiple-Choice) Evaluation**

```bash
python medagentsbench_evaluation.py \
    --input_log_dir /path/to/agent_logs \
    --output_log_dir /path/to/processed_logs \
    --dataset_name medagentsbench \
    --evaluator_model_key deepseek-chat-official \
    --models_to_evaluate ColaCare MedAgent
```
-   `--input_log_dir`: Root directory of the raw logs from agent runs.
-   `--output_log_dir`: Directory to save processed logs and the final accuracy report.
-   `--dataset_name`: Name of the dataset, used to structure log directories.
-   `--evaluator_model_key`: The key from `config.toml` for the LLM to use for answer extraction fallback.
-   `--models_to_evaluate`: A space-separated list of agent/model names to evaluate.

#### **HLE Evaluation**

The usage is identical to `MedAgentsBench`, just with a different dataset name.

```bash
python hle_evaluation.py \
    --input_log_dir /path/to/agent_logs \
    --output_log_dir /path/to/processed_logs \
    --dataset_name hle \
    --evaluator_model_key deepseek-chat-official \
    --models_to_evaluate ColaCare SingleLLM_gpt4
```

## Output Structure

-   **Complex Task Evaluators (`ehrflowbench`, `medagentboard`):**
    -   For each `QID`, a subdirectory is created in the output directory.
    -   Inside, a `<evaluator_name>.json` file is saved for each LLM judge, containing the detailed rubric scores and text analysis.
    -   A `_average_scores.json` file summarizes the scores for that specific task.
    -   Finally, a `_final_summary.json` is created in the root of the output directory, containing the overall average scores across all tasks.

-   **QA Accuracy Evaluators (`medagentsbench`, `hle`):**
    -   A `processed_logs` directory mirrors the structure of the input logs, with each JSON file now containing an `extracted_answer` field.
    -   A final `evaluation_summary.csv` file is generated in the output directory. This CSV contains a table with the accuracy, total samples, and correct counts for each evaluated model, sorted by performance.