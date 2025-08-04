import os
import json
import glob
import traceback
import toml
import concurrent.futures
import threading
import argparse  # 导入 argparse 模块
from tqdm import tqdm
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# --- 全局常量和线程锁 ---
# 创建一个线程锁，用于在多线程环境中安全地向tqdm写入日志
tqdm_lock = threading.Lock()

# File type definitions
CODE_EXTENSIONS = ['.py']
TEXT_EXTENSIONS = ['.txt', '.json', '.md', '.html', '.css', '.js', '.log']


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Run HealthFlow evaluation using command-line arguments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
        default='healthflow',
        help="Path to the agent's output dataset directory (contains QID subfolders)."
    )
    parser.add_argument(
        '--output-dir', 
        type=str,
        default='healthflow_evaluation',
        help="Directory to save all evaluation results."
    )
    parser.add_argument(
        '--qid-range', 
        type=str,
        default=None, # 默认处理所有找到的QID
        help="Specify a range or list of QIDs to process. Examples: '1-10', '5', '1,3,8'. If not set, all QIDs will be processed."
    )
    
    return parser.parse_args()


# --- 1. 配置加载与模型初始化 ---
def load_llm_configs(config_path):
    """从 TOML 配置文件中加载所有 LLM 配置，包括 max_workers。"""
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
    """根据配置列表初始化所有 LLM 实例。"""
    models = {}
    for config in configs:
        try:
            model_instance = ChatOpenAI(
                model_name=config['model_name'],
                openai_api_key=config['api_key'],
                openai_api_base=config['base_url'],
                temperature=0.1
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


# --- 2. 核心工作函数 ---
def read_text_file(file_path, max_chars=20000):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(max_chars + 1)
            if len(content) > max_chars:
                return content[:max_chars] + "\n... [File content truncated due to size] ..."
            return content
    except Exception as e:
        return f"Error reading text file: {e}"

def get_directory_structure(root_dir):
    structure = f"Directory structure for root '{os.path.basename(root_dir)}':\n"
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith('.')]
        relative_path = os.path.relpath(dirpath, root_dir)
        level = 0 if relative_path == "." else len(relative_path.split(os.sep))
        indent = '    ' * level
        structure += f"{indent}└── {os.path.basename(dirpath)}/\n"
        sub_indent = '    ' * (level + 1)
        for f in sorted(filenames):
            structure += f"{sub_indent}├── {f}\n"
    return structure

def build_evaluation_prompt(qid_path):
    """构建评估所需的完整Prompt。"""
    result_json_path = os.path.join(qid_path, 'result.json')
    if not os.path.exists(result_json_path):
        return None, "result.json not found"
    
    with open(result_json_path, 'r', encoding='utf-8') as f:
        result_data = json.load(f)
        
    task_brief = result_data.get('task', 'N/A')
    reference_answer = json.dumps(result_data.get('reference_answer', 'N/A'), indent=2, ensure_ascii=False)
    agent_reasoning = json.dumps(result_data.get('generated_answer', 'N/A'), indent=2, ensure_ascii=False)
    
    code_files_content, csv_files_summary = "", ""
    file_structure_info = get_directory_structure(qid_path)
    all_files = glob.glob(os.path.join(qid_path, '**', '*'), recursive=True)

    for file_path in sorted(all_files):
        if not os.path.isfile(file_path) or os.path.basename(file_path) == 'result.json':
            continue
            
        file_name = os.path.relpath(file_path, qid_path)
        ext = os.path.splitext(file_name)[1].lower()
        
        if ext in CODE_EXTENSIONS:
            code_files_content += f"\n--- Start of Code File: {file_name} ---\n{read_text_file(file_path)}\n--- End of Code File: {file_name} ---\n"
        elif ext == '.csv':
            try:
                df = pd.read_csv(file_path)
                csv_files_summary += f"\n--- Start of CSV File Descriptive Stats: {file_name} ---\n{df.describe(include='all').to_string()}\n--- End of CSV File Stats: {file_name} ---\n"
            except Exception as e:
                csv_files_summary += f"Could not process CSV file {file_name}. Error: {e}\n"

    generated_files_parts = [f"### File Directory Structure\n{file_structure_info}"]
    if code_files_content: generated_files_parts.append(f"### Generated Code Files (.py)\n{code_files_content}")
    if csv_files_summary: generated_files_parts.append(f"### Generated CSV File Summaries\n{csv_files_summary}")
    files_content_block = "\n\n".join(generated_files_parts)
    
    prompt_text = f"""
As a **highly critical and meticulous AI Quality Engineer**, your task is to evaluate an agent's response with a tough but fair assessment. You are known for your extremely high standards.

**CRITICAL OUTPUT FORMAT:**
Your entire response MUST be a single, valid, flat (non-nested) JSON object. No other text is allowed. All scores must be an **integer from 0 to 5**.

The JSON structure must be:
{{
  "method_soundness": "Your analysis justifying the score for the new combined dimension, based on the rubric.",
  "presentation_quality_analysis": "Your analysis justifying the score based on the rubric.",
  "artifact_generation_analysis": "Your analysis justifying the score based on the rubric.",
  "summary": "Your overall summary of the agent's performance.",
  
  "method_soundness_score": <An integer from 0 to 5>,
  "presentation_quality_score": <An integer from 0 to 5>,
  "artifact_generation_score": <An integer from 0 to 5>,
  "overall_score": <An integer from 0 to 5>
}}

---
**SCORING RUBRIC**
---

**1. method_soundness**

Evaluate the overall quality of the solution path. Assess the chosen method's soundness and correctness. More importantly, **this score heavily weights the quality and completeness of the agent's justification for its approach**. A well-reasoned, comprehensive report that discusses problem framing, results, and limitations should score highly, even if its method intelligently deviates from the reference answer. The reference serves as a benchmark, not a strict mandate.

*   **5 (Exemplary):** **MUST meet all criteria for a score of 4.** In addition, the **Methodology Justification** is exceptionally insightful, comparing the approach to viable alternatives and providing a profound analysis of the **generated results** and limitations.
*   **4 (Strong):** **A score of 4 is IMPOSSIBLE without clear evidence of successful execution** (explicit statements OR output files like `.png`/`.csv` in `{file_structure_info}`). The response must also contain a true **Methodology Justification** (as defined above), not just a code explanation.
*   **3 (Acceptable):** **This is the ABSOLUTE MAXIMUM score for any solution that LACKS execution evidence.** To achieve this score, the response MUST provide a strong **Methodology Justification** (the "Why"). It presents a well-reasoned strategic plan, even if it's not proven with results.
*   **2 (Weak):** **Assign this score if a response's justification consists ONLY of code and a Code Explanation (the "What" and "How"), like the provided example.** This score is the correct rating for a submission that presents a function or script but fails to provide the strategic "Why" behind it. This score also applies if the method is flawed or execution failed.
*   **1 (Poor):** The method is fundamentally flawed or irrelevant, and there is no meaningful justification of any kind.

**2. presentation_quality**

Clarity, structure, and completeness of the final answer's explanation and formatting. Is it easy to read and understand?

*   **5 (Exemplary):** The presentation is exceptionally clear, professional, and well-structured. It uses formatting, language, and structure to make complex information easy to digest.
*   **4 (Strong):** The presentation is clear, well-structured, and complete. It is easy to read and understand with only minor room for refinement.
*   **3 (Acceptable):** The core message is communicated clearly and the structure is adequate. A reader can understand the answer, though it may have minor issues with clarity, organization, or formatting.
*   **2 (Weak):** The presentation is disorganized, unclear, or incomplete, making it difficult for the reader to follow.
*   **1 (Poor):** The presentation is confusing, unstructured, or poorly formatted, failing to communicate the information effectively.

**3. artifact_generation**

Functionality, correctness, and completeness of generated files (e.g., code, plots, data). Are they usable and aligned with the task?

*   **5 (Exemplary):** Artifacts are not only correct and well-organized into files but also demonstrate exemplary software engineering practices. The code architecture is robust and clean (e.g., using functions/classes), well-commented, and easily understandable or reusable.
*   **4 (Strong):** Generates correct and functional artifacts. The code is well-organized, logically structured, and is appropriately saved into distinct files (e.g., `.py` for code, `.csv` for data).
*   **3 (Acceptable):** Artifacts are generated and are largely functional. The code may require minor corrections to run, ript), but it successfully implements the core logic.
*   **2 (Weak):** Artifacts are generated but contain significant errors or are incomplete. This score recognizes the attempt to generate code but notes its flaws.
*   **1 (Poor):** Fails to generate the required artifacts.

**4. overall_score**

Your holistic assessment, **not a simple average**. A critical failure (score of 0 or 1) in one key area should heavily penalize the overall score. Conversely, exceptional performance (4 or 5) in key areas should elevate it.

*   **5 (Exemplary):** An exemplary performance (score of 5) on `solution_approach_justification` with at least strong performance (4) on other dimensions. A model answer.
*   **4 (Strong):** Strong performance (4 or higher) across all key dimensions, or an exemplary performance in one key area balanced by acceptable performance elsewhere. A high-quality submission.
*   **3 (Acceptable):** Meets expectations (score of 3) on all dimensions. A solid, complete solution with no major flaws.
*   **2 (Weak):** Weak performance (score of 2) in one or more dimensions, especially `solution_approach_justification` or `artifact_generation`, but is not a complete failure.
*   **1 (Poor):** Contains a fundamental flaw or critical error (score of 1) in a key dimension, severely compromising the value of the submission.

---
**EVALUATION MATERIALS**
---

### 1. Task Brief
{task_brief}

### 2. Reference Answer
**Note:** This is a reference, not the only correct solution. Focus on semantic correctness and the overall approach.
{reference_answer}

### 3. Agent's Reasoning and Final Answer
{agent_reasoning}

### Agent-Generated Files and Content
{files_content_block}

---
**EVALUATION INSTRUCTIONS**
---

Based on all materials and the rigorous scoring rubric provided above, perform your evaluation. Provide your detailed analysis for each dimension, an overall summary, and the required scores in the specified flat JSON format.
"""
    return prompt_text, None


# --- 3. 评估阶段 (Phase 1) ---
def evaluate_qid_with_single_model(qid_path, llm_name, model_instance, output_dir):
    """用一个指定的LLM评估一个QID，并将结果保存到特定文件。"""
    qid = os.path.basename(qid_path)
    qid_output_dir = os.path.join(output_dir, qid)
    os.makedirs(qid_output_dir, exist_ok=True)
    
    output_file = os.path.join(qid_output_dir, f"{llm_name}.json")
    
    try:
        prompt_text, error = build_evaluation_prompt(qid_path)
        if error:
            return qid, llm_name, "error", f"Prompt creation failed: {error}"

        message = HumanMessage(content=prompt_text)
        response = model_instance.invoke([message])
        llm_output = response.content

        if llm_output.strip().startswith("```json"):
            llm_output = llm_output.strip()[7:-3].strip()
        
        parsed_json = json.loads(llm_output)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(parsed_json, f, indent=4, ensure_ascii=False)
            
        return qid, llm_name, "success", "Evaluation successful."

    except json.JSONDecodeError:
        error_file = os.path.join(qid_output_dir, f"{llm_name}_error.txt")
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write(llm_output)
        return qid, llm_name, "warning", f"Invalid JSON. Raw output saved to {os.path.basename(error_file)}"
    except Exception as e:
        return qid, llm_name, "error", f"API call failed: {e}\n{traceback.format_exc()}"

def run_evaluation_phase(llm_models, qid_folders, output_dir):
    """为每个LLM创建独立的线程池，并同时运行所有评估任务。"""
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
                output_file = os.path.join(output_dir, qid, f"{llm_name}.json")
                if os.path.exists(output_file):
                    continue 

                model_instance = model_info["instance"]
                executor = executors[llm_name]
                future = executor.submit(evaluate_qid_with_single_model, qid_path, llm_name, model_instance, output_dir)
                all_futures.append(future)

        if not all_futures:
            print("All evaluation tasks for the specified QID range have already been completed. Nothing to do.")
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


# --- 4. 聚合阶段 (Phase 2) ---
def run_aggregation_phase(qid_folders, output_dir):
    """聚合所有评估结果，计算平均分。"""
    print("\n--- Starting Aggregation Phase ---")
    
    all_qids_summaries = []
    
    score_keys = [
        "method_soundness_score", 
        "presentation_quality_score", 
        "artifact_generation_score", 
        "overall_score"
    ]
    
    grand_totals = {key: 0 for key in score_keys}
    grand_counts = {key: 0 for key in score_keys}

    for qid_path in tqdm(qid_folders, desc="Aggregating results"):
        qid = os.path.basename(qid_path)
        qid_output_dir = os.path.join(output_dir, qid)
        
        if not os.path.isdir(qid_output_dir):
            continue

        qid_score_sums = {key: 0 for key in score_keys}
        eval_files = glob.glob(os.path.join(qid_output_dir, "*.json"))
        
        successful_evals = 0
        for file_path in eval_files:
            if os.path.basename(file_path) == "_average_scores.json":
                continue
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if all(isinstance(data.get(key), (int, float)) for key in score_keys):
                    for key in score_keys:
                        qid_score_sums[key] += data.get(key, 0)
                    successful_evals += 1
                else:
                    with tqdm_lock:
                        incomplete_keys = [key for key in score_keys if not isinstance(data.get(key), (int, float))]
                        tqdm.write(f"[Warning] Skipping incomplete score file: {file_path}. Missing/invalid keys: {incomplete_keys}")

            except (json.JSONDecodeError, IOError) as e:
                with tqdm_lock:
                    tqdm.write(f"[Warning] Could not process file {file_path}: {e}")
        
        qid_averages = {}
        if successful_evals > 0:
            for key in score_keys:
                avg = qid_score_sums[key] / successful_evals
                qid_averages[f"average_{key}"] = round(avg, 2)
                grand_totals[key] += qid_score_sums[key]
                grand_counts[key] += successful_evals
        
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
    if grand_counts and grand_counts.get(score_keys[0], 0) > 0:
        for key in score_keys:
            final_averages[f"overall_average_{key}"] = round(grand_totals[key] / grand_counts[key], 2)
    
    final_summary = {
        "overall_average_scores": final_averages,
        "total_qids_processed": len(qid_folders),
        "total_successful_evaluations": grand_counts.get(score_keys[0], 0),
        "per_qid_summary": all_qids_summaries
    }
    
    final_summary_path = os.path.join(output_dir, "_final_summary.json")
    with open(final_summary_path, 'w', encoding='utf-8') as f:
        json.dump(final_summary, f, indent=4, ensure_ascii=False)
        
    print("\n--- Aggregation Phase Finished ---")
    print("\nFinal Overall Average Scores:")
    print(json.dumps(final_averages, indent=4))
    print(f"\nDetailed summary saved to: {final_summary_path}")

# --- 5. 主执行流程 ---
def get_qid_folders_to_process(qid_range_str, dataset_path):
    """根据指定的QID范围返回要处理的文件夹路径列表。"""
    all_potential_folders = glob.glob(os.path.join(dataset_path, '*'))

    qid_folders_to_process = []
    for folder_path in all_potential_folders:
        if os.path.isdir(folder_path):
            qid_str = os.path.basename(folder_path)
            if qid_str.isdigit():
                qid_folders_to_process.append(folder_path)

    # 按QID对文件夹进行排序
    qid_folders = sorted(
        qid_folders_to_process,
        key=lambda x: int(os.path.basename(x))
    )

    # 如果没有指定范围，则返回所有文件夹
    if not qid_range_str:
        return qid_folders

    # 根据提供的QID范围进行过滤
    try:
        if ',' in qid_range_str: # 处理列表格式, e.g., '1,4,6,7,8'
            qid_set = set(map(int, qid_range_str.split(',')))
        elif '-' in qid_range_str: # 处理连续范围格式, e.g., '1-5'
            start, end = map(int, qid_range_str.split('-'))
            qid_set = set(range(start, end + 1))
        else: # 处理单个数字格式, e.g., '5'
            qid_set = {int(qid_range_str)}
        
        qid_folders = [folder for folder in qid_folders if int(os.path.basename(folder)) in qid_set]
    except ValueError:
        print(f"[Warning] Invalid QID range format: '{qid_range_str}'. Processing all QIDs.")

    return qid_folders


def main(args):
    """主执行函数"""
    # 确保输出目录存在 (使用传入的参数)
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        llm_configs = load_llm_configs(args.config_file)
        llm_models = initialize_llms(llm_configs)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"[CRITICAL ERROR] {e}")
        return

    # 使用传入的参数检索QID文件夹
    qid_folders = get_qid_folders_to_process(args.qid_range, args.dataset_path)

    if not qid_folders:
        print(f"No QID folders found in '{args.dataset_path}' for the specified range. Exiting.")
        return
    
    print(f"\nFound {len(qid_folders)} QIDs to process: {[os.path.basename(p) for p in qid_folders]}")

    # 运行评估和聚合阶段
    run_evaluation_phase(llm_models, qid_folders, args.output_dir)
    run_aggregation_phase(qid_folders, args.output_dir)
    
    print("\n\nAll processes completed successfully!")

if __name__ == '__main__':
    args = parse_arguments()
    main(args)