import os
import json
import glob
import argparse
import concurrent.futures
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from tqdm import tqdm

REFINEMENT_PROMPT_TEMPLATE = """
# ROLE
You are a meticulous and articulate Principal AI Research Scientist specializing in computational biology and healthcare analytics. Your task is to transform the AI agent’s execution-oriented output into a polished, professional, and peer-review-ready technical report.

# GUIDING PRINCIPLES — READ THIS FIRST
This is the most critical instruction. Your primary goal is to **accurately represent the agent’s actual accomplishments**.

1. **Identify the True Goal:** First, analyze the `[TASK_DESCRIPTION]` and structure your evaluation around the intended task objective.

2. **Code as the Primary Deliverable:** For tasks where no readable dataset is provided, **the generated code may be the main deliverable**. If the code fully implements the required functionality, it should be considered the primary result; otherwise, you must identify its deficiencies and explain their impact.

3. **Do Not Fabricate Content or Omit Issues:** You **must not fabricate code snippets or conceal problems**. The report must **faithfully reflect the actual execution demonstrated in `[GENERATED_CODE_FILES]` and `[AGENT_LOGS]`**, including any errors, warnings, or missing functionality.


# CONTEXT
An AI agent was assigned a biomedical or healthcare-related task. The agent has completed the task by planning, writing, and executing code, producing artifacts such as scripts, logs, and data files. Your job is to synthesize all available materials into a final report that accurately conveys the background, methodology, and results—strictly following the principles above.

# INSTRUCTIONS
Based on the materials provided below, you must generate a comprehensive and detailed final report of at least 1000 words. The report must be written in a formal, narrative style, using well-structured paragraphs that flow logically from one to the next, and minimizing the use of bullet points or itemized lists unless strictly necessary. Your elaboration should focus on providing deeper context, explaining the technical and scientific rationale in detail, and expanding on the significance and potential applications of the work.

### **Structure of the Final Report**

**1. Executive Summary:**
- Briefly summarize the background and technical approach.
- Provide a clear, one-sentence statement of the primary achievement.

**2. Problem Statement & Objectives:**
- Restate the core problem from the `[TASK_DESCRIPTION]`.
- Clearly define the task objectives, especially whether the main goal was the implementation of a function or algorithm.

**3. Methodology & Implementation:**
- **Technical Approach:** Describe the overall strategy the agent followed.
- **Implementation Details & Final Code:**
  - **Present the final functional code from `[GENERATED_CODE_FILES]` in a prominent code block.**
  - Briefly explain how key parts of the code fulfill the problem objectives.
  - If applicable, comment on any scientific or clinical rationale behind design choices.

- **Verification & Validation:**
  - Explain how, or if, the code’s correctness was verified (e.g., through log-based tests, synthetic input simulations, boundary condition testing).
  - If any errors or potential issues were found, explain their potential consequences.

**4. Results & Analysis:**
- **Execution Overview:** State whether the agent's task attempt was successfully completed and whether the expected files were generated: [LIST_OF_GENERATED_FILES].
- **Key Outputs:**
  - **If the primary output is code:** Explicitly evaluate its level of completion based on whether it correctly implements the task logic.
  - **If data or numerical results were generated:** Present and analyze them for correctness and representativeness.

- **Qualitative Analysis:** Comment on the code quality—e.g., is it readable, standards-compliant, well-documented, and robust?

**5. Conclusion & Future Work:**
- **Task Completeness:** Assess whether the agent successfully completed the task.
- **Summary of Achievements:** Summarize the accomplished parts of the task and any shortcomings.
- **Future Work:** Propose recommendations for future work based on the task's completion status.

---
## EVALUATION MATERIALS
---

### 1. Task Description
```
[TASK_DESCRIPTION]
```

[GENERATED_ANSWER_SECTION]
### [LOGS_SECTION_NUM]. Agent Execution Logs (STDOUT/STDERR)
```
[AGENT_LOGS]
```

### [CODE_SECTION_NUM]. Generated Code Files
```python
[GENERATED_CODE_FILES]
```

### [FILES_SECTION_NUM]. List of Generated Files in Workspace
```
[LIST_OF_GENERATED_FILES]
```
"""

def get_llm(model_name: str):
    """Initializes and returns the ChatOpenAI model instance."""
    load_dotenv()
    API_KEY = os.getenv("OPENAI_API_KEY")
    API_BASE = os.getenv("OPENAI_API_BASE")

    if not API_KEY or not API_BASE:
        raise ValueError(
            "Please ensure OPENAI_API_KEY and OPENAI_API_BASE environment variables are set."
        )

    return ChatOpenAI(
        model_name=model_name,
        openai_api_key=API_KEY,
        openai_api_base=API_BASE,
        temperature=0.1,
        max_tokens=4096,
    )

def read_file_content(file_path, max_chars=10000):
    """Safely reads file content."""
    if not os.path.exists(file_path):
        return f"File not found: {os.path.basename(file_path)}"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                return f"File is empty: {os.path.basename(file_path)}"
            if len(content) > max_chars:
                return content[:max_chars] + "\n... [Content truncated] ..."
            return content
    except Exception as e:
        return f"Error reading file {os.path.basename(file_path)}: {e}"

def find_python_files(directory):
    """Finds all .py files and returns their content and a list of their names."""
    py_files = glob.glob(os.path.join(directory, '*.py'))
    script_name = os.path.basename(__file__)
    py_files = [f for f in py_files if os.path.basename(f) != script_name]

    if not py_files:
        return "No Python scripts found for execution.", []

    all_code, filenames = "", []
    for py_file in py_files:
        filename = os.path.basename(py_file)
        filenames.append(filename)
        code_content = read_file_content(py_file)
        all_code += f"\n--- Code File: {filename} ---\n```python\n{code_content}\n```\n"
    return all_code, filenames

def list_generated_files(directory, known_files):
    """Lists all files in the directory, excluding known source/log files."""
    try:
        all_items = os.listdir(directory)
        generated = [
            item for item in all_items
            if os.path.isfile(os.path.join(directory, item)) and item not in known_files
        ]
        if not generated:
            return "No new files were generated during execution."
        return "\n".join(f"- {f}" for f in generated)
    except FileNotFoundError:
        return "Workspace directory not found."

def process_qid_folder(qid_path, llm):
    """
    Processes a single qid folder: gathers all artifacts, populates the prompt template,
    invokes the LLM, and updates result.json.
    """
    qid = os.path.basename(qid_path)

    result_json_path = os.path.join(qid_path, 'result.json')
    result_data = {}
    task_description = "Task description not found in result.json."
    reference_answer = "Reference answer not found in result.json."
    generated_answer = None
    try:
        with open(result_json_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if content:
                result_data = json.loads(content)
                task_description = result_data.get("task", task_description)
                reference_answer = result_data.get("reference_answer", reference_answer)
                generated_answer = result_data.get("generated_answer")
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    generated_answer_section = ""
    logs_num, code_num, files_num = 2, 3, 4

    if generated_answer and "no answer" not in str(generated_answer).lower() and "no result" not in str(generated_answer).lower():
        formatted_answer = json.dumps({'generated_answer': generated_answer}, indent=2, ensure_ascii=False)
        generated_answer_section = f"### 2. Generated Answer\n```json\n{formatted_answer}\n```"
        logs_num, code_num, files_num = 3, 4, 5

    python_code_content, python_filenames = find_python_files(qid_path)

    agent_logs = f"""
### Execution Log (execution.log)
{read_file_content(os.path.join(qid_path, 'execution.log'))}
"""

    known_source_files = set(['result.json', 'stdout.txt', 'stderr.txt', 'execution.log'] + python_filenames)
    plan_files = glob.glob(os.path.join(qid_path, 'task_list_v*.md'))
    for pf in plan_files:
        known_source_files.add(os.path.basename(pf))
    generated_files_list = list_generated_files(qid_path, known_source_files)

    final_prompt = REFINEMENT_PROMPT_TEMPLATE.replace(
        '[TASK_DESCRIPTION]', task_description
    ).replace(
        '[GENERATED_ANSWER_SECTION]', generated_answer_section
    ).replace(
        '[LOGS_SECTION_NUM]', str(logs_num)
    ).replace(
        '[CODE_SECTION_NUM]', str(code_num)
    ).replace(
        '[FILES_SECTION_NUM]', str(files_num)
    ).replace(
        '[AGENT_LOGS]', agent_logs
    ).replace(
        '[GENERATED_CODE_FILES]', python_code_content
    ).replace(
        '[LIST_OF_GENERATED_FILES]', generated_files_list
    )

    try:
        response = llm.invoke([HumanMessage(content=final_prompt)])
        summary = response.content

        result_data["final_answer"] = summary
        with open(result_json_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=4)

        return f"Successfully generated report for QID: {qid}"
    except Exception as e:
        raise Exception(f"Failed to process {qid}: {e}")


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Batch process QID folders to generate final reports using an LLM.")

    parser.add_argument("--root_dir", type=str, required=True,
                        help="The root directory containing the QID subfolders.")

    parser.add_argument("--qid_start", type=int, default=1,
                        help="The starting QID number to process.")

    parser.add_argument("--qid_end", type=int, default=100,
                        help="The ending QID number to process.")

    parser.add_argument("--model_name", type=str, default="deepseek-chat",
                        help="The name of the ChatOpenAI model to use.")

    parser.add_argument("--max_workers", type=int, default=20,
                        help="The maximum number of concurrent threads for processing.")

    return parser.parse_args()


def main():
    """
    Main function to process a range of QID folders concurrently,
    with breakpoint continuation support, configured via command-line arguments.
    """
    # --- 1. 解析命令行参数 ---
    args = parse_arguments()

    print(f"--- Starting Batch Processing ---")
    print(f"Root Directory: {args.root_dir}")
    print(f"Target QID Range: {args.qid_start} to {args.qid_end}")
    print(f"Model: {args.model_name}")
    print(f"Max Workers: {args.max_workers}")

    # --- 2. 查找所有潜在的任务文件夹 ---
    all_qids = [str(i) for i in range(args.qid_start, args.qid_end + 1)]
    all_target_paths = []
    for qid in all_qids:
        path = os.path.join(args.root_dir, qid)
        if os.path.isdir(path):
            all_target_paths.append(path)
        else:
            print(f"Warning: Directory for QID {qid} not found at '{path}', skipping.")

    # --- 3. 检查需要完成的任务 ---
    print("\n--- Checking for Previously Completed Tasks ---")
    tasks_to_process = []
    skipped_count = 0

    for path in tqdm(all_target_paths, desc="Scanning QIDs"):
        result_json_path = os.path.join(path, 'result.json')
        is_completed = False
        if os.path.exists(result_json_path):
            try:
                with open(result_json_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():
                        data = json.loads(content)
                        if data.get("final_answer"):
                            is_completed = True
            except (json.JSONDecodeError, IOError):
                is_completed = False

        if is_completed:
            skipped_count += 1
        else:
            tasks_to_process.append(path)

    print(f"\nScan complete. Total tasks found: {len(all_target_paths)}")
    print(f"Skipping {skipped_count} already completed tasks.")
    print(f"Tasks to process now: {len(tasks_to_process)}")
    print("-" * 33)

    if not tasks_to_process:
        print("All tasks in the specified range are already completed. Exiting.")
        return

    # --- 4. 初始化LLM ---
    try:
        llm = get_llm(args.model_name)
    except ValueError as e:
        print(f"Initialization Error: {e}")
        return

    # --- 5. 并发处理剩余任务 ---
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_qid = {
            executor.submit(process_qid_folder, path, llm): os.path.basename(path)
            for path in tasks_to_process
        }

        for future in tqdm(concurrent.futures.as_completed(future_to_qid), total=len(tasks_to_process), desc="Processing QIDs"):
            qid = future_to_qid[future]
            try:
                _ = future.result()
            except Exception as exc:
                print(f'\nError processing QID {qid}: {exc}')

    print("\nBatch report generation complete!")

if __name__ == "__main__":
    main()
