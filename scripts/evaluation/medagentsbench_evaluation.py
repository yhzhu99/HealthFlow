import os
import re
import json
import time
import logging
import argparse

from openai import OpenAI
from tqdm import tqdm
import pandas as pd

from healthflow.tools.llm_configs import LLM_MODELS_SETTINGS

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MedAgentsBench")
    parser.add_argument("--input_log_dir", type=str, default="logs", help="Input log directory")
    parser.add_argument("--output_log_dir", type=str, default="processed_logs", help="Output log directory")
    parser.add_argument("--dataset_name", type=str, default="medagentsbench", help="Dataset name")
    parser.add_argument("--evaluator_model_key", type=str, default="deepseek-chat-official", help="Evaluator model key")
    parser.add_argument("--models_to_evaluate", type=str, nargs="+", default=["ColaCare", "MedAgent", "MDAgents", "SingleLLM_deepseek-reasoner-official_zero_shot", "SingleLLM_deepseek-chat-official_zero_shot", "SingleLLM_huatuogpt_zero_shot", "SingleLLM_medgemma_zero_shot"], help="Models to evaluate")
    return parser.parse_args()

def get_medagentsbench_prompt(generated_answer: str, reference_answer: str) -> tuple[str, str]:
    """
    Generate System and User Prompts for evaluating MedAgentsBench dataset.
    """
    system_prompt = (
        "You are a Large Language Model Answer Evaluator. Your task is to assess responses on a benchmark "
        "known as 'MedAgentsBench'. This benchmark is specifically designed to evaluate the complex medical "
        "reasoning capabilities of Large Language Models (LLMs). It moves beyond basic medical knowledge, "
        "testing models with challenging scenarios that require multi-step clinical reasoning, diagnosis "
        "formulation, and treatment planning. The questions are sourced from seven established medical "
        "datasets and are filtered to ensure they are challenging, meaning most current models struggle "
        "to answer them correctly."
    )

    user_prompt = f"""# Task
Your core mission is to accurately extract the single multiple-choice option (e.g., A, B, C, D) from the model's generated answer (`generated_answer`). The format of the model's response may vary—it could contain only the letter, or the letter followed by the full text of the option. Your job is to precisely identify and isolate the single letter that represents the final choice.

# Input
```json
{{
"generated_answer": {json.dumps(generated_answer)},
"reference_answer": "{reference_answer}"
}}
```
# Output Format
You must process the input and return the result in a strict JSON format. The output JSON must contain the following three fields: reference_answer, generated_answer, and extracted_answer.

# Output
"""
    return system_prompt, user_prompt

def extract_answer_with_regex(answer: str) -> str | None:
    """
    Use regular expressions to quickly extract single letter options.
    Supports two main formats:
    1. Answer at the beginning, like: "A", "A.", "A)", " A "...
    2. Answer in \\boxed{}, like: "... The final answer is \\boxed{A}"
    """
    if not isinstance(answer, str):
        return None

    # Format 1: Check if answer is at the beginning of string
    # Use re.match to match the beginning of the string.
    match_start = re.match(r"^\s*\(?([A-Z])[\s\.\)]*", answer.strip(), re.IGNORECASE)
    if match_start:
        return match_start.group(1).upper()

    # Format 2: If no match at beginning, search for \boxed{} format in entire string
    # Use re.search to match any position in the string.
    # Need to escape special characters in LaTeX commands: \ -> \\, { -> \{, } -> \}
    match_boxed = re.search(r"\\boxed\{\s*([A-Z])\s*\}", answer, re.IGNORECASE)
    if match_boxed:
        return match_boxed.group(1).upper()

    # If neither format matches, return None
    return None

def extract_answer_with_llm(
        generated_answer: str,
        reference_answer: str,
        model_key: str,
        max_retries: int = 3
    ) -> dict | None:
    """
    When regex matching fails, use LLM to extract answers.
    """
    system_prompt, user_prompt = get_medagentsbench_prompt(generated_answer, reference_answer)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    model_settings = LLM_MODELS_SETTINGS[model_key]
    client = OpenAI(api_key=model_settings["api_key"], base_url=model_settings["base_url"])

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_settings["model_name"],
                messages=messages,
                stream=False,
                response_format={"type": "json_object"}, # Force JSON output
                temperature=0.0
            )

            response_content = response.choices[0].message.content
            # Parse the JSON string returned by LLM
            parsed_json = json.loads(response_content)

            # Verify if the returned JSON contains required fields
            if all(k in parsed_json for k in ["reference_answer", "generated_answer", "extracted_answer"]):
                extracted = parsed_json.get("extracted_answer")
                if isinstance(extracted, str) and re.fullmatch(r"[A-Z]", str(extracted).strip(), re.IGNORECASE):
                    # Ensure the extracted is also a single letter
                    parsed_json["extracted_answer"] = str(extracted).strip().upper()
                    return parsed_json
                else:
                    logging.warning(f"LLM response has valid JSON keys but invalid 'extracted_answer' content: '{extracted}'")
            else:
                logging.warning(f"LLM response missing required keys: {response_content}")

        except json.JSONDecodeError:
            logging.error(f"Failed to decode LLM response into JSON. Response: {response_content}")
        except Exception as e:
            logging.error(f"An error occurred during LLM call (attempt {attempt + 1}/{max_retries}): {e}")

        time.sleep(1) # Wait before retrying

    logging.error(f"Failed to get a valid response from LLM after {max_retries} retries for answer: {generated_answer}")
    return None

def process_all_results(config: dict):
    """
    Traverse all models and result files, extract answers and save processed results.
    """
    for model_name in config["models_to_evaluate"]:
        input_dir = os.path.join(config["input_log_dir"], config["dataset_name"], model_name)
        output_dir = os.path.join(config["output_log_dir"], config["dataset_name"], model_name)

        if not os.path.exists(input_dir):
            logging.warning(f"Input directory not found for model {model_name}, skipping: {input_dir}")
            continue

        os.makedirs(output_dir, exist_ok=True)
        files = os.listdir(input_dir)

        logging.info(f"--- Processing model: {model_name} ---")
        for file in tqdm(files, desc=f"Processing {model_name}"):
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, file)

            if os.path.exists(output_path):
                logging.info(f"Skipping existing file: {output_path}")
                continue

            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            generated_answer = data.get("generated_answer")
            reference_answer = data.get("reference_answer")

            # 1. Try using regular expressions to extract
            if isinstance(generated_answer, str):
                extracted_answer = extract_answer_with_regex(generated_answer)
            elif isinstance(generated_answer, list):
                extracted_answer = "".join(generated_answer)
            else:
                extracted_answer = None

            # 2. If regex fails, use LLM
            if extracted_answer is None:
                continue
                logging.info(f"Regex failed for {file}, using LLM for extraction...")
                llm_result = extract_answer_with_llm(generated_answer, reference_answer, config["evaluator_model_key"])

                if llm_result:
                    final_data = llm_result
                else:
                    # If LLM also fails, log error and skip
                    final_data = data.copy()
                    final_data["extracted_answer"] = "EXTRACT_FAILED" # Mark as failed
            else:
                # Regex successful, build unified format
                final_data = data.copy()
                final_data["extracted_answer"] = extracted_answer

            # 3. Save processed file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, indent=4)

def calculate_accuracy(config: dict):
    """
    Calculate accuracy for each model from processed logs and save results to CSV file.
    """
    logging.info("\n--- Calculating Final Accuracies and Saving to CSV ---")

    results_list = []

    for model_name in config["models_to_evaluate"]:
        processed_dir = os.path.join(config["output_log_dir"], config["dataset_name"], model_name)

        if not os.path.exists(processed_dir):
            logging.warning(f"Processed directory for model {model_name} not found, skipping.")
            continue

        files = os.listdir(processed_dir)
        total_samples = 0
        correct_predictions = 0
        failed_extractions = 0

        for file in files:
            with open(os.path.join(processed_dir, file), 'r', encoding='utf-8') as f:
                data = json.load(f)

            total_samples += 1
            extracted = data.get("extracted_answer")
            reference = data.get("reference_answer")

            if extracted == "EXTRACT_FAILED":
                failed_extractions += 1
                continue

            if extracted == reference:
                correct_predictions += 1

        if total_samples > 0:
            # Subtract failed extraction samples to calculate effective accuracy
            valid_total = total_samples - failed_extractions
            accuracy = (correct_predictions / valid_total) * 100 if valid_total > 0 else 0

            # Create a result dictionary for current model
            model_summary = {
                "Model": model_name,
                "Accuracy (%)": accuracy,
                "Correct": correct_predictions,
                "Valid Total": valid_total,
                "Failed Extractions": failed_extractions,
                "Total Samples": total_samples
            }
            results_list.append(model_summary)
        else:
            logging.info(f"No samples found for model {model_name}.")

    # Check if there are results to save
    if not results_list:
        logging.warning("No results to save to CSV.")
        return

    # Use pandas to create DataFrame and save to CSV
    df = pd.DataFrame(results_list)

    # Sort DataFrame by accuracy in descending order
    df = df.sort_values(by="Accuracy (%)", ascending=False).reset_index(drop=True)

    # Define output CSV file path
    output_csv_path = os.path.join(config["output_log_dir"], config["dataset_name"], "evaluation_summary.csv")

    try:
        df.to_csv(output_csv_path, index=False, float_format='%.2f')
        logging.info(f"\nEvaluation summary successfully saved to: {output_csv_path}")
        logging.info("--- Evaluation Summary ---")
        logging.info("\n" + df.to_string())
    except Exception as e:
        logging.error(f"Failed to save summary to CSV: {e}")

if __name__ == "__main__":
    args = parse_args()
    config = vars(args)
    process_all_results(config)
    calculate_accuracy(config)