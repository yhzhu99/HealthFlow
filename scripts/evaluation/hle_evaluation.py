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
from healthflow.tools.json_utils import load_jsonl

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MedAgentsBench")
    parser.add_argument("--dataset_dir", type=str, default="healthflow_datasets", help="Dataset directory")
    parser.add_argument("--input_log_dir", type=str, default="logs", help="Input log directory")
    parser.add_argument("--output_log_dir", type=str, default="processed_logs", help="Output log directory")
    parser.add_argument("--dataset_name", type=str, default="hle", help="Dataset name")
    parser.add_argument("--evaluator_model_key", type=str, default="deepseek-v3-official", help="Evaluator model key")
    parser.add_argument("--models_to_evaluate", type=str, nargs="+", default=["ColaCare", "MedAgent", "MDAgents", "SingleLLM_deepseek-r1-official_zero_shot", "SingleLLM_deepseek-v3-official_zero_shot", "SingleLLM_huatuogpt_zero_shot", "SingleLLM_medgemma_zero_shot"], help="Models to evaluate")
    return parser.parse_args()

def get_hle_prompt(question: str, generated_answer: str, reference_answer: str) -> tuple[str, str]:
    """
    Generate System and User Prompt for evaluating the HLE dataset.
    """
    system_prompt = (
        "You are a Large Language Model Answer Evaluator. Your task is to assess responses on a benchmark "
        "known as 'HLE'. This benchmark is specifically designed to evaluate the complex medical "
        "reasoning capabilities of Large Language Models (LLMs)."
    )

    user_prompt = f"""Judge whether the following [response] to [question] is correct or not
based on the precise and unambiguous [correct_answer] below.
[question]: {question}
[response]: {generated_answer}
[correct_answer]: {reference_answer}
Your judgement must be in the JSON format and criteria specified below:
extracted_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.
reasoning: Explain why the extracted_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful
differences between [correct_answer] and the extracted_answer. Do not comment on any background to the problem, do not attempt to solve
the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.
correct: Answer 'yes' if extracted_answer matches the [correct_answer] given above, or is within a small margin of error for
numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.
"""
    return system_prompt, user_prompt

def extract_answer_with_regex(answer: str) -> str | None:
    """
    Use regular expressions to quickly extract single letter options.
    Support two main formats:
    1. Answer at the beginning, such as: "A", "A.", "A)", " A "...
    2. Answer in \\boxed{}, such as: "... The final answer is \\boxed{A}"
    """
    if not isinstance(answer, str):
        return None

    # Format 1: Check if the answer is at the beginning of the string
    # Use re.match to match the beginning part of the string.
    match_start = re.match(r"^\s*\(?([A-Z])[\s\.\)]*", answer.strip(), re.IGNORECASE)
    if match_start:
        return match_start.group(1).upper()

    # Format 2: If there's no match at the beginning, search for \boxed{} format in the entire string
    # Use re.search to match any position in the string.
    # Need to escape special characters in LaTeX commands: \ -> \\, { -> \{, } -> \}
    match_boxed = re.search(r"\\boxed\{\s*([A-Z])\s*\}", answer, re.IGNORECASE)
    if match_boxed:
        return match_boxed.group(1).upper()

    # If neither format matches, return None
    return None

def extract_answer_with_llm(
        question: str,
        generated_answer: str,
        reference_answer: str,
        model_key: str,
        max_retries: int = 3
    ) -> dict | None:
    """
    Use LLM to extract answers when regular expression matching fails.
    """
    system_prompt, user_prompt = get_hle_prompt(question, generated_answer, reference_answer)
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

            # Verify if the returned JSON contains the required fields
            extracted = parsed_json.get("extracted_answer")
            if isinstance(extracted, str):
                parsed_json["extracted_answer"] = extracted
                return parsed_json
            else:
                logging.warning(f"LLM response has valid JSON keys but invalid 'extracted_answer' content: '{extracted}'")

        except json.JSONDecodeError:
            logging.error(f"Failed to decode LLM response into JSON. Response: {response_content}")
        except Exception as e:
            logging.error(f"An error occurred during LLM call (attempt {attempt + 1}/{max_retries}): {e}")

        time.sleep(1) # Wait before retrying

    logging.error(f"Failed to get a valid response from LLM after {max_retries} retries for answer: {generated_answer}")
    return None

def process_all_results(config: dict):
    """
    Iterate through all models and result files, extract answers and save processed results.
    """
    for model_name in config["models_to_evaluate"]:
        input_dir = os.path.join(config["input_log_dir"], config["dataset_name"], model_name)
        output_dir = os.path.join(config["output_log_dir"], config["dataset_name"], model_name)

        if not os.path.exists(input_dir):
            logging.warning(f"Input directory not found for model {model_name}, skipping: {input_dir}")
            continue

        os.makedirs(output_dir, exist_ok=True)

        data_items = load_jsonl(os.path.join(config["dataset_dir"], config["dataset_name"] + ".jsonl"))
        input_logs = list(sorted(os.listdir(input_dir), key=lambda x: int(x.split("-")[0])))

        logging.info(f"--- Processing model: {model_name} ---")
        for item, file in tqdm(zip(data_items, input_logs), desc=f"Processing {model_name}"):
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, file)

            if os.path.exists(output_path):
                # logging.info(f"Skipping existing file: {output_path}")
                continue

            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            question = data.get("question") or data.get("task")
            generated_answer = data.get("generated_answer")
            reference_answer = data.get("reference_answer")

            # 1. Try to extract using regular expressions
            task_type = item.get("type")
            if task_type == "mc":
                extracted_answer = extract_answer_with_regex(generated_answer)
            else:
                extracted_answer = None

            # 2. If regex fails, use LLM
            if extracted_answer is None:
                logging.info(f"Regex failed for {file}, using LLM for extraction...")
                llm_result = extract_answer_with_llm(question, generated_answer, reference_answer, config["evaluator_model_key"])

                if llm_result:
                    final_data = llm_result
                    final_data["question"] = question
                    final_data["generated_answer"] = generated_answer
                    final_data["reference_answer"] = reference_answer
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
    Calculate the accuracy of each model from processed logs and save the results to a CSV file.
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
                logging.info(f"Skipping failed extraction for {file}")
                continue

            if extracted == reference:
                correct_predictions += 1

        if total_samples > 0:
            # Subtract failed extraction samples to calculate effective accuracy
            valid_total = total_samples - failed_extractions
            accuracy = (correct_predictions / valid_total) * 100 if valid_total > 0 else 0

            # Create a result dictionary for the current model
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

    # Define the path for the output CSV file
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