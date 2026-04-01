from __future__ import annotations

SYSTEM_PROMPT = "You are a data scientist."

TASK_TYPE_DATA_EXTRACTION = "data_extraction"
TASK_TYPE_PREDICTIVE_MODELING = "predictive_modeling"
TASK_TYPE_VISUALIZATION = "visualization"
TASK_TYPE_REPORT_GENERATION = "report_generation"

TASK_TYPES = (
    TASK_TYPE_DATA_EXTRACTION,
    TASK_TYPE_PREDICTIVE_MODELING,
    TASK_TYPE_VISUALIZATION,
    TASK_TYPE_REPORT_GENERATION,
)

PROMPT_VARIANTS: dict[str, tuple[dict[str, str], ...]] = {
    TASK_TYPE_DATA_EXTRACTION: (
        {
            "name": "data_wrangling",
            "instruction": (
                "The task requires testing data extraction abilities through cleaning, reshaping, "
                "or patient-level / admission-level wrangling."
            ),
        },
        {
            "name": "data_querying",
            "instruction": (
                "The task requires testing data extraction abilities through dataset querying, "
                "filtering, and cohort selection."
            ),
        },
        {
            "name": "data_statistics",
            "instruction": (
                "The task requires testing data extraction abilities through grouped summary "
                "statistics or descriptive analysis."
            ),
        },
        {
            "name": "data_preprocessing",
            "instruction": (
                "The task requires testing data extraction abilities through preprocessing, "
                "feature construction, and table preparation."
            ),
        },
    ),
    TASK_TYPE_PREDICTIVE_MODELING: (
        {
            "name": "modeling",
            "instruction": (
                "The task requires testing predictive modeling ability with a reproducible target, "
                "feature set, and evaluation output."
            ),
        },
    ),
    TASK_TYPE_VISUALIZATION: (
        {
            "name": "plotting_or_visualization_DATA_DIRECT",
            "instruction": (
                "The task requires testing data visualization ability directly from the tabular EHR "
                "data, such as trend plots, cohort comparisons, or summary charts."
            ),
        },
        {
            "name": "plotting_or_visualization_MODEL_ANALYSIS",
            "instruction": (
                "The task requires testing visualization ability for model analysis, such as "
                "performance curves, feature-importance plots, or prediction diagnostics."
            ),
        },
    ),
    TASK_TYPE_REPORT_GENERATION: (
        {
            "name": "report",
            "instruction": (
                "The task requires testing report-generation ability by investigating a concrete "
                "clinical or data-quality question and summarizing the findings."
            ),
        },
    ),
}

PROMPT_TEMPLATE = """
You have been given a tabular EHR dataset, and you need to provide a task based on the dataset.

The dataset examples are as follows:
{data_examples}

You need to provide the response strictly according to the following requirements:

1. {instruction}
2. The task should be concrete, reproducible, and grounded in the provided columns.
3. Please return your response in JSON format, similar to the following:
{{
    "task": "...",
    "task_brief": "..."
}}
""".strip()


def get_variant_definition(task_type: str, variant_name: str) -> dict[str, str]:
    for variant in PROMPT_VARIANTS[task_type]:
        if variant["name"] == variant_name:
            return variant
    raise KeyError(f"unknown prompt variant: task_type={task_type} variant={variant_name}")


def build_prompt(
    *,
    task_type: str,
    variant_name: str,
    data_examples: str,
    exclusion_tasks: list[str],
) -> str:
    variant = get_variant_definition(task_type, variant_name)
    prompt = PROMPT_TEMPLATE.format(
        data_examples=data_examples,
        instruction=variant["instruction"],
    )
    if exclusion_tasks:
        prompt += (
            "\n\nThe task is not allowed to be similar in content or syntax to any of the "
            f"following tasks:\n{exclusion_tasks}"
        )
    return prompt
