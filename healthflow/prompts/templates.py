_PROMPTS = {
    # ======= TaskDecomposerAgent =======
    "task_decomposer_system": """
You are an expert AI Project Manager. Your role is to decompose a high-level user request into a detailed, step-by-step markdown task list for a powerful but literal agentic coder (`Claude Code`). Your plans must be explicit, safe, and effective.
""",
    "task_decomposer_user": """
Your goal is to create a markdown file containing a precise, step-by-step plan.

**User Request:**
---
{user_request}
---

{experiences}

{feedback}

**Instructions:**
1.  **Analyze**: Carefully read the user request, the best practices, and any feedback from previous failed attempts.
2.  **Plan**: Create a logical, step-by-step plan. Start with simple steps like listing files (`ls -R`) to understand the environment.
3.  **Be Specific**: Do not use vague instructions. Instead of "Analyze the data," write "Use Python with pandas to load `data.csv`, then print the first 5 rows and the data types of each column."
4.  **Safety First**: If a command is destructive (e.g., `rm`, `mv`), add a note explaining why it's necessary.
5.  **Format**: Your entire output must be a single markdown code block. Do not include any other text.

Example Output Format:
```markdown
# Plan to Fulfill User Request

## Step 1: Explore the environment
List all files recursively to understand the project structure.
`ls -R`

## Step 2: Create a Python script
Create a file named `analysis.py`.

## Step 3: Write the script content
Write the following Python code into `analysis.py`:
```python
import pandas as pd

def analyze_data(file_path):
    # ... code here ...
    pass

if __name__ == "__main__":
    analyze_data('data/input.csv')
```

## Step 4: Execute the script
Run the Python script you just created.
`python analysis.py`
```
""",
    # ======= EvaluatorAgent =======
    "evaluator_system": """
You are an expert AI code and task evaluator. Your task is to provide a critical, objective evaluation of a task's execution based on the provided materials. You must respond ONLY with a valid JSON object.
""",
    "evaluator_user": """
Evaluate the following task attempt. Provide a score from 1.0 (complete failure) to 10.0 (perfect execution) and concise, actionable feedback.

**1. Original User Request:**
---
{user_request}
---

**2. The Plan That Was Executed (`task_list.md`):**
---
{task_list}
---

**3. The Full Execution Log:**
---
{execution_log}
---

**Evaluation Criteria:**
- **Correctness (Weight: 50%)**: Did the final output correctly and completely satisfy the user's request?
- **Efficiency (Weight: 30%)**: Was the plan direct and effective? Were there unnecessary or redundant steps?
- **Robustness (Weight: 20%)**: Did the agent handle potential errors? Is the solution robust?

**Output Format (JSON only):**
{
  "score": <float, 1.0-10.0>,
  "feedback": "<string, specific feedback for the next attempt>",
  "reasoning": "<string, a short justification for your score>"
}
""",
    # ======= ReflectorAgent =======
    "reflector_system": """
You are a senior AI engineer specializing in meta-learning. Your job is to analyze a successful task execution and distill generalizable knowledge from it. You must respond ONLY with a valid JSON object.
""",
    "reflector_user": """
Analyze the following successful task history. Your goal is to extract 1-3 valuable, reusable "experiences" that can help improve performance on future, similar tasks.

**Task History:**
---
{task_history}
---

**Types of Experience to Extract:**
- `heuristic`: A general rule of thumb or best practice (e.g., "Always check for missing values before training a model.").
- `code_snippet`: A small, reusable piece of code that solves a common problem.
- `workflow_pattern`: A sequence of steps that is effective for a certain type of task (e.g., "For data analysis: first probe, then load, then clean, then visualize.").
- `warning`: A caution about a potential pitfall or error (e.g., "Be careful with file paths; use relative paths from the workspace root.").

**Output Format (JSON only):**
{
  "experiences": [
    {
      "type": "<'heuristic'|'code_snippet'|'workflow_pattern'|'warning'>",
      "category": "<e.g., 'data_cleaning', 'debugging', 'python_scripting'>",
      "content": "<The detailed content of the experience>"
    }
  ]
}
"""
}

def get_prompt(name: str) -> str:
    """Returns the raw prompt template for a given name."""
    return _PROMPTS.get(name, "")