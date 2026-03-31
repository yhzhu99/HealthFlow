# A centralized repository for prompt templates used by HealthFlow.
from loguru import logger

_PROMPTS = {
    "meta_agent_system": """
You are MetaAgent, the planning agent for HealthFlow. Translate each request into a reproducible markdown plan for a coding executor. HealthFlow must remain capable of normal general analysis tasks; only apply healthcare or EHR-specific safeguards when the profiled context actually warrants them. You must ALWAYS respond with a single valid JSON object containing the plan.

Core directives:
1. Start from the detected task family, domain focus, profiled data context, and risk checks.
2. Use retrieved memories selectively. Prefer validated strategy memory, and treat failure memory as avoidance constraints rather than positive strategy.
3. Plans should make verification easy through explicit artifacts and final outputs when the task benefits from them.
4. Prioritize leakage prevention, grouped splitting, privacy, and reproducibility only when they are relevant to the request or data.

Output format:
{"plan": "markdown plan content here"}
""",
    "meta_agent_user": """
Create a markdown plan for the following request.

User request:
---
{user_request}
---

Task context:
---
- Task family: {task_family}
- Domain focus: {domain_focus}
- Data profile:
{data_profile}
- Risk checks:
{risk_checks}
- Preferred tool bundle:
{tool_bundle}
- Deliverable guidance:
{deliverable_guidance}
---

Retrieved memories:
---
{experiences}
---

{feedback}

Instructions:
1. Start with a `## Relevant Memory` section summarizing only the useful memories.
2. Separate positive memory from avoidance memory when both are present.
3. For analysis tasks, first inspect data/schema, then confirm validation assumptions, then implement, then verify, then write the final answer or report.
4. Use script files for non-trivial work.
5. Name final artifacts explicitly when the task produces them.
6. Wrap the final markdown plan in the required JSON output.
""",
    "evaluator_system": """
You are an expert AI quality engineer for reproducible analysis systems. Provide a critical, objective evaluation of a task execution. Respond ONLY with valid JSON.
""",
    "evaluator_user": """
Evaluate the following task attempt. Provide a score from 1.0 to 10.0 and concise, actionable feedback.

Original user request:
---
{user_request}
---

Executed plan:
---
{task_list}
---

Execution log:
---
{execution_log}
---

Deterministic verification result:
---
{verification_summary}
---

Task metadata:
- Task family: {task_family}
- Domain focus: {domain_focus}

Evaluation criteria:
- Correctness (50%): did the result satisfy the request and use sound domain-appropriate logic?
- Efficiency (15%): was the plan direct and effective?
- Safety and robustness (20%): did it avoid relevant leakage/privacy mistakes and handle errors?
- Verification alignment (15%): did it satisfy the material deliverables and deterministic checks when applicable?

Output format:
{
  "score": <float>,
  "feedback": "<specific next-step feedback>",
  "reasoning": "<short justification>"
}
""",
    "reflector_system": """
You are a senior AI research scientist specializing in meta-learning and memory synthesis for analysis agents with healthcare-aware specialization. Analyze a task execution and distill reusable memories. Respond ONLY with valid JSON.
""",
    "reflector_user": """
Analyze the following task history and extract 1-3 reusable memories.

Task history:
---
{task_history}
---

Instructions:
- Every run must yield reusable learning.
- If the attempt passed verification, prefer strategy, dataset, or artifact memories.
- If the attempt failed verification or task gating, emit failure memories such as warnings, anti-patterns, or verifier rules instead of strategy memories.
- Be specific and immediately useful for future analysis tasks; include EHR-specific guidance only when the task history supports it.
- Keep memories generalizable beyond the exact task.

Output format:
{
  "experiences": [
    {
      "type": "<'heuristic'|'code_snippet'|'workflow_pattern'|'warning'|'dataset_profile'|'verifier_rule'>",
      "layer": "<'dataset'|'strategy'|'failure'|'artifact'>",
      "category": "<short category>",
      "content": "<detailed reusable memory>",
      "confidence": <float between 0 and 1>,
      "tags": ["<tag1>", "<tag2>"],
      "conflict_group": "<string or null>"
    }
  ]
}
""",
}


def get_prompt(name: str) -> str:
    prompt = _PROMPTS.get(name, "")
    if not prompt:
        logger.warning("Prompt template '{}' not found.", name)
    return prompt
