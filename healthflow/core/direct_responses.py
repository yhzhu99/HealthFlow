from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, Field

from .llm_provider import LLMMessage, LLMProvider, parse_json_content


_WORD_RE = re.compile(r"[a-z0-9']+")
_PATHISH_RE = re.compile(r"[/\\`]|[A-Za-z0-9_]+\.(?:py|ipynb|csv|tsv|json|jsonl|md|txt|sql|yaml|yml|toml)\b")

_DISQUALIFYING_TOKENS = {
    "analyze",
    "analysis",
    "build",
    "bug",
    "code",
    "config",
    "create",
    "csv",
    "data",
    "dataset",
    "debug",
    "edit",
    "error",
    "evaluate",
    "file",
    "files",
    "fix",
    "implement",
    "inspect",
    "json",
    "model",
    "plot",
    "readme",
    "report",
    "run",
    "summarize",
    "summary",
    "test",
    "train",
    "workspace",
    "write",
}

_DIRECT_RESPONSE_ANSWERS = {
    "identity": (
        "Hi! I'm HealthFlow, an AI assistant that can help with analysis, coding, "
        "and structured task execution in this workspace."
    ),
    "capabilities": (
        "I'm HealthFlow. I can help inspect code, reason about tasks, summarize findings, "
        "and work through analysis or implementation requests in this workspace."
    ),
    "thanks": "You're welcome. I'm here if you need anything else.",
    "goodbye": "Bye. Reach out again if you need help.",
    "greeting": "Hi! I'm HealthFlow. How can I help?",
}

_ROUTER_SYSTEM_PROMPT = """
You are a lightweight intent router for HealthFlow.

Decide whether the user request should use HealthFlow's built-in direct-response path instead of the full planning and execution loop.

Only set respond_directly=true for lightweight conversational prompts that require no workspace inspection, no code changes, no file operations, and no task execution.

Allowed categories:
- identity: the user is asking who you are, what you are, or what your name is.
- capabilities: the user is asking what you can do.
- greeting: greetings or salutations.
- thanks: short acknowledgements or thanks.
- goodbye: short closings.
- none: anything task-oriented, ambiguous, or not covered above.

Identity and capability questions must refer to the public assistant identity as HealthFlow, not to any internal planner, evaluator, reflector, or executor backend.

Return only a single JSON object with this schema:
{
  "respond_directly": true or false,
  "category": "none" | "identity" | "capabilities" | "greeting" | "thanks" | "goodbye",
  "reason": "<brief classification reason>"
}

If you are unsure, choose respond_directly=false and category="none".
""".strip()


class DirectResponseDecision(BaseModel):
    respond_directly: bool = Field(default=False)
    category: Literal["none", "identity", "capabilities", "greeting", "thanks", "goodbye"] = "none"
    reason: str = ""


@dataclass(frozen=True)
class DirectResponse:
    mode: str
    category: str
    answer: str
    reason: str
    usage: dict[str, Any] = field(default_factory=dict)
    model_name: str | None = None
    estimated_cost_usd: float | None = None


class DirectResponseRouter:
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    async def maybe_build_direct_response(
        self,
        user_request: str,
        has_uploaded_files: bool = False,
    ) -> DirectResponse | None:
        if not _is_direct_response_candidate(user_request, has_uploaded_files=has_uploaded_files):
            return None

        try:
            decision, response = await self.llm_provider.generate_structured(
                [
                    LLMMessage(role="system", content=_ROUTER_SYSTEM_PROMPT),
                    LLMMessage(
                        role="user",
                        content=(
                            "Classify the following user request for HealthFlow's direct-response path.\n\n"
                            f"User request:\n---\n{user_request}\n---"
                        ),
                    ),
                ],
                lambda content: DirectResponseDecision(**parse_json_content(content)),
                temperature=0.0,
                max_tokens=200,
            )
        except Exception as exc:
            logger.warning("Direct-response routing failed; falling back to orchestration: {}", exc)
            return None

        if not decision.respond_directly or decision.category == "none":
            return None

        answer = _DIRECT_RESPONSE_ANSWERS.get(decision.category)
        if not answer:
            logger.warning(
                "Direct-response router returned unsupported category '{}' for request: {}",
                decision.category,
                user_request,
            )
            return None

        reason = decision.reason.strip() or f"Classified as a lightweight {decision.category} prompt."
        return DirectResponse(
            mode="direct_response",
            category=decision.category,
            answer=answer,
            reason=reason,
            usage=response.usage,
            model_name=response.model_name,
            estimated_cost_usd=response.estimated_cost_usd,
        )


def _is_direct_response_candidate(user_request: str, *, has_uploaded_files: bool) -> bool:
    if has_uploaded_files:
        return False

    normalized = _normalize(user_request)
    tokens = set(_WORD_RE.findall(normalized))
    if not tokens:
        return False
    if len(tokens) > 16 or len(normalized) > 120:
        return False
    if _PATHISH_RE.search(user_request):
        return False
    if tokens.intersection(_DISQUALIFYING_TOKENS):
        return False
    return True


def _normalize(user_request: str) -> str:
    lowered = user_request.lower().strip()
    return re.sub(r"\s+", " ", lowered)
