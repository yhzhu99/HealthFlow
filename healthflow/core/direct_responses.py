from __future__ import annotations

import re
from dataclasses import dataclass


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

_GREETING_TOKENS = {"hello", "hey", "hi"}
_THANKS_TOKENS = {"thanks", "thank", "thx"}
_GOODBYE_TOKENS = {"bye", "goodbye", "farewell"}


@dataclass(frozen=True)
class DirectResponse:
    mode: str
    category: str
    answer: str
    reason: str


def maybe_build_direct_response(user_request: str, has_uploaded_files: bool = False) -> DirectResponse | None:
    if has_uploaded_files:
        return None

    normalized = _normalize(user_request)
    tokens = set(_WORD_RE.findall(normalized))
    if not tokens:
        return None
    if len(tokens) > 16 or len(normalized) > 120:
        return None
    if _PATHISH_RE.search(user_request):
        return None
    if tokens.intersection(_DISQUALIFYING_TOKENS):
        return None

    if _matches_identity(tokens, normalized):
        return DirectResponse(
            mode="direct_response",
            category="identity",
            answer=(
                "Hi! I'm HealthFlow, an AI assistant that can help with analysis, coding, "
                "and structured task execution in this workspace."
            ),
            reason="Matched a lightweight identity-style prompt.",
        )
    if _matches_capabilities(tokens, normalized):
        return DirectResponse(
            mode="direct_response",
            category="capabilities",
            answer=(
                "I'm HealthFlow. I can help inspect code, reason about tasks, summarize findings, "
                "and work through analysis or implementation requests in this workspace."
            ),
            reason="Matched a lightweight capabilities-style prompt.",
        )
    if tokens.intersection(_THANKS_TOKENS):
        return DirectResponse(
            mode="direct_response",
            category="thanks",
            answer="You're welcome. I'm here if you need anything else.",
            reason="Matched a lightweight acknowledgement prompt.",
        )
    if tokens.intersection(_GOODBYE_TOKENS):
        return DirectResponse(
            mode="direct_response",
            category="goodbye",
            answer="Bye. Reach out again if you need help.",
            reason="Matched a lightweight closing prompt.",
        )
    if tokens.intersection(_GREETING_TOKENS):
        return DirectResponse(
            mode="direct_response",
            category="greeting",
            answer="Hi! I'm HealthFlow. How can I help?",
            reason="Matched a lightweight greeting prompt.",
        )
    return None


def _normalize(user_request: str) -> str:
    lowered = user_request.lower().strip()
    return re.sub(r"\s+", " ", lowered)


def _matches_identity(tokens: set[str], normalized: str) -> bool:
    return (
        "who are you" in normalized
        or "what are you" in normalized
        or ({"who", "you"} <= tokens and "are" in tokens)
        or ({"what", "you"} <= tokens and "are" in tokens)
    )


def _matches_capabilities(tokens: set[str], normalized: str) -> bool:
    return "what can you do" in normalized or {"what", "can", "you", "do"} <= tokens
