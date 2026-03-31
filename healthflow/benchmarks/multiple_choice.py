from __future__ import annotations

import re
from typing import Any


MC_START_RE = re.compile(r"^\s*\(?([A-Z])[\s\.\):,-]*", re.IGNORECASE)
MC_BOXED_RE = re.compile(r"\\boxed\{\s*([A-Z])\s*\}", re.IGNORECASE)
MC_FINAL_RE = re.compile(
    r"(?:final answer|answer|option|choice|select(?:ed)?|prediction)\s*(?:is|:)?\s*\(?([A-Z])\)?\b",
    re.IGNORECASE,
)


def normalize_generated_text(value: Any) -> str:
    if isinstance(value, list):
        return "".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


def extract_multiple_choice_answer(value: Any) -> str | None:
    text = normalize_generated_text(value).strip()
    if not text:
        return None

    for pattern in (MC_START_RE, MC_BOXED_RE, MC_FINAL_RE):
        match = pattern.search(text)
        if match:
            return match.group(1).upper()

    standalone = re.findall(r"\b([A-Z])\b", text.upper())
    if len(standalone) == 1:
        return standalone[0]
    return None
