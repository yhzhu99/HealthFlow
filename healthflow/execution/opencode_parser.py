from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class OpenCodeParsedRun:
    log: str
    usage: dict[str, Any]
    telemetry: dict[str, Any]


def parse_opencode_json_events(stdout: str) -> OpenCodeParsedRun:
    log_lines: list[str] = []
    raw_lines: list[str] = []
    steps: list[dict[str, Any]] = []
    tool_names: set[str] = set()
    step_reasons: dict[str, int] = {}
    session_id: str | None = None
    current_step: dict[str, Any] | None = None

    totals = {
        "estimated_cost_usd": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "tool_call_count": 0,
        "tool_time_seconds": 0.0,
        "model_time_seconds": 0.0,
        "step_count": 0,
    }
    parsed_event_count = 0

    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            raw_lines.append(line)
            continue

        parsed_event_count += 1
        event_type = event.get("type")
        timestamp_ms = _to_int(event.get("timestamp"))
        part = event.get("part")
        if not isinstance(part, dict):
            part = {}
        session_id = (
            session_id
            or _to_str(event.get("sessionID"))
            or _to_str(part.get("sessionID"))
        )

        if event_type == "step_start":
            current_step = {
                "step_index": len(steps) + 1,
                "start_timestamp_ms": timestamp_ms,
                "reason": None,
                "estimated_cost_usd": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "reasoning_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "tool_call_count": 0,
                "tool_time_seconds": 0.0,
                "model_time_seconds": 0.0,
                "duration_seconds": 0.0,
                "text_chars": 0,
                "tool_names": [],
            }
            steps.append(current_step)
            totals["step_count"] += 1
            log_lines.append(f"EVENT: step_start #{current_step['step_index']}")
            continue

        if event_type == "text":
            text = _to_str(part.get("text"), default="")
            if current_step is not None:
                current_step["text_chars"] += len(text)
            _append_prefixed_text(log_lines, "STDOUT: ", text)
            continue

        if event_type == "tool_use":
            tool_name = _to_str(part.get("tool"), default="unknown")
            tool_names.add(tool_name)
            totals["tool_call_count"] += 1
            state = part.get("state")
            if not isinstance(state, dict):
                state = {}
            duration_seconds = _time_range_seconds(state.get("time"))
            totals["tool_time_seconds"] += duration_seconds
            if current_step is not None:
                current_step["tool_call_count"] += 1
                current_step["tool_time_seconds"] += duration_seconds
                if tool_name not in current_step["tool_names"]:
                    current_step["tool_names"].append(tool_name)

            status = _to_str(state.get("status"), default="unknown")
            input_preview = _render_preview(state.get("input"))
            output_preview = _render_preview(state.get("output"))
            log_lines.append(f"TOOL[{tool_name}] status={status}")
            if input_preview:
                log_lines.append(f"TOOL[{tool_name}] input={input_preview}")
            if output_preview:
                log_lines.append(f"TOOL[{tool_name}] output={output_preview}")
            continue

        if event_type == "step_finish":
            step = current_step
            if step is None:
                step = {
                    "step_index": len(steps) + 1,
                    "start_timestamp_ms": timestamp_ms,
                    "reason": None,
                    "estimated_cost_usd": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "reasoning_tokens": 0,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "tool_call_count": 0,
                    "tool_time_seconds": 0.0,
                    "model_time_seconds": 0.0,
                    "duration_seconds": 0.0,
                    "text_chars": 0,
                    "tool_names": [],
                }
                steps.append(step)
                totals["step_count"] += 1

            tokens = part.get("tokens")
            if not isinstance(tokens, dict):
                tokens = {}
            cache = tokens.get("cache")
            if not isinstance(cache, dict):
                cache = {}
            step["reason"] = _to_str(part.get("reason"))
            step["estimated_cost_usd"] = _to_float(part.get("cost"))
            step["input_tokens"] = _to_int(tokens.get("input"))
            step["output_tokens"] = _to_int(tokens.get("output"))
            step["total_tokens"] = _to_int(tokens.get("total"))
            step["reasoning_tokens"] = _to_int(tokens.get("reasoning"))
            step["cache_read_tokens"] = _to_int(cache.get("read"))
            step["cache_write_tokens"] = _to_int(cache.get("write"))
            step["duration_seconds"] = _elapsed_seconds(step.get("start_timestamp_ms"), timestamp_ms)
            step["model_time_seconds"] = max(step["duration_seconds"] - step["tool_time_seconds"], 0.0)

            totals["estimated_cost_usd"] += step["estimated_cost_usd"]
            totals["input_tokens"] += step["input_tokens"]
            totals["output_tokens"] += step["output_tokens"]
            totals["total_tokens"] += step["total_tokens"]
            totals["reasoning_tokens"] += step["reasoning_tokens"]
            totals["cache_read_tokens"] += step["cache_read_tokens"]
            totals["cache_write_tokens"] += step["cache_write_tokens"]
            totals["model_time_seconds"] += step["model_time_seconds"]

            if step["reason"]:
                step_reasons[step["reason"]] = step_reasons.get(step["reason"], 0) + 1
            log_lines.append(
                "EVENT: step_finish "
                f"#{step['step_index']} reason={step['reason'] or 'unknown'} "
                f"cost=${step['estimated_cost_usd']:.6f} "
                f"tokens(in={step['input_tokens']}, out={step['output_tokens']}, total={step['total_tokens']}, "
                f"cache_read={step['cache_read_tokens']}, cache_write={step['cache_write_tokens']})"
            )
            current_step = None
            continue

        log_lines.append(f"EVENT: {event_type}")

    if raw_lines:
        if log_lines:
            log_lines.append("EVENT: unparsed_stdout")
        _append_prefixed_text(log_lines, "STDOUT: ", "\n".join(raw_lines))

    rounded_steps = [_round_step(step) for step in steps]
    usage = {
        "estimated_cost_usd": round(totals["estimated_cost_usd"], 8),
        "input_tokens": totals["input_tokens"],
        "output_tokens": totals["output_tokens"],
        "total_tokens": totals["total_tokens"],
        "reasoning_tokens": totals["reasoning_tokens"],
        "cache_read_tokens": totals["cache_read_tokens"],
        "cache_write_tokens": totals["cache_write_tokens"],
        "tool_call_count": totals["tool_call_count"],
        "tool_time_seconds": round(totals["tool_time_seconds"], 4),
        "model_time_seconds": round(totals["model_time_seconds"], 4),
        "step_count": totals["step_count"],
    }
    telemetry = {
        "session_id": session_id,
        "models": [],
        "event_count": parsed_event_count,
        "parse_error_count": len(raw_lines),
        "step_count": totals["step_count"],
        "step_reasons": step_reasons,
        "tool_names": sorted(tool_names),
        "steps": rounded_steps,
    }
    return OpenCodeParsedRun(
        log="\n".join(log_lines).strip(),
        usage=usage,
        telemetry=telemetry,
    )


def _round_step(step: dict[str, Any]) -> dict[str, Any]:
    rounded = dict(step)
    rounded.pop("start_timestamp_ms", None)
    rounded["estimated_cost_usd"] = round(float(rounded.get("estimated_cost_usd", 0.0)), 8)
    for key in ["tool_time_seconds", "model_time_seconds", "duration_seconds"]:
        rounded[key] = round(float(rounded.get(key, 0.0)), 4)
    return rounded


def _append_prefixed_text(lines: list[str], prefix: str, content: str) -> None:
    if not content:
        return
    for raw_line in content.splitlines():
        lines.append(f"{prefix}{raw_line}")


def _render_preview(value: Any, max_chars: int = 240) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, str):
        rendered = value.strip()
    else:
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if len(rendered) > max_chars:
        return rendered[: max_chars - 16] + "... [truncated]"
    return rendered


def _time_range_seconds(value: Any) -> float:
    if not isinstance(value, dict):
        return 0.0
    return _elapsed_seconds(value.get("start"), value.get("end"))


def _elapsed_seconds(start_ms: Any, end_ms: Any) -> float:
    start_value = _to_int(start_ms)
    end_value = _to_int(end_ms)
    if start_value <= 0 or end_value <= 0 or end_value < start_value:
        return 0.0
    return round((end_value - start_value) / 1000.0, 4)


def _to_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_str(value: Any, default: str | None = None) -> str | None:
    if value is None:
        return default
    return str(value)
