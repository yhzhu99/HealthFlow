import json
from typing import Any, Callable, List, TypeVar

import httpx
from loguru import logger
from openai import NOT_GIVEN, AsyncOpenAI
from pydantic import BaseModel, Field
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential

from .config import LLMProviderConfig

T = TypeVar("T")

class LLMMessage(BaseModel):
    """Represents a single message in a chat conversation."""
    role: str
    content: str

class LLMResponse(BaseModel):
    """Represents a response from the LLM."""
    content: str
    model_name: str
    usage: dict[str, Any] = Field(default_factory=dict)
    estimated_cost_usd: float | None = None


class StructuredResponseError(ValueError):
    """Raised when a structured response cannot be parsed after retries."""

    def __init__(self, message: str, response: LLMResponse | None = None):
        super().__init__(message)
        self.response = response


def parse_json_content(content: str) -> Any:
    """Best-effort JSON extraction for providers that wrap JSON in prose or fences."""
    stripped = content.strip()
    if not stripped:
        raise ValueError("Response content was empty.")

    decoder = json.JSONDecoder()
    candidates = [stripped]

    fenced_chunks = []
    fence = "```"
    start = 0
    while True:
        open_index = stripped.find(fence, start)
        if open_index == -1:
            break
        content_start = stripped.find("\n", open_index)
        if content_start == -1:
            break
        close_index = stripped.find(fence, content_start + 1)
        if close_index == -1:
            break
        fenced = stripped[content_start + 1 : close_index].strip()
        if fenced:
            fenced_chunks.append(fenced)
        start = close_index + len(fence)
    candidates.extend(fenced_chunks)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        for index, char in enumerate(candidate):
            if char not in "{[":
                continue
            try:
                parsed, _ = decoder.raw_decode(candidate[index:])
                return parsed
            except json.JSONDecodeError:
                continue

    raise ValueError("Response did not contain a valid JSON object or array.")

class LLMProvider:
    """
    A wrapper around an OpenAI-compatible LLM client.
    Handles API calls with automatic retries for transient errors.
    """
    def __init__(self, config: LLMProviderConfig):
        """Initializes the provider with a specific LLM's configuration."""
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            http_client=httpx.AsyncClient(timeout=config.timeout),
        )
        self.model_name = config.model_name
        logger.info(f"LLMProvider initialized for model: {self.model_name} at {config.base_url}")

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> LLMResponse:
        """
        Generates a chat completion from the LLM.

        Args:
            messages: A list of LLMMessage objects representing the conversation history.
            temperature: The creativity of the response.
            max_tokens: The maximum number of tokens to generate.
            json_mode: Whether to enable JSON mode for the response.

        Returns:
            An LLMResponse object with the generated content.
        """
        logger.debug(f"Generating LLM response with model {self.model_name}. JSON mode: {json_mode}")
        try:
            # Set response format based on json_mode flag
            response_format = {"type": "json_object"} if json_mode else {"type": "text"}
            request_kwargs = self._completion_request_kwargs(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )

            try:
                completion = await self.client.chat.completions.create(**request_kwargs)
            except Exception as exc:
                if not self._should_retry_without_reasoning_effort(exc):
                    raise
                logger.warning(
                    "Provider rejected reasoning_effort='{}' for model '{}'; retrying without it. Error: {}",
                    self.config.reasoning_effort,
                    self.model_name,
                    exc,
                )
                request_kwargs["reasoning_effort"] = NOT_GIVEN
                completion = await self.client.chat.completions.create(**request_kwargs)
            content = completion.choices[0].message.content or ""
            usage = self._normalize_usage(getattr(completion, "usage", None))
            estimated_cost_usd = self._estimate_cost_usd(usage)
            logger.debug(f"LLM response received. Length: {len(content)}")
            return LLMResponse(
                content=content,
                model_name=self.model_name,
                usage=usage,
                estimated_cost_usd=estimated_cost_usd,
            )
        except RetryError as e:
            logger.error(f"LLM API call failed after multiple retries: {e}")
            raise # Re-raise the exception after logging
        except Exception as e:
            logger.error(f"An unexpected error occurred during LLM API call: {e}")
            # This will trigger a retry if tenacity is configured for it
            raise

    def _completion_request_kwargs(
        self,
        *,
        messages: List[LLMMessage],
        temperature: float,
        max_tokens: int,
        response_format: dict[str, str],
    ) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "messages": [msg.model_dump() for msg in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
            "reasoning_effort": self.config.reasoning_effort or NOT_GIVEN,
        }

    def _should_retry_without_reasoning_effort(self, exc: Exception) -> bool:
        if not self.config.reasoning_effort:
            return False

        message = str(exc).lower()
        if "reasoning_effort" not in message and "reasoning effort" not in message:
            return False

        unsupported_markers = (
            "unsupported",
            "unknown",
            "unrecognized",
            "unexpected",
            "extra_forbidden",
            "extra inputs are not permitted",
            "invalid",
            "not allowed",
            "not supported",
        )
        return any(marker in message for marker in unsupported_markers)

    async def generate_structured(
        self,
        messages: List[LLMMessage],
        parser: Callable[[str], T],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_attempts: int = 3,
    ) -> tuple[T, LLMResponse]:
        """Generate a structured response and retry locally when parsing fails."""
        repair_messages = list(messages)
        last_error: Exception | None = None
        last_response: LLMResponse | None = None

        for attempt in range(1, max_attempts + 1):
            response = await self.generate(
                repair_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=True,
            )
            last_response = response
            try:
                return parser(response.content), response
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Structured response parse failed for model '{}' on local attempt {}/{}: {}",
                    self.model_name,
                    attempt,
                    max_attempts,
                    exc,
                )
                if attempt >= max_attempts:
                    break
                repair_messages = list(messages)
                if response.content.strip():
                    repair_messages.extend(
                        [
                            LLMMessage(role="assistant", content=response.content),
                            LLMMessage(
                                role="user",
                                content=(
                                    "Your previous reply was not valid for the requested schema. "
                                    "Return only a single valid JSON object matching the required output. "
                                    "Do not add markdown fences, commentary, or trailing text."
                                ),
                            ),
                        ]
                    )

        raise StructuredResponseError(
            f"Failed to parse a structured response after {max_attempts} attempts: {last_error}",
            response=last_response,
        ) from last_error

    def _normalize_usage(self, usage: Any) -> dict[str, Any]:
        if usage is None:
            return {}
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        if not isinstance(usage, dict):
            return {}

        input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
        output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
        total_tokens = usage.get("total_tokens")
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens

        normalized = {
            "input_tokens": int(input_tokens) if input_tokens is not None else None,
            "output_tokens": int(output_tokens) if output_tokens is not None else None,
            "total_tokens": int(total_tokens) if total_tokens is not None else None,
        }
        return {key: value for key, value in normalized.items() if value is not None}

    def _estimate_cost_usd(self, usage: dict[str, Any]) -> float | None:
        if not usage:
            return None
        if self.config.input_cost_per_million_tokens is None and self.config.output_cost_per_million_tokens is None:
            return None

        input_tokens = float(usage.get("input_tokens", 0))
        output_tokens = float(usage.get("output_tokens", 0))
        input_cost = (
            input_tokens / 1_000_000.0 * self.config.input_cost_per_million_tokens
            if self.config.input_cost_per_million_tokens is not None
            else 0.0
        )
        output_cost = (
            output_tokens / 1_000_000.0 * self.config.output_cost_per_million_tokens
            if self.config.output_cost_per_million_tokens is not None
            else 0.0
        )
        return round(input_cost + output_cost, 8)

def create_llm_provider(config: LLMProviderConfig) -> LLMProvider:
    """Factory function to create the LLM provider from its specific config."""
    return LLMProvider(config)
