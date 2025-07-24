"""
LLM Provider Abstraction to support various OpenAI-compatible APIs.
This is a simplified version for HealthFlow's needs.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, Field


class LLMMessage(BaseModel):
    role: str
    content: str

class LLMResponse(BaseModel):
    content: str
    usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None

class LLMProvider(ABC):
    @abstractmethod
    async def generate(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        timeout: int = 120,
    ) -> LLMResponse:
        pass

class OpenAICompatibleProvider(LLMProvider):
    """A provider for any OpenAI-compatible API."""

    def __init__(self, api_key: str, base_url: str, model_name: str):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.AsyncClient(
                proxies=None,
                transport=httpx.AsyncHTTPTransport(local_address="0.0.0.0"),
            ),
        )
        self.model_name = model_name

    async def generate(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        timeout: int = 120,
    ) -> LLMResponse:
        """Generates a response from the LLM."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[msg.model_dump() for msg in messages],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            )
            return LLMResponse(
                content=response.choices[0].message.content or "",
                usage=self._safe_usage_extract(response.usage) if response.usage else None,
                model=response.model,
            )
        except Exception as e:
            # logger.error(f"LLM generation failed: {e}", exc_info=True)
            raise RuntimeError(f"LLM API error: {e}") from e

    def _safe_usage_extract(self, usage) -> Dict[str, int]:
        """Safely extract usage information handling different API response formats."""
        try:
            if hasattr(usage, 'model_dump'):
                usage_dict = usage.model_dump()
            else:
                usage_dict = dict(usage)
            
            # Clean up the usage dict to ensure all values are integers
            safe_usage = {}
            for key, value in usage_dict.items():
                if key.endswith('_details') or value is None:
                    continue  # Skip detail fields and None values
                try:
                    safe_usage[key] = int(value) if value is not None else 0
                except (ValueError, TypeError):
                    safe_usage[key] = 0
            
            return safe_usage
        except Exception:
            # Return minimal safe usage if extraction fails
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

def create_llm_provider(api_key: str, base_url: str, model_name: str) -> LLMProvider:
    """Factory function to create the appropriate LLM provider."""
    return OpenAICompatibleProvider(api_key, base_url, model_name)