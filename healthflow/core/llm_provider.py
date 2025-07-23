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
                usage=response.usage.model_dump() if response.usage else None,
                model=response.model,
            )
        except Exception as e:
            # logger.error(f"LLM generation failed: {e}", exc_info=True)
            raise RuntimeError(f"LLM API error: {e}") from e

def create_llm_provider(api_key: str, base_url: str, model_name: str) -> LLMProvider:
    """Factory function to create the appropriate LLM provider."""
    return OpenAICompatibleProvider(api_key, base_url, model_name)