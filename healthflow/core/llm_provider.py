import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel
from typing import List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger

from .config import LLMProviderConfig

class LLMMessage(BaseModel):
    role: str
    content: str

class LLMResponse(BaseModel):
    content: str

class LLMProvider:
    def __init__(self, config: LLMProviderConfig):
        """Initializes the provider with a specific LLM's configuration."""
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
        logger.debug(f"Generating LLM response with model {self.model_name}. JSON mode: {json_mode}")
        try:
            response_format = {"type": "json_object"} if json_mode else {"type": "text"}

            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[msg.model_dump() for msg in messages],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
            content = response.choices[0].message.content or ""
            logger.debug(f"LLM response received. Length: {len(content)}")
            return LLMResponse(content=content)
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise

def create_llm_provider(config: LLMProviderConfig) -> LLMProvider:
    """Factory function to create the LLM provider from its specific config."""
    return LLMProvider(config)