"""
LLM Provider Abstraction and Implementations
Support for multiple LLM providers: OpenAI, Anthropic, Google Gemini
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncIterator
from dataclasses import dataclass
import json

import openai
import httpx
from openai import OpenAI, AsyncOpenAI


@dataclass
class LLMMessage:
    """Represents a message in LLM conversation"""
    role: str  # "system", "user", "assistant"
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    """Response from LLM provider"""
    content: str
    usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None
    
    
class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, api_key: str, base_url: str, model_name: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
    
    @abstractmethod
    async def generate(
        self,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response from messages"""
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate streaming response from messages"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1", model_name: str = "gpt-4-turbo-preview"):
        super().__init__(api_key, base_url, model_name)
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    
    async def generate(
        self,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using OpenAI API"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[msg.to_dict() for msg in messages],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                usage=response.usage.dict() if response.usage else None,
                model=response.model
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")
    
    async def generate_stream(
        self,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate streaming response using OpenAI API"""
        try:
            stream = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[msg.to_dict() for msg in messages],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                **kwargs
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise RuntimeError(f"OpenAI streaming API error: {str(e)}")


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.anthropic.com", model_name: str = "claude-3-opus-20240229"):
        super().__init__(api_key, base_url, model_name)
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }
    
    def _convert_messages(self, messages: List[LLMMessage]) -> Dict[str, Any]:
        """Convert messages to Anthropic format"""
        system_msg = ""
        converted_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_msg += msg.content + "\n"
            else:
                converted_messages.append({"role": msg.role, "content": msg.content})
        
        payload = {"messages": converted_messages}
        if system_msg:
            payload["system"] = system_msg.strip()
            
        return payload
    
    async def generate(
        self,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Anthropic API"""
        try:
            async with httpx.AsyncClient() as client:
                payload = self._convert_messages(messages)
                payload.update({
                    "model": self.model_name,
                    "max_tokens": max_tokens or 4096,
                    "temperature": temperature or 0.7,
                    **kwargs
                })
                
                response = await client.post(
                    f"{self.base_url}/v1/messages",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                
                data = response.json()
                return LLMResponse(
                    content=data["content"][0]["text"],
                    usage=data.get("usage"),
                    model=data.get("model")
                )
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {str(e)}")
    
    async def generate_stream(
        self,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate streaming response using Anthropic API"""
        try:
            async with httpx.AsyncClient() as client:
                payload = self._convert_messages(messages)
                payload.update({
                    "model": self.model_name,
                    "max_tokens": max_tokens or 4096,
                    "temperature": temperature or 0.7,
                    "stream": True,
                    **kwargs
                })
                
                async with client.stream(
                    "POST",
                    f"{self.base_url}/v1/messages",
                    headers=self.headers,
                    json=payload
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                if data["type"] == "content_block_delta":
                                    yield data["delta"]["text"]
                            except (json.JSONDecodeError, KeyError):
                                continue
        except Exception as e:
            raise RuntimeError(f"Anthropic streaming API error: {str(e)}")


class GeminiProvider(LLMProvider):
    """Google Gemini API provider"""
    
    def __init__(self, api_key: str, base_url: str = "https://generativelanguage.googleapis.com/v1beta", model_name: str = "gemini-pro"):
        super().__init__(api_key, base_url, model_name)
        self.headers = {"Content-Type": "application/json"}
    
    def _convert_messages(self, messages: List[LLMMessage]) -> Dict[str, Any]:
        """Convert messages to Gemini format"""
        contents = []
        for msg in messages:
            if msg.role == "system":
                contents.append({"role": "user", "parts": [{"text": f"System: {msg.content}"}]})
            else:
                role = "model" if msg.role == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": msg.content}]})
        
        return {"contents": contents}
    
    async def generate(
        self,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Gemini API"""
        try:
            async with httpx.AsyncClient() as client:
                payload = self._convert_messages(messages)
                
                generation_config = {}
                if max_tokens:
                    generation_config["maxOutputTokens"] = max_tokens
                if temperature is not None:
                    generation_config["temperature"] = temperature
                    
                if generation_config:
                    payload["generationConfig"] = generation_config
                
                url = f"{self.base_url}/models/{self.model_name}:generateContent?key={self.api_key}"
                response = await client.post(url, headers=self.headers, json=payload)
                response.raise_for_status()
                
                data = response.json()
                content = data["candidates"][0]["content"]["parts"][0]["text"]
                
                return LLMResponse(
                    content=content,
                    usage=data.get("usageMetadata"),
                    model=self.model_name
                )
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {str(e)}")
    
    async def generate_stream(
        self,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate streaming response using Gemini API"""
        try:
            async with httpx.AsyncClient() as client:
                payload = self._convert_messages(messages)
                
                generation_config = {}
                if max_tokens:
                    generation_config["maxOutputTokens"] = max_tokens
                if temperature is not None:
                    generation_config["temperature"] = temperature
                    
                if generation_config:
                    payload["generationConfig"] = generation_config
                
                url = f"{self.base_url}/models/{self.model_name}:streamGenerateContent?key={self.api_key}"
                
                async with client.stream("POST", url, headers=self.headers, json=payload) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                data = json.loads(line)
                                if "candidates" in data:
                                    content = data["candidates"][0]["content"]["parts"][0]["text"]
                                    yield content
                            except (json.JSONDecodeError, KeyError):
                                continue
        except Exception as e:
            raise RuntimeError(f"Gemini streaming API error: {str(e)}")


def create_llm_provider(provider_type: str, api_key: str, base_url: str, model_name: str) -> LLMProvider:
    """Factory function to create LLM provider"""
    provider_type = provider_type.lower()
    
    if provider_type == "openai" or "openai" in base_url:
        return OpenAIProvider(api_key, base_url, model_name)
    elif provider_type == "anthropic" or "anthropic" in base_url:
        return AnthropicProvider(api_key, base_url, model_name)
    elif provider_type == "gemini" or "generativelanguage" in base_url:
        return GeminiProvider(api_key, base_url, model_name)
    else:
        # Default to OpenAI format for compatibility
        return OpenAIProvider(api_key, base_url, model_name)