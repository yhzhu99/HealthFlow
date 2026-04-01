import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from openai import NOT_GIVEN

from healthflow.core.config import LLMProviderConfig
from healthflow.core.llm_provider import LLMMessage, LLMProvider, LLMResponse, parse_json_content


class _StubStructuredProvider(LLMProvider):
    def __init__(self, responses: list[LLMResponse]):
        super().__init__(
            LLMProviderConfig(
                api_key="key",
                base_url="https://example.com/v1",
                model_name="test-model",
            )
        )
        self._responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    async def generate(self, messages, temperature=0.2, max_tokens=4096, json_mode=False):
        self.calls.append([message.model_dump() for message in messages])
        return self._responses.pop(0)


class JSONParsingTests(unittest.TestCase):
    def test_parse_json_content_accepts_fenced_json(self):
        parsed = parse_json_content("```json\n{\"status\": \"ok\"}\n```")
        self.assertEqual(parsed, {"status": "ok"})

    def test_parse_json_content_extracts_json_from_wrapped_text(self):
        parsed = parse_json_content("Here is the result:\n{\"status\": \"ok\", \"score\": 9}\nThanks.")
        self.assertEqual(parsed, {"status": "ok", "score": 9})


class StructuredGenerationTests(unittest.IsolatedAsyncioTestCase):
    async def test_generate_structured_retries_after_malformed_response(self):
        provider = _StubStructuredProvider(
            [
                LLMResponse(content="not json", model_name="test-model"),
                LLMResponse(content='{"status":"ok"}', model_name="test-model"),
            ]
        )

        parsed, response = await provider.generate_structured(
            messages=[],
            parser=parse_json_content,
            temperature=0.0,
            max_attempts=2,
        )

        self.assertEqual(parsed, {"status": "ok"})
        self.assertEqual(response.model_name, "test-model")
        self.assertEqual(len(provider.calls), 2)


class ProviderGenerateTests(unittest.IsolatedAsyncioTestCase):
    async def test_generate_passes_reasoning_effort_when_configured(self):
        mock_create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))],
                usage=None,
            )
        )
        mock_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=mock_create)))

        with patch("healthflow.core.llm_provider.AsyncOpenAI", return_value=mock_client):
            provider = LLMProvider(
                LLMProviderConfig(
                    api_key="key",
                    base_url="https://example.com/v1",
                    model_name="test-model",
                    reasoning_effort="high",
                )
            )
            await provider.generate([LLMMessage(role="user", content="hi")])

        self.assertEqual(mock_create.await_count, 1)
        self.assertEqual(mock_create.await_args.kwargs["reasoning_effort"], "high")

    async def test_generate_retries_without_reasoning_effort_when_provider_rejects_it(self):
        mock_create = AsyncMock(
            side_effect=[
                ValueError("Unsupported parameter: reasoning_effort"),
                SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))],
                    usage=None,
                ),
            ]
        )
        mock_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=mock_create)))

        with patch("healthflow.core.llm_provider.AsyncOpenAI", return_value=mock_client):
            provider = LLMProvider(
                LLMProviderConfig(
                    api_key="key",
                    base_url="https://example.com/v1",
                    model_name="test-model",
                    reasoning_effort="high",
                )
            )
            response = await provider.generate([LLMMessage(role="user", content="hi")])

        self.assertEqual(response.content, "hello")
        self.assertEqual(mock_create.await_count, 2)
        self.assertEqual(mock_create.await_args_list[0].kwargs["reasoning_effort"], "high")
        self.assertIs(mock_create.await_args_list[1].kwargs["reasoning_effort"], NOT_GIVEN)


if __name__ == "__main__":
    unittest.main()
