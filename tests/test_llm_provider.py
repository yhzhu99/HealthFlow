import unittest

from healthflow.core.config import LLMProviderConfig
from healthflow.core.llm_provider import LLMProvider, LLMResponse, parse_json_content


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


if __name__ == "__main__":
    unittest.main()
