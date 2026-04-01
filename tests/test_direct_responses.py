import json
import unittest

from healthflow.core.direct_responses import DirectResponseRouter
from healthflow.core.llm_provider import LLMResponse


class _StubStructuredProvider:
    def __init__(self, payload: dict[str, object]):
        self.payload = payload
        self.calls: list[list[dict[str, str]]] = []

    async def generate_structured(self, messages, parser, temperature=0.0, max_tokens=4096, max_attempts=3):
        self.calls.append([message.model_dump() for message in messages])
        content = json.dumps(self.payload)
        return (
            parser(content),
            LLMResponse(
                content=content,
                model_name="router-test-model",
                usage={"input_tokens": 12, "output_tokens": 4, "total_tokens": 16},
                estimated_cost_usd=0.0001,
            ),
        )


class DirectResponseRouterTests(unittest.IsolatedAsyncioTestCase):
    async def test_routes_name_question_to_healthflow_identity(self):
        provider = _StubStructuredProvider(
            {
                "respond_directly": True,
                "category": "identity",
                "reason": "The user is asking for the assistant's name.",
            }
        )
        router = DirectResponseRouter(provider)

        response = await router.maybe_build_direct_response("what's your name?")

        self.assertIsNotNone(response)
        self.assertEqual(response.category, "identity")
        self.assertIn("HealthFlow", response.answer)
        self.assertEqual(response.usage["total_tokens"], 16)
        self.assertEqual(len(provider.calls), 1)

    async def test_routes_identity_paraphrase_without_exact_phrase_matching(self):
        provider = _StubStructuredProvider(
            {
                "respond_directly": True,
                "category": "identity",
                "reason": "The user wants the assistant's public identity.",
            }
        )
        router = DirectResponseRouter(provider)

        response = await router.maybe_build_direct_response("how should I address you?")

        self.assertIsNotNone(response)
        self.assertEqual(response.category, "identity")
        self.assertIn("HealthFlow", response.answer)
        self.assertEqual(len(provider.calls), 1)

    async def test_skips_llm_routing_for_obvious_task_requests(self):
        provider = _StubStructuredProvider(
            {
                "respond_directly": True,
                "category": "identity",
                "reason": "This payload should not be used.",
            }
        )
        router = DirectResponseRouter(provider)

        response = await router.maybe_build_direct_response("Summarize the repository structure briefly.")

        self.assertIsNone(response)
        self.assertEqual(provider.calls, [])


if __name__ == "__main__":
    unittest.main()
