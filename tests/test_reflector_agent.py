import json
import unittest

from healthflow.agents.reflector_agent import ReflectorAgent
from healthflow.core.contracts import EvaluationVerdict
from healthflow.core.llm_provider import LLMResponse
from healthflow.experience.experience_models import MemoryKind, SourceOutcome


class _StubStructuredProvider:
    def __init__(self, payload: dict[str, object]):
        self.payload = payload
        self.calls: list[list[dict[str, str]]] = []
        self.model_name = "reflector-test-model"
        self.last_structured_trace = {}

    async def generate_structured(self, messages, parser, temperature=0.0, max_tokens=None, max_attempts=3):
        self.calls.append([message.model_dump() for message in messages])
        content = json.dumps(self.payload)
        return (
            parser(content),
            LLMResponse(
                content=content,
                model_name=self.model_name,
                usage={"input_tokens": 12, "output_tokens": 4, "total_tokens": 16},
                estimated_cost_usd=0.0001,
            ),
        )


def _full_history(*, attempts: int = 1, risk_findings: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "task_id": "task-1",
        "user_request": "Build a readmission model.",
        "data_profile": {
            "task_family": "predictive_modeling",
            "dataset_signature": "abc123",
        },
        "risk_findings": risk_findings or [],
        "backend": "codex",
        "attempts": [
            {
                "attempt": attempt_index + 1,
                "memory": {"retrieval": {"selected": [], "safeguard_overrides": []}},
                "execution": {"success": attempt_index == attempts - 1, "return_code": 0, "timed_out": False, "log": ""},
                "evaluation": {"status": "success" if attempt_index == attempts - 1 else "needs_retry"},
            }
            for attempt_index in range(attempts)
        ],
    }


class ReflectorAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_writeback_keeps_workflow_dataset_anchor_and_code_snippet_only(self):
        provider = _StubStructuredProvider(
            {
                "experiences": [
                    {
                        "kind": "safeguard",
                        "category": "split_policy",
                        "content": "Always use patient-level splitting before reporting metrics.",
                        "confidence": 0.95,
                        "applicability_scope": "domain_ehr",
                        "risk_tags": ["validation_strategy"],
                        "schema_tags": ["domain:ehr"],
                        "tags": [],
                        "supersedes": [],
                    },
                    {
                        "kind": "workflow",
                        "category": "model_training",
                        "content": "Inspect the schema, train the model, and save metrics.",
                        "confidence": 0.8,
                        "applicability_scope": "task_family",
                        "risk_tags": [],
                        "schema_tags": ["domain:ehr"],
                        "tags": [],
                        "supersedes": [],
                    },
                    {
                        "kind": "dataset_anchor",
                        "category": "core_schema",
                        "content": "Dataset abc123 includes subject_id and readmission labels.",
                        "confidence": 0.7,
                        "applicability_scope": "task_family",
                        "risk_tags": [],
                        "schema_tags": ["domain:ehr"],
                        "tags": [],
                        "supersedes": [],
                    },
                    {
                        "kind": "code_snippet",
                        "category": "split_assertion",
                        "content": "assert set(train.subject_id).isdisjoint(set(test.subject_id))",
                        "confidence": 0.75,
                        "applicability_scope": "task_family",
                        "risk_tags": [],
                        "schema_tags": ["domain:ehr"],
                        "tags": ["python"],
                        "supersedes": [],
                    },
                ],
                "memory_updates": [],
            }
        )
        agent = ReflectorAgent(provider)

        result = await agent.synthesize_experience(
            _full_history(),
            final_verdict=EvaluationVerdict(
                status="success",
                score=0.95,
                failure_type="none",
                feedback="Looks good.",
                repair_instructions=[],
                retry_recommended=False,
                memory_worthy_insights=[],
                reasoning="The task succeeded cleanly.",
            ),
        )

        self.assertEqual([exp.kind for exp in result.experiences], [
            MemoryKind.WORKFLOW,
            MemoryKind.DATASET_ANCHOR,
            MemoryKind.CODE_SNIPPET,
        ])
        self.assertTrue(all(exp.source_outcome == SourceOutcome.SUCCESS for exp in result.experiences))
        self.assertEqual(result.experiences[1].applicability_scope, "dataset_exact")

    async def test_failed_writeback_keeps_only_one_safeguard(self):
        provider = _StubStructuredProvider(
            {
                "experiences": [
                    {
                        "kind": "workflow",
                        "category": "split_policy",
                        "content": "Patient-level splitting is required to avoid leakage.",
                        "confidence": 0.9,
                        "applicability_scope": "task_family",
                        "risk_tags": ["validation_strategy"],
                        "schema_tags": ["domain:ehr"],
                        "tags": [],
                        "supersedes": [],
                    },
                    {
                        "kind": "safeguard",
                        "category": "missingness_guard",
                        "content": "Do not impute clinically meaningful missing values without validating the clinical meaning first.",
                        "confidence": 0.8,
                        "applicability_scope": "domain_ehr",
                        "risk_tags": ["unsafe_missing_value_handling"],
                        "schema_tags": ["domain:ehr"],
                        "tags": [],
                        "supersedes": [],
                    },
                    {
                        "kind": "code_snippet",
                        "category": "split_assertion",
                        "content": "assert set(train.subject_id).isdisjoint(set(test.subject_id))",
                        "confidence": 0.7,
                        "applicability_scope": "task_family",
                        "risk_tags": [],
                        "schema_tags": ["domain:ehr"],
                        "tags": ["python"],
                        "supersedes": [],
                    },
                ],
                "memory_updates": [],
            }
        )
        agent = ReflectorAgent(provider)

        result = await agent.synthesize_experience(
            _full_history(risk_findings=[{"category": "validation_strategy", "severity": "high"}]),
            final_verdict=EvaluationVerdict(
                status="failed",
                score=0.2,
                failure_type="analysis_incomplete",
                feedback="The task failed.",
                repair_instructions=["Fix the split strategy."],
                retry_recommended=False,
                memory_worthy_insights=[],
                reasoning="The task failed.",
            ),
        )

        self.assertEqual(len(result.experiences), 1)
        self.assertEqual(result.experiences[0].kind, MemoryKind.SAFEGUARD)
        self.assertEqual(result.experiences[0].source_outcome, SourceOutcome.FAILED)

    async def test_recovered_writeback_keeps_safeguard_plus_one_corrected_procedure(self):
        provider = _StubStructuredProvider(
            {
                "experiences": [
                    {
                        "kind": "safeguard",
                        "category": "split_policy",
                        "content": "Require patient-level splitting before reporting metrics.",
                        "confidence": 0.92,
                        "applicability_scope": "domain_ehr",
                        "risk_tags": ["validation_strategy", "patient_linkage"],
                        "schema_tags": ["domain:ehr"],
                        "tags": [],
                        "supersedes": [],
                    },
                    {
                        "kind": "workflow",
                        "category": "model_training",
                        "content": "After fixing the split, retrain the model and save metrics plus split evidence.",
                        "confidence": 0.81,
                        "applicability_scope": "task_family",
                        "risk_tags": [],
                        "schema_tags": ["domain:ehr"],
                        "tags": [],
                        "supersedes": [],
                    },
                    {
                        "kind": "code_snippet",
                        "category": "split_assertion",
                        "content": "assert set(train.subject_id).isdisjoint(set(test.subject_id))",
                        "confidence": 0.88,
                        "applicability_scope": "task_family",
                        "risk_tags": [],
                        "schema_tags": ["domain:ehr"],
                        "tags": ["python"],
                        "supersedes": [],
                    },
                    {
                        "kind": "dataset_anchor",
                        "category": "core_schema",
                        "content": "Dataset abc123 includes subject_id and readmission labels.",
                        "confidence": 0.99,
                        "applicability_scope": "dataset_exact",
                        "risk_tags": [],
                        "schema_tags": ["domain:ehr"],
                        "tags": [],
                        "supersedes": [],
                    },
                ],
                "memory_updates": [],
            }
        )
        agent = ReflectorAgent(provider)

        result = await agent.synthesize_experience(
            _full_history(attempts=2, risk_findings=[{"category": "validation_strategy", "severity": "medium"}]),
            final_verdict=EvaluationVerdict(
                status="success",
                score=0.93,
                failure_type="none",
                feedback="Recovered after fixing the split.",
                repair_instructions=[],
                retry_recommended=False,
                memory_worthy_insights=[],
                reasoning="The second attempt corrected the earlier issue.",
            ),
        )

        self.assertEqual([exp.kind for exp in result.experiences], [MemoryKind.SAFEGUARD, MemoryKind.CODE_SNIPPET])
        self.assertTrue(all(exp.source_outcome == SourceOutcome.RECOVERED for exp in result.experiences))


if __name__ == "__main__":
    unittest.main()
