import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from healthflow.core.config import (
    EvaluationConfig,
    EnvironmentConfig,
    ExecutorConfig,
    HealthFlowConfig,
    LLMProviderConfig,
    LLMRoleConfig,
    LoggingConfig,
    MemoryConfig,
    SystemConfig,
    default_executor_backends,
)
from run_benchmark import (
    BENCHMARK_MEMORY_POLICY_SOURCE,
    BENCHMARK_MEMORY_WRITE_POLICY,
    _force_benchmark_memory_policy,
    aggregate_cost_totals,
)


class BenchmarkRunnerTests(unittest.TestCase):
    def test_force_benchmark_memory_policy_overrides_config_without_mutating_original(self):
        config = HealthFlowConfig(
            active_llm_name="test-llm",
            active_executor_name="opencode",
            llm_registry={
                "test-llm": LLMProviderConfig(
                    api_key="key",
                    base_url="https://example.com/v1",
                    model_name="test-model",
                )
            },
            llm=LLMProviderConfig(
                api_key="key",
                base_url="https://example.com/v1",
                model_name="test-model",
            ),
            llm_roles=LLMRoleConfig(),
            system=SystemConfig(),
            environment=EnvironmentConfig(),
            executor=ExecutorConfig(active_backend="opencode", backends=default_executor_backends()),
            memory=MemoryConfig(write_policy="append"),
            evaluation=EvaluationConfig(),
            logging=LoggingConfig(),
        )

        frozen_config = _force_benchmark_memory_policy(config)

        self.assertEqual(config.memory.write_policy, "append")
        self.assertEqual(frozen_config.memory.write_policy, BENCHMARK_MEMORY_WRITE_POLICY)
        self.assertEqual(frozen_config.active_executor_name, config.active_executor_name)
        self.assertEqual(frozen_config.active_llm_name, config.active_llm_name)

    def test_aggregate_cost_totals_combines_stage_and_executor_costs(self):
        results = [
            {
                "cost_summary": {
                    "llm_estimated_cost_usd": 0.01,
                    "executor_estimated_cost_usd": 0.2,
                    "total_estimated_cost_usd": 0.21,
                },
                "memory_write_policy": BENCHMARK_MEMORY_WRITE_POLICY,
                "memory_write_policy_forced": True,
                "memory_write_policy_source": BENCHMARK_MEMORY_POLICY_SOURCE,
                "cost_analysis": {
                    "run_total": {
                        "planning": {"estimated_cost_usd": 0.004},
                        "execution": {"estimated_cost_usd": 0.2},
                        "evaluation": {"estimated_cost_usd": 0.006},
                        "reflection": {},
                        "total_estimated_cost_usd": 0.21,
                    }
                },
            },
            {
                "cost_summary": {
                    "llm_estimated_cost_usd": 0.015,
                    "executor_estimated_cost_usd": 0.05,
                    "total_estimated_cost_usd": 0.065,
                },
                "memory_write_policy": BENCHMARK_MEMORY_WRITE_POLICY,
                "memory_write_policy_forced": True,
                "memory_write_policy_source": BENCHMARK_MEMORY_POLICY_SOURCE,
                "cost_analysis": {
                    "run_total": {
                        "planning": {"estimated_cost_usd": 0.005},
                        "execution": {"estimated_cost_usd": 0.05},
                        "evaluation": {"estimated_cost_usd": 0.007},
                        "reflection": {"estimated_cost_usd": 0.003},
                        "total_estimated_cost_usd": 0.065,
                    }
                },
            },
        ]

        totals = aggregate_cost_totals(results)

        self.assertEqual(totals["total_llm_estimated_cost_usd"], 0.025)
        self.assertEqual(totals["total_executor_estimated_cost_usd"], 0.25)
        self.assertEqual(totals["total_estimated_cost_usd"], 0.275)
        self.assertEqual(
            totals["stage_cost_totals_usd"],
            {
                "planning": 0.009,
                "execution": 0.25,
                "evaluation": 0.013,
                "reflection": 0.003,
                "total": 0.275,
            },
        )

    def test_benchmark_results_can_record_forced_freeze_metadata(self):
        result = {
            "memory_write_policy": BENCHMARK_MEMORY_WRITE_POLICY,
            "memory_write_policy_forced": True,
            "memory_write_policy_source": BENCHMARK_MEMORY_POLICY_SOURCE,
        }

        self.assertEqual(result["memory_write_policy"], "freeze")
        self.assertTrue(result["memory_write_policy_forced"])
        self.assertEqual(result["memory_write_policy_source"], "benchmark_forced_freeze")


if __name__ == "__main__":
    unittest.main()
