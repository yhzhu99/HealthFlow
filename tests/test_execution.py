import unittest

from healthflow.core.config import BackendCLIConfig
from healthflow.execution.cli_adapters import CLISubprocessExecutor, HealthFlowAgentExecutor, PiExecutor
from healthflow.execution.factory import create_executor_adapter


class ExecutionFactoryTests(unittest.TestCase):
    def test_healthflow_agent_uses_integrated_executor(self):
        executor = create_executor_adapter(
            "healthflow_agent",
            BackendCLIConfig(binary="healthflow-agent", args=["-p"]),
        )
        self.assertIsInstance(executor, HealthFlowAgentExecutor)

    def test_unknown_backend_falls_back_to_generic_cli_executor(self):
        executor = create_executor_adapter(
            "external_backend",
            BackendCLIConfig(binary="external-agent", args=["--print"]),
        )
        self.assertIsInstance(executor, CLISubprocessExecutor)
        self.assertNotIsInstance(executor, HealthFlowAgentExecutor)

    def test_pi_backend_uses_pi_executor(self):
        executor = create_executor_adapter(
            "pi",
            BackendCLIConfig(binary="pi"),
        )
        self.assertIsInstance(executor, PiExecutor)


if __name__ == "__main__":
    unittest.main()
