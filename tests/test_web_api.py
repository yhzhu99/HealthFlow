import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app import create_app


async def _fake_runner(task: str, active_llm: str, active_executor: str | None, uploaded_files: dict[str, bytes]):
    return {
        "success": True,
        "answer": f"handled: {task}",
        "final_summary": "fake run complete",
        "backend": active_executor or "healthflow_agent",
        "reasoning_model": active_llm,
        "memory_mode": "accumulate_eval",
        "verification_passed": True,
        "execution_time": 0.01,
        "workspace_path": "/tmp/workspace/demo",
        "task_family": "predictive_modeling",
        "dataset_signature": "abc123",
        "log_path": None,
        "verification_path": None,
        "memory_context_path": None,
        "run_result_path": None,
        "run_manifest_path": None,
        "received_files": sorted(uploaded_files.keys()),
    }


class WebApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app(task_runner=_fake_runner, project_root=Path(__file__).resolve().parents[1]))

    def test_options_endpoint_exposes_executor_defaults(self):
        response = self.client.get("/api/options")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("healthflow_agent", payload["executor_options"])

    def test_run_endpoint_accepts_form_and_files(self):
        response = self.client.post(
            "/api/run",
            data={
                "task": "Analyze this cohort.",
                "active_llm": "test-llm",
                "active_executor": "healthflow_agent",
            },
            files=[("files", ("patients.csv", b"subject_id,label\n1,0\n", "text/csv"))],
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["received_files"], ["patients.csv"])


if __name__ == "__main__":
    unittest.main()
