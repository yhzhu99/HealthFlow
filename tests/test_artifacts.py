import tempfile
import unittest
from pathlib import Path

from healthflow.artifacts import collect_task_artifacts, read_structured_preview
from healthflow.session import TaskTurnRecord


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.task_root = Path(self.temp_dir.name)
        (self.task_root / "sandbox").mkdir(parents=True, exist_ok=True)
        (self.task_root / "runtime").mkdir(parents=True, exist_ok=True)
        (self.task_root / "uploads" / "turn_001").mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_collect_task_artifacts_returns_visible_workspace_files(self):
        upload_path = self.task_root / "uploads" / "turn_001" / "notes.md"
        upload_path.write_text("# Notes\n\nImportant upload.\n", encoding="utf-8")
        uploaded_workspace_path = self.task_root / "sandbox" / "notes.md"
        uploaded_workspace_path.write_text("# Notes\n\nImportant upload.\n", encoding="utf-8")
        generated_path = self.task_root / "sandbox" / "results.csv"
        generated_path.write_text("metric,value\nauroc,0.91\n", encoding="utf-8")
        report_path = self.task_root / "runtime" / "report.md"
        report_path.write_text("# HealthFlow Report\n\nSummary.\n", encoding="utf-8")

        history = [
            TaskTurnRecord(
                turn_number=1,
                user_message="Analyze the uploaded notes",
                answer="Done",
                status="success",
                runtime_dir=str(self.task_root / "runtime" / "turns" / "turn_001"),
                uploaded_files=[
                    {
                        "original_name": "notes.md",
                        "upload_path": "uploads/turn_001/notes.md",
                        "sandbox_path": "sandbox/notes.md",
                    }
                ],
            )
        ]

        catalog = collect_task_artifacts(self.task_root, history)
        origins = {item["task_relative_path"]: item["origin"] for item in catalog}
        preview_kinds = {item["task_relative_path"]: item["preview_kind"] for item in catalog}

        self.assertNotIn("uploads/turn_001/notes.md", origins)
        self.assertEqual(origins["sandbox/notes.md"], "uploaded")
        self.assertEqual(origins["sandbox/results.csv"], "generated")
        self.assertEqual(origins["runtime/report.md"], "report")
        self.assertEqual(preview_kinds["runtime/report.md"], "markdown")
        self.assertEqual(preview_kinds["sandbox/results.csv"], "table")

    def test_read_structured_preview_handles_csv_and_json(self):
        csv_path = self.task_root / "sandbox" / "metrics.csv"
        csv_path.write_text("metric,value\nauroc,0.91\nauprc,0.73\n", encoding="utf-8")
        json_path = self.task_root / "sandbox" / "summary.json"
        json_path.write_text('{"auroc": 0.91, "folds": 5}', encoding="utf-8")

        csv_preview = read_structured_preview(csv_path)
        json_preview = read_structured_preview(json_path)

        self.assertEqual(csv_preview["headers"], ["metric", "value"])
        self.assertEqual(csv_preview["rows"][0], ["auroc", "0.91"])
        self.assertEqual(json_preview["headers"], ["Field", "Value"])
        self.assertEqual(json_preview["rows"][0][0], "auroc")


if __name__ == "__main__":
    unittest.main()
