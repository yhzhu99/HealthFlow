import tempfile
import unittest
from pathlib import Path

from healthflow.ehr import profile_workspace_data
from healthflow.verification import WorkspaceVerifier


class VerificationTests(unittest.TestCase):
    def test_modeling_workspace_passes_when_required_artifacts_exist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            (workspace / "data.csv").write_text("subject_id,outcome,age\n1,0,65\n", encoding="utf-8")
            (workspace / "final_report.md").write_text(
                "# Task Summary\n# Data Profile\n# Method\n# Verification\n# Limitations\n",
                encoding="utf-8",
            )
            (workspace / "metrics.json").write_text('{"auroc": 0.81}', encoding="utf-8")
            profile = profile_workspace_data(workspace, "Train a prediction model on the uploaded cohort.")
            verifier = WorkspaceVerifier(["Task Summary", "Data Profile", "Method", "Verification", "Limitations"])
            result = verifier.verify(workspace, profile.task_family, "STDOUT: done", profile)
            self.assertTrue(result.passed)
            self.assertTrue(any(path.endswith("metrics.json") for path in result.artifact_paths))

    def test_verification_fails_without_required_sections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            (workspace / "final_report.md").write_text("# Task Summary\n", encoding="utf-8")
            profile = profile_workspace_data(workspace, "Generate a report from public EHR data.")
            verifier = WorkspaceVerifier(["Task Summary", "Data Profile"])
            result = verifier.verify(workspace, profile.task_family, "STDOUT: done", profile)
            self.assertFalse(result.passed)


if __name__ == "__main__":
    unittest.main()
