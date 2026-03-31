import tempfile
import unittest
from pathlib import Path

from healthflow.ehr import profile_workspace_data
from healthflow.verification import WorkspaceVerifier


class VerificationTests(unittest.TestCase):
    def test_ehr_modeling_workspace_passes_when_required_artifacts_exist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            (workspace / "data.csv").write_text("subject_id,outcome,age,event_time\n1,0,65,2020-01-01\n", encoding="utf-8")
            (workspace / "final_report.md").write_text(
                "# Task Summary\n# Data Profile\n# Method\n# Verification\n# Limitations\n# Cohort Definition\n# Split Evidence\n# Leakage Audit\n# Metrics Summary\n",
                encoding="utf-8",
            )
            (workspace / "cohort_definition.json").write_text('{"name": "readmission cohort"}', encoding="utf-8")
            (workspace / "split_evidence.json").write_text('{"train": [1], "val": [], "test": []}', encoding="utf-8")
            (workspace / "leakage_audit.md").write_text("# Leakage Audit\nNo leakage found.\n", encoding="utf-8")
            (workspace / "metrics.json").write_text('{"auroc": 0.81}', encoding="utf-8")
            profile = profile_workspace_data(workspace, "Train a readmission prediction model on the uploaded EHR cohort.")
            verifier = WorkspaceVerifier(["Task Summary", "Data Profile", "Method", "Verification", "Limitations"])
            result = verifier.verify(workspace, profile.task_family, "STDOUT: done", profile)
            self.assertTrue(result.passed)
            self.assertTrue(any(path.endswith("metrics.json") for path in result.artifact_paths))
            self.assertTrue(any(check.name == "split_evidence" and check.passed for check in result.checks))

    def test_ehr_verification_fails_without_split_evidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            (workspace / "data.csv").write_text("subject_id,outcome\n1,0\n", encoding="utf-8")
            (workspace / "final_report.md").write_text(
                "# Task Summary\n# Data Profile\n# Method\n# Verification\n# Limitations\n# Cohort Definition\n# Leakage Audit\n# Metrics Summary\n",
                encoding="utf-8",
            )
            (workspace / "cohort_definition.json").write_text('{"name": "readmission cohort"}', encoding="utf-8")
            (workspace / "metrics.json").write_text('{"auroc": 0.81}', encoding="utf-8")
            profile = profile_workspace_data(workspace, "Train a readmission prediction model on the uploaded EHR cohort.")
            verifier = WorkspaceVerifier(["Task Summary", "Data Profile", "Method", "Verification", "Limitations"])
            result = verifier.verify(workspace, profile.task_family, "STDOUT: done", profile)
            self.assertFalse(result.passed)
            self.assertTrue(any(check.name == "split_evidence" and not check.passed for check in result.checks))

    def test_verification_accepts_oneehr_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            (workspace / "manifest.json").write_text('{"run": "demo"}', encoding="utf-8")
            (workspace / "preprocess").mkdir()
            (workspace / "preprocess" / "split.json").write_text('{"train_patients": [1], "val_patients": [], "test_patients": []}', encoding="utf-8")
            (workspace / "test").mkdir()
            (workspace / "test" / "metrics.json").write_text('{"auroc": 0.81}', encoding="utf-8")
            (workspace / "final_report.md").write_text(
                "# Task Summary\n# Data Profile\n# Method\n# Verification\n# Limitations\n# Cohort Definition\n# Split Evidence\n# Leakage Audit\n# Metrics Summary\n",
                encoding="utf-8",
            )
            profile = profile_workspace_data(workspace, "Train a readmission prediction model from EHR data.")
            verifier = WorkspaceVerifier(["Task Summary", "Data Profile", "Method", "Verification", "Limitations"])
            result = verifier.verify(workspace, profile.task_family, "STDOUT: done", profile)
            self.assertTrue(result.passed)
            self.assertTrue(any(path.endswith("preprocess/split.json") for path in result.artifact_paths))

    def test_general_modeling_can_pass_without_ehr_cohort_contracts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            (workspace / "sales.csv").write_text("customer_id,target,revenue\n1,120,1000\n2,95,900\n", encoding="utf-8")
            (workspace / "split_evidence.json").write_text('{"train": [1], "val": [], "test": [2]}', encoding="utf-8")
            (workspace / "metrics.json").write_text('{"rmse": 5.1}', encoding="utf-8")
            profile = profile_workspace_data(workspace, "Train a regression model to predict revenue from the uploaded sales data.")
            verifier = WorkspaceVerifier(["Task Summary", "Data Profile", "Method", "Verification", "Limitations"])
            result = verifier.verify(workspace, profile.task_family, "STDOUT: done", profile)
            self.assertTrue(result.passed)
            self.assertFalse(any(check.name == "cohort_definition" for check in result.checks))


if __name__ == "__main__":
    unittest.main()
