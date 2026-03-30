import tempfile
import unittest
from pathlib import Path

from healthflow.ehr import detect_risk_findings, profile_workspace_data


class EHRProfilingTests(unittest.TestCase):
    def test_workspace_profile_and_risk_checks_capture_ehr_signals(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            (workspace / "patients.csv").write_text(
                "subject_id,mortality,label,age\n1,0,0,65\n2,1,1,71\n",
                encoding="utf-8",
            )
            (workspace / "notes.txt").write_text(
                "Discharge summary for retrospective mortality analysis.\n",
                encoding="utf-8",
            )
            request = "Build a mortality prediction model from discharge notes and structured EHR data."
            profile = profile_workspace_data(workspace, request)

            self.assertEqual(profile.task_family, "predictive_modeling")
            self.assertIn("structured_tabular", profile.modalities)
            self.assertIn("clinical_text", profile.modalities)
            self.assertTrue(profile.dataset_signature)

            findings = detect_risk_findings(request, profile)
            finding_text = " ".join(item.message for item in findings).lower()
            self.assertIn("patient-aware", finding_text)
            self.assertIn("target-like", finding_text)


if __name__ == "__main__":
    unittest.main()
