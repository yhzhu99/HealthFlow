import tempfile
import unittest
from pathlib import Path

from healthflow.ehr import detect_risk_findings, profile_workspace_data
from healthflow.tools import ToolBroker


class EHRProfilingTests(unittest.TestCase):
    def test_workspace_profile_and_risk_checks_capture_ehr_signals(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            (workspace / "patients.csv").write_text(
                "subject_id,mortality,label,age,discharge_time,event_time\n1,0,0,65,2020-01-02,2020-01-01\n2,1,1,71,2020-01-03,2020-01-02\n",
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
            self.assertIn("subject_id", profile.patient_id_columns)
            self.assertIn("label", profile.target_columns)
            self.assertIn("event_time", profile.time_columns)

            findings = detect_risk_findings(request, profile)
            finding_text = " ".join(item.message for item in findings).lower()
            self.assertIn("patient-aware", finding_text)
            self.assertIn("target-like", finding_text)
            self.assertIn("split evidence", finding_text)

            bundle = ToolBroker().select_bundle(profile.task_family, profile)
            self.assertIn("patient-level split audit", bundle)
            self.assertIn("leakage + temporal audit", bundle)


if __name__ == "__main__":
    unittest.main()
