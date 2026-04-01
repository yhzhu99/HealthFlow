import tempfile
import unittest
from pathlib import Path

from healthflow.ehr import detect_risk_findings, profile_workspace_data
from healthflow.execution import WorkflowRecommendationBroker


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
            self.assertEqual(profile.domain_focus, "ehr")
            self.assertIn("structured_tabular", profile.modalities)
            self.assertIn("clinical_text", profile.modalities)
            self.assertTrue(profile.dataset_signature)
            self.assertIn("subject_id", profile.group_id_columns)
            self.assertIn("subject_id", profile.patient_id_columns)
            self.assertIn("label", profile.target_columns)
            self.assertIn("event_time", profile.time_columns)

            findings = detect_risk_findings(request, profile)
            finding_text = " ".join(item.message for item in findings).lower()
            self.assertIn("patient-aware", finding_text)
            self.assertIn("target-like", finding_text)
            self.assertIn("split evidence", finding_text)

            recommendations = WorkflowRecommendationBroker().recommend(request, profile)
            recommendation_text = " ".join(recommendations).lower()
            self.assertIn("oneehr", recommendation_text)
            self.assertIn("split", recommendation_text)
            self.assertIn("leakage", recommendation_text)

    def test_general_modeling_request_stays_general_without_ehr_overlay(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            (workspace / "sales.csv").write_text(
                "customer_id,target,revenue,order_date\n1,120,1000,2024-01-01\n2,95,900,2024-01-02\n",
                encoding="utf-8",
            )
            request = "Train a regression model to predict next-month revenue from the uploaded sales table."
            profile = profile_workspace_data(workspace, request)

            self.assertEqual(profile.task_family, "predictive_modeling")
            self.assertEqual(profile.domain_focus, "general")
            self.assertIn("structured_tabular", profile.modalities)
            self.assertIn("customer_id", profile.group_id_columns)
            self.assertFalse(profile.patient_id_columns)

            findings = detect_risk_findings(request, profile)
            finding_text = " ".join(item.message for item in findings).lower()
            self.assertIn("entity-aware", finding_text)
            self.assertNotIn("patient-aware", finding_text)

            recommendations = WorkflowRecommendationBroker().recommend(request, profile)
            recommendation_text = " ".join(recommendations).lower()
            self.assertNotIn("oneehr", recommendation_text)
            self.assertIn("uv run", recommendation_text)

    def test_tooluniverse_is_only_suggested_for_explicit_tool_lookup_requests(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            request = "Use ToolUniverse to find a biomedical tool for pathway analysis on the uploaded gene list."
            profile = profile_workspace_data(workspace, request)

            recommendations = WorkflowRecommendationBroker().recommend(request, profile)
            recommendation_text = " ".join(recommendations).lower()

            self.assertIn("tooluniverse", recommendation_text)
            self.assertIn("tu", recommendation_text)


if __name__ == "__main__":
    unittest.main()
