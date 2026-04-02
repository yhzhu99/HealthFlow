from collections import Counter
from pathlib import Path
import unittest

from data.ehrflowbench.scripts.curate_generated_tasks import GeneratedTaskCandidate
from data.ehrflowbench.scripts.curate_generated_tasks import ReviewedTask
from data.ehrflowbench.scripts.curate_generated_tasks import allocate_bucket_quotas
from data.ehrflowbench.scripts.curate_generated_tasks import assign_qids
from data.ehrflowbench.scripts.curate_generated_tasks import assign_splits
from data.ehrflowbench.scripts.curate_generated_tasks import build_dataset_row
from data.ehrflowbench.scripts.curate_generated_tasks import build_task_prompt
from data.ehrflowbench.scripts.curate_generated_tasks import classify_required_input_set
from data.ehrflowbench.scripts.curate_generated_tasks import review_candidate
from data.ehrflowbench.scripts.curate_generated_tasks import select_best_task_per_paper


def make_reviewed_task(
    *,
    paper_id: int,
    task_idx: int,
    bucket: str,
    quality_score: int,
    input_dataset: str = "tjh",
    method_family: str = "prediction",
    target_family: str = "outcome_prediction",
    lightweight_hint: bool = False,
    has_local_only_constraint: bool = True,
) -> ReviewedTask:
    required_inputs = {
        "tjh": (
            "data/ehrflowbench/processed/tjh/tjh_formatted_ehr.parquet",
            "data/ehrflowbench/processed/tjh/split_metadata.json",
        ),
        "mimic_iv_demo": (
            "data/ehrflowbench/processed/mimic_iv_demo/mimic_iv_demo_formatted_ehr.parquet",
            "data/ehrflowbench/processed/mimic_iv_demo/split_metadata.json",
            "data/ehrflowbench/processed/mimic_iv_demo/mimic_iv_demo_value_reference.md",
        ),
    }[input_dataset]

    candidate = GeneratedTaskCandidate(
        paper_id=paper_id,
        paper_title=f"Paper {paper_id}",
        task_idx=task_idx,
        task_brief=f"Task {paper_id}-{task_idx}",
        task_type="report_generation",
        focus_areas=("focus",),
        task="Build a deterministic benchmark with report.md and metrics.json.",
        required_inputs=required_inputs,
        deliverables=("report.md", "metrics.json", "tables/results.csv"),
        report_requirements=("Report metrics.",) * 6,
        tasks_path=Path("/tmp") / f"{paper_id}_tasks.json",
        response_path=Path("/tmp") / f"{paper_id}_response.json",
    )
    return ReviewedTask(
        source=candidate,
        primary_bucket=bucket,
        method_family=method_family,
        target_family=target_family,
        quality_score=quality_score,
        feasibility_score=3,
        specificity_score=2,
        evaluability_score=2,
        practicality_score=2,
        novelty_score=max(0, quality_score - 9),
        has_local_only_constraint=has_local_only_constraint,
        has_figure_or_table_artifact=True,
        has_numeric_artifact=True,
        lightweight_hint=lightweight_hint,
        input_dataset=input_dataset,
        has_valid_single_dataset_inputs=True,
        has_split_metadata_inputs=True,
    )


class EHRFlowBenchCurationTests(unittest.TestCase):
    def test_classify_required_input_set_accepts_single_dataset_and_rejects_mixed(self):
        matched_dataset, is_valid, has_split, is_mixed = classify_required_input_set(
            (
                "data/ehrflowbench/processed/tjh/tjh_formatted_ehr.parquet",
                "data/ehrflowbench/processed/tjh/split_metadata.json",
            )
        )

        self.assertEqual(matched_dataset, "tjh")
        self.assertTrue(is_valid)
        self.assertTrue(has_split)
        self.assertFalse(is_mixed)

        matched_dataset, is_valid, has_split, is_mixed = classify_required_input_set(
            (
                "data/ehrflowbench/processed/tjh/tjh_formatted_ehr.parquet",
                "data/ehrflowbench/processed/tjh/split_metadata.json",
                "data/ehrflowbench/processed/mimic_iv_demo/mimic_iv_demo_formatted_ehr.parquet",
            )
        )

        self.assertIsNone(matched_dataset)
        self.assertFalse(is_valid)
        self.assertFalse(has_split)
        self.assertTrue(is_mixed)

    def test_allocate_bucket_quotas_preserves_total_and_minimums(self):
        quotas = allocate_bucket_quotas(
            Counter(
                {
                    "other": 49,
                    "graph": 29,
                    "representation": 21,
                    "temporal": 21,
                    "prediction": 12,
                    "causal": 10,
                    "unsupervised": 6,
                    "interpretability": 4,
                    "fairness": 1,
                }
            ),
            110,
        )

        self.assertEqual(sum(quotas.values()), 110)
        self.assertGreaterEqual(quotas["fairness"], 1)
        self.assertGreaterEqual(quotas["graph"], 1)
        self.assertGreaterEqual(quotas["causal"], 1)

    def test_select_best_task_per_paper_prefers_lightweight_hint_on_tie(self):
        stronger = make_reviewed_task(
            paper_id=7,
            task_idx=1,
            bucket="prediction",
            quality_score=8,
            lightweight_hint=True,
        )
        weaker = make_reviewed_task(
            paper_id=7,
            task_idx=2,
            bucket="prediction",
            quality_score=8,
            lightweight_hint=False,
        )

        selected, rejected = select_best_task_per_paper([weaker, stronger], min_quality_score=7)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].task_idx, 1)
        self.assertIn((7, 2), rejected)
        self.assertTrue(any("paper_peer_lower_rank" in reason for reason in rejected[(7, 2)]))

    def test_assign_splits_keeps_high_score_fairness_task_out_of_train(self):
        selected = [
            make_reviewed_task(paper_id=1, task_idx=1, bucket="graph", quality_score=9),
            make_reviewed_task(paper_id=2, task_idx=1, bucket="temporal", quality_score=9),
            make_reviewed_task(paper_id=3, task_idx=1, bucket="representation", quality_score=9),
            make_reviewed_task(paper_id=4, task_idx=1, bucket="prediction", quality_score=9),
            make_reviewed_task(paper_id=5, task_idx=1, bucket="causal", quality_score=9, method_family="causal", target_family="utilization"),
            make_reviewed_task(paper_id=6, task_idx=1, bucket="unsupervised", quality_score=9, method_family="unsupervised", target_family="clustering"),
            make_reviewed_task(paper_id=7, task_idx=1, bucket="interpretability", quality_score=9, method_family="feature_selection"),
            make_reviewed_task(paper_id=8, task_idx=1, bucket="fairness", quality_score=9, method_family="fairness"),
            make_reviewed_task(paper_id=9, task_idx=1, bucket="other", quality_score=8),
            make_reviewed_task(paper_id=10, task_idx=1, bucket="graph", quality_score=8),
            make_reviewed_task(paper_id=11, task_idx=1, bucket="temporal", quality_score=8),
        ]

        train, eval_tasks = assign_splits(selected, train_size=10, test_size=1)

        self.assertEqual(len(train), 10)
        self.assertEqual(len(eval_tasks), 1)
        self.assertEqual(eval_tasks[0].paper_id, 8)
        self.assertTrue(all(task.primary_bucket != "fairness" for task in train))
        self.assertTrue({"graph", "temporal", "representation", "prediction", "causal", "unsupervised", "interpretability"}.issubset({task.primary_bucket for task in train}))

    def test_build_dataset_row_uses_expected_manifest_reference(self):
        train = [make_reviewed_task(paper_id=21, task_idx=1, bucket="graph", quality_score=9)]
        eval_tasks = [make_reviewed_task(paper_id=22, task_idx=1, bucket="prediction", quality_score=8)]

        assign_qids(train, eval_tasks)
        row = build_dataset_row(train[0])

        self.assertEqual(row["qid"], 1)
        self.assertEqual(row["reference_answer"], "expected/1/task_manifest.json")

    def test_build_task_prompt_mentions_single_assigned_dataset(self):
        task = make_reviewed_task(
            paper_id=31,
            task_idx=1,
            bucket="prediction",
            quality_score=9,
            input_dataset="mimic_iv_demo",
        )

        prompt = build_task_prompt(task)

        self.assertIn("MIMIC-IV-demo only", prompt)
        self.assertNotIn("both TJH and MIMIC-IV-demo", prompt)
        self.assertIn("Do not call external APIs", prompt)
        self.assertIn("Objective", prompt)

    def test_review_candidate_rejects_cross_dataset_reference_and_missing_figure_artifact(self):
        candidate = GeneratedTaskCandidate(
            paper_id=41,
            paper_title="Paper 41",
            task_idx=1,
            task_brief="Cross-dataset task",
            task_type="report_generation",
            focus_areas=("prediction", "benchmarking"),
            task=(
                "Use only the TJH dataset, but compare the final model against MIMIC-IV-demo "
                "and report the result in a summary."
            ),
            required_inputs=(
                "data/ehrflowbench/processed/tjh/tjh_formatted_ehr.parquet",
                "data/ehrflowbench/processed/tjh/split_metadata.json",
            ),
            deliverables=("report.md", "metrics.json"),
            report_requirements=("Report metrics.",) * 6,
            tasks_path=Path("/tmp/41_tasks.json"),
            response_path=Path("/tmp/41_response.json"),
        )

        reviewed = review_candidate(candidate)

        self.assertIn("mentions_other_dataset", reviewed.hard_reject_reasons)
        self.assertIn("missing_figure_or_table_artifact", reviewed.hard_reject_reasons)

    def test_review_candidate_rejects_source_paper_references_and_legacy_cross_dataset_targets(self):
        candidate = GeneratedTaskCandidate(
            paper_id=42,
            paper_title="Paper 42",
            task_idx=1,
            task_brief="Legacy task",
            task_type="report_generation",
            focus_areas=("prediction", "reporting"),
            task=(
                "As described in the paper, reproduce Table 2 and choose one binary target that is present in "
                "both TJH and MIMIC-IV-demo. Use the processed TJH dataset only for execution."
            ),
            required_inputs=(
                "data/ehrflowbench/processed/tjh/tjh_formatted_ehr.parquet",
                "data/ehrflowbench/processed/tjh/split_metadata.json",
            ),
            deliverables=("report.md", "metrics.json", "tables/results.csv"),
            report_requirements=("Report metrics.",) * 6,
            tasks_path=Path("/tmp/42_tasks.json"),
            response_path=Path("/tmp/42_response.json"),
        )

        reviewed = review_candidate(candidate)

        self.assertIn("source_paper_reference", reviewed.hard_reject_reasons)
        self.assertIn("legacy_cross_dataset_target_instruction", reviewed.hard_reject_reasons)


if __name__ == "__main__":
    unittest.main()
