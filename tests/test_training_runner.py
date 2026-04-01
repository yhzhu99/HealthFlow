from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from run_training import TrainingRunner


class TrainingRunnerTests(unittest.TestCase):
    def test_load_training_data_accepts_reference_answer_only(self):
        runner = TrainingRunner(system=None, experience_path=Path("."))
        with TemporaryDirectory() as tmp_dir:
            training_path = Path(tmp_dir) / "train.jsonl"
            training_path.write_text(
                '{"qid": 1, "task": "task one", "reference_answer": "reference_answers/1/answer_manifest.json"}\n',
                encoding="utf-8",
            )

            examples = runner._load_training_data(training_path)

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].qid, "1")
        self.assertEqual(examples[0].answer, "reference_answers/1/answer_manifest.json")


if __name__ == "__main__":
    unittest.main()
