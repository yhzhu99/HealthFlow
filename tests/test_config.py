import tempfile
import unittest
from pathlib import Path

from healthflow.core.config import get_config


class ConfigTests(unittest.TestCase):
    def test_executor_defaults_are_applied_when_backends_are_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text(
                """
[llm.test]
api_key = "key"
base_url = "https://example.com/v1"
model_name = "model"
""".strip(),
                encoding="utf-8",
            )
            config = get_config(config_path, "test")
            self.assertEqual(config.active_executor_name, "claude_code")
            self.assertIn("claude_code", config.executor.backends)
            self.assertIn("opencode", config.executor.backends)
            self.assertIn("pi", config.executor.backends)


if __name__ == "__main__":
    unittest.main()
