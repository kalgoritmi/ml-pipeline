"""Integration tests for the ML pipeline."""

import tempfile
import unittest
from pathlib import Path

from src.config import load_config
from src.pipeline import run_pipeline

ACCURACY_THRESHOLD = 0.5
CONFIG_PATH = Path(__file__).parent / "test_integration.yaml"


class TestPipelineIntegration(unittest.TestCase):
    def test_pipeline_accuracy_above_threshold(self):
        """Test that the pipeline achieves accuracy above threshold."""
        config = load_config(CONFIG_PATH)
        result = run_pipeline(config, CONFIG_PATH.parent)

        self.assertGreaterEqual(result["sklearn_accuracy"], ACCURACY_THRESHOLD)

    def test_checkpoint_count_matches_operations(self):
        """Test that the number of saved checkpoints equals the number of operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_config(CONFIG_PATH)
            config.checkpoint_path = tmpdir
            # Clear cached property to use new checkpoint path
            if "checkpoint_dir" in config.__dict__:
                del config.__dict__["checkpoint_dir"]

            run_pipeline(config, CONFIG_PATH.parent)

            checkpoint_files = list(Path(tmpdir).rglob("*.csv"))
            self.assertEqual(len(checkpoint_files), len(config.operations))


if __name__ == "__main__":
    unittest.main()
