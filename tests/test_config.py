"""Unit tests for config module."""

import tempfile
import unittest
from pathlib import Path

import yaml

from src.config import (
    ColumnWeight,
    OperationConfig,
    PipelineConfig,
    SplittingConfig,
    load_config,
)


class TestColumnWeight(unittest.TestCase):
    def test_valid(self):
        cw = ColumnWeight(name="col1", weight=1.5)
        self.assertEqual(cw.name, "col1")
        self.assertEqual(cw.weight, 1.5)

    def test_negative_weight(self):
        cw = ColumnWeight(name="col1", weight=-2.0)
        self.assertEqual(cw.weight, -2.0)


class TestSplittingConfig(unittest.TestCase):
    def test_valid_split(self):
        sc = SplittingConfig(train=0.8, validation=0.2)
        self.assertEqual(sc.train, 0.8)
        self.assertEqual(sc.validation, 0.2)

    def test_invalid_train_ratio_zero(self):
        with self.assertRaises(ValueError):
            SplittingConfig(train=0.0, validation=1.0)

    def test_invalid_train_ratio_one(self):
        with self.assertRaises(ValueError):
            SplittingConfig(train=1.0, validation=0.0)

    def test_invalid_validation_ratio(self):
        with self.assertRaises(ValueError):
            SplittingConfig(train=0.5, validation=1.5)


class TestOperationConfig(unittest.TestCase):
    def test_index_operation(self):
        op = OperationConfig(type="IndexOperation", column="Time")
        self.assertEqual(op.type, "IndexOperation")
        self.assertEqual(op.column, "Time")

    def test_remove_columns(self):
        op = OperationConfig(type="RemoveColumns", columns=["a", "b"])
        self.assertEqual(op.columns, ["a", "b"])

    def test_compute_target(self):
        op = OperationConfig(
            type="ComputeTarget",
            columns=[{"name": "col1", "weight": 1.0}],
            target_column="target",
            threshold=0.5,
        )
        self.assertEqual(op.target_column, "target")
        self.assertEqual(op.threshold, 0.5)

    def test_shuffle(self):
        op = OperationConfig(type="Shuffle", random_state=42)
        self.assertEqual(op.random_state, 42)

    def test_limit_rows(self):
        op = OperationConfig(type="LimitRows", n_rows=100)
        self.assertEqual(op.n_rows, 100)

    def test_invalid_type(self):
        with self.assertRaises(ValueError):
            OperationConfig(type="InvalidOp")


class TestPipelineConfig(unittest.TestCase):
    def test_minimal_config(self):
        config = PipelineConfig(
            dataset_file="data.csv",
            operations=[],
            target="target",
            splitting=SplittingConfig(train=0.8, validation=0.2),
        )
        self.assertEqual(config.dataset_file, "data.csv")
        self.assertEqual(config.version, "0.1")

    def test_full_config(self):
        config = PipelineConfig(
            version="2.0",
            dataset_file="data.csv",
            dataset_url="http://example.com/data.csv",
            operations=[OperationConfig(type="Shuffle")],
            target="target",
            splitting=SplittingConfig(train=0.7, validation=0.3),
            checkpoint_path="./checkpoints",
        )
        self.assertEqual(config.version, "2.0")
        self.assertEqual(config.dataset_url, "http://example.com/data.csv")

    def test_checkpoint_dir_none_when_no_path(self):
        config = PipelineConfig(
            dataset_file="data.csv",
            operations=[],
            target="target",
            splitting=SplittingConfig(train=0.8, validation=0.2),
        )
        self.assertIsNone(config.checkpoint_dir)

    def test_checkpoint_dir_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                dataset_file="data.csv",
                operations=[],
                target="target",
                splitting=SplittingConfig(train=0.8, validation=0.2),
                checkpoint_path=tmpdir,
            )
            config._config_filename = "test_config"
            checkpoint = config.checkpoint_dir
            self.assertIsNotNone(checkpoint)
            self.assertTrue(checkpoint.exists())
            self.assertIn("test_config", str(checkpoint))


class TestLoadConfig(unittest.TestCase):
    def test_load_valid_config(self):
        config_dict = {
            "dataset_file": "data/test.csv",
            "operations": [{"type": "Shuffle"}],
            "target": "target",
            "splitting": {"train": 0.8, "validation": 0.2},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config_dict, f)

            config = load_config(config_path)
            self.assertEqual(config.dataset_file, "data/test.csv")
            self.assertEqual(config._config_filename, "test_config")

    def test_load_config_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")


if __name__ == "__main__":
    unittest.main()
