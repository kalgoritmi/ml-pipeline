"""Unit tests for operations module."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.config import OperationConfig
from src.operations import (
    apply_operation,
    compute_target,
    index_operation,
    limit_rows,
    remove_columns,
    shuffle,
)


class TestIndexOperation(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "Time": pd.date_range("2023-01-01", periods=10, freq="h"),
            "Value": range(10),
        })

    def test_sets_datetime_index(self):
        op = OperationConfig(type="IndexOperation", column="Time")
        result = index_operation(self.df.copy(), op)
        self.assertIsInstance(result.index, pd.DatetimeIndex)

    def test_removes_time_column(self):
        op = OperationConfig(type="IndexOperation", column="Time")
        result = index_operation(self.df.copy(), op)
        self.assertNotIn("Time", result.columns)


class TestRemoveColumns(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": [4, 5, 6],
            "C": [7, 8, 9],
        })

    def test_removes_single_column(self):
        op = OperationConfig(type="RemoveColumns", columns=["A"])
        result = remove_columns(self.df.copy(), op)
        self.assertNotIn("A", result.columns)
        self.assertIn("B", result.columns)

    def test_removes_multiple_columns(self):
        op = OperationConfig(type="RemoveColumns", columns=["A", "B"])
        result = remove_columns(self.df.copy(), op)
        self.assertNotIn("A", result.columns)
        self.assertNotIn("B", result.columns)
        self.assertIn("C", result.columns)

    def test_raises_on_missing_column(self):
        op = OperationConfig(type="RemoveColumns", columns=["NonExistent"])
        with self.assertRaises(KeyError):
            remove_columns(self.df.copy(), op)


class TestComputeTarget(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "A": [10, 20, 30],
            "B": [5, 10, 15],
        })

    def test_creates_target_column(self):
        op = OperationConfig(
            type="ComputeTarget",
            columns=[{"name": "A", "weight": 1.0}],
            target_column="target",
            threshold=15,
        )
        result = compute_target(self.df.copy(), op)
        self.assertIn("target", result.columns)

    def test_binary_values(self):
        op = OperationConfig(
            type="ComputeTarget",
            columns=[{"name": "A", "weight": 1.0}],
            target_column="target",
            threshold=15,
        )
        result = compute_target(self.df.copy(), op)
        self.assertTrue(set(result["target"].unique()).issubset({0, 1}))

    def test_threshold_logic(self):
        op = OperationConfig(
            type="ComputeTarget",
            columns=[{"name": "A", "weight": 1.0}],
            target_column="target",
            threshold=15,
        )
        result = compute_target(self.df.copy(), op)
        # A = [10, 20, 30], threshold=15
        # 10 < 15 -> 0, 20 >= 15 -> 1, 30 >= 15 -> 1
        expected = [0, 1, 1]
        self.assertEqual(list(result["target"]), expected)

    def test_weighted_sum(self):
        op = OperationConfig(
            type="ComputeTarget",
            columns=[
                {"name": "A", "weight": 1.0},
                {"name": "B", "weight": -1.0},
            ],
            target_column="target",
            threshold=0,
        )
        result = compute_target(self.df.copy(), op)
        # A - B = [5, 10, 15], all >= 0 -> [1, 1, 1]
        self.assertEqual(list(result["target"]), [1, 1, 1])


class TestShuffle(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "A": list(range(100)),
        })

    def test_preserves_length(self):
        op = OperationConfig(type="Shuffle", random_state=42)
        result = shuffle(self.df.copy(), op)
        self.assertEqual(len(result), len(self.df))

    def test_changes_order(self):
        op = OperationConfig(type="Shuffle", random_state=42)
        result = shuffle(self.df.copy(), op)
        self.assertFalse(result["A"].equals(self.df["A"]))

    def test_reproducible_with_seed(self):
        op = OperationConfig(type="Shuffle", random_state=42)
        result1 = shuffle(self.df.copy(), op)
        result2 = shuffle(self.df.copy(), op)
        self.assertTrue(result1["A"].equals(result2["A"]))


class TestLimitRows(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "A": list(range(100)),
        })

    def test_limits_rows(self):
        op = OperationConfig(type="LimitRows", n_rows=10)
        result = limit_rows(self.df.copy(), op)
        self.assertEqual(len(result), 10)

    def test_returns_all_when_none(self):
        op = OperationConfig(type="LimitRows", n_rows=None)
        result = limit_rows(self.df.copy(), op)
        self.assertEqual(len(result), 100)

    def test_returns_all_when_larger(self):
        op = OperationConfig(type="LimitRows", n_rows=200)
        result = limit_rows(self.df.copy(), op)
        self.assertEqual(len(result), 100)


class TestApplyOperation(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "A": list(range(10)),
            "B": list(range(10, 20)),
        })

    def test_applies_shuffle(self):
        op = OperationConfig(type="Shuffle", random_state=42)
        result = apply_operation(self.df.copy(), op)
        self.assertEqual(len(result), len(self.df))

    def test_saves_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            op = OperationConfig(type="Shuffle", random_state=42)
            apply_operation(self.df.copy(), op, checkpoint_dir)
            self.assertTrue((checkpoint_dir / "Shuffle.csv").exists())


if __name__ == "__main__":
    unittest.main()
