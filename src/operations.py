"""Data transformation operations."""

from pathlib import Path

import pandas as pd

from .config import ColumnWeight, OperationConfig


def index_operation(df: pd.DataFrame, op: OperationConfig) -> pd.DataFrame:
    df[op.column] = pd.to_datetime(df[op.column])
    return df.set_index(op.column)


def remove_columns(df: pd.DataFrame, op: OperationConfig) -> pd.DataFrame:
    return df.drop(columns=op.columns)


def compute_target(df: pd.DataFrame, op: OperationConfig) -> pd.DataFrame:
    weights = [
        ColumnWeight.model_validate(c) if isinstance(c, dict) else c
        for c in op.columns
    ]
    result = sum(df[w.name] * w.weight for w in weights)
    df[op.target_column] = (result >= op.threshold).astype(int)
    return df


def shuffle(df: pd.DataFrame, op: OperationConfig) -> pd.DataFrame:
    return df.sample(frac=1, random_state=op.random_state)


def limit_rows(df: pd.DataFrame, op: OperationConfig) -> pd.DataFrame:
    if op.n_rows is None:
        return df
    return df.head(op.n_rows)


OPERATIONS = {
    "IndexOperation": index_operation,
    "RemoveColumns": remove_columns,
    "ComputeTarget": compute_target,
    "Shuffle": shuffle,
    "LimitRows": limit_rows,
}


def apply_operation(
    df: pd.DataFrame,
    op: OperationConfig,
    checkpoint_dir: Path | None = None,
) -> pd.DataFrame:
    """Apply a single operation to the dataframe."""
    result = OPERATIONS[op.type](df.copy(), op)

    if checkpoint_dir is not None:
        result.to_csv(checkpoint_dir / f"{op.type}.csv")

    return result
