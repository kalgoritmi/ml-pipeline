"""Main pipeline execution."""

import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from .config import PipelineConfig, load_config
from .operations import apply_operation

logger = logging.getLogger(__name__)


def manual_accuracy(y_true: list, y_pred: list) -> float:
    """Calculate accuracy without sklearn."""
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def split_data(df: pd.DataFrame, train_ratio: float, target: str):
    """Split data into train/validation sets (no sklearn)."""
    split_idx = int(len(df) * train_ratio)
    train, val = df.iloc[:split_idx], df.iloc[split_idx:]
    return (
        train.drop(columns=[target]),
        val.drop(columns=[target]),
        train[target],
        val[target],
    )


def run_pipeline(config: PipelineConfig, base_path: Path) -> dict:
    """Execute the full pipeline."""
    # Ensure dataset exists (download if needed)
    config.ensure_dataset(base_path)

    # Load data
    df = pd.read_csv(base_path / config.dataset_file)
    logger.info("Loaded %d rows", len(df))

    # Apply operations
    for op in config.operations:
        df = apply_operation(df, op, config.checkpoint_dir)
        logger.info("%s: %s", op.type, df.shape)

    # Validate target is binary
    unique_targets = set(df[config.target].unique())
    assert unique_targets.issubset({0, 1}), f"Target must be binary, got {unique_targets}"

    # Split
    X_train, X_val, y_train, y_val = split_data(df, config.splitting.train, config.target)
    logger.info("Train: %d, Val: %d", len(X_train), len(X_val))

    # Train
    model = RandomForestClassifier(random_state=1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    # Evaluate
    sklearn_acc = accuracy_score(y_val, predictions)
    manual_acc = manual_accuracy(y_val.tolist(), predictions.tolist())

    logger.info("sklearn accuracy: %.4f", sklearn_acc)
    logger.info("Manual accuracy: %.4f", manual_acc)

    return {"sklearn_accuracy": sklearn_acc, "manual_accuracy": manual_acc}


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    )

    if len(sys.argv) != 2:
        config_path = Path(__file__).parent.parent / "config.yaml"  # Default path
    else:
        config_path = Path(sys.argv[1])

    config = load_config(config_path)
    run_pipeline(config, config_path.parent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
