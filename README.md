# ML Pipeline

Config-driven ML pipeline for binary classification on time-series power consumption data.

## Quick Start

```bash
poetry install
python -m src.pipeline config.yaml
```

## Architecture

```bash
src/
├── config.py      # Pydantic schemas, YAML loading, dataset download
├── operations.py  # Data transformations (index, remove, compute, shuffle, limit)
└── pipeline.py    # Orchestration, train/val split, model training
```

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| Single file operations | Avoided over engineering; all ops are <10 lines each |
| Dictionary dispatch | Simpler than class registry pattern for 5 operations |
| Pydantic validation | Type safety + clear error messages at config load time |
| Manual train/val split | As an exercise to sklearn for splitting |

## Configuration

```yaml
version: '1.0'
dataset_file: 'data/CLEAN_House1.csv'
dataset_url: 'https://...'  # auto downloads if file missing

operations:
  - type: 'LimitRows'
    n_rows: 10000

  - type: 'IndexOperation'
    column: 'Time'

  - type: 'RemoveColumns'
    columns: ['Unix', 'Aggregate', 'Issues']

  - type: 'ComputeTarget'
    columns:
      - name: 'Appliance1'
        weight: 5.0
      - name: 'Appliance2'
        weight: -7.0
    target_column: 'Custom Target'
    threshold: 0  # weighted_sum >= threshold -> 1, else 0

  - type: 'Shuffle'
    random_state: 42

target: 'Custom Target'
splitting:
  train: 0.8
  validation: 0.2
checkpoint_path: 'checkpoints'  # Optional: saves CSV after each operation
```

### Operations

| Type | Parameters | Description |
|------|------------|-------------|
| `LimitRows` | `n_rows` | Truncate dataset to first N rows |
| `IndexOperation` | `column` | Set datetime column as index |
| `RemoveColumns` | `columns` | Drop specified columns |
| `ComputeTarget` | `columns`, `target_column`, `threshold` | Create binary target from weighted sum |
| `Shuffle` | `random_state` | Randomize row order |

## Assumptions

1. **Binary classification only** - Target must be 0/1 after ComputeTarget or existing column
2. **Temporal ordering irrelevant** - Shuffle is valid; no timeseries forecasting
3. **Dataset fits in memory** - No chunked processing
4. **CSV format** - Input and checkpoints are CSV

## Testing

```bash
# Unit tests (config, operations)
python -m unittest discover tests/ -v

# Integration test (requires one-time dataset download)
python -m unittest tests.test_pipeline -v
```

### Test Coverage

| Module | Coverage |
|--------|----------|
| config.py | Validation, checkpoint dir creation, file loading |
| operations.py | All 5 operations, checkpoint saving |
| pipeline.py | End-to-end accuracy threshold, checkpoint count |

## Output

```bash
[timestamp] [src.pipeline] [INFO] Loaded 10000 rows
[timestamp] [src.pipeline] [INFO] LimitRows: (10000, 11)
[timestamp] [src.pipeline] [INFO] IndexOperation: (10000, 10)
[timestamp] [src.pipeline] [INFO] RemoveColumns: (10000, 7)
[timestamp] [src.pipeline] [INFO] ComputeTarget: (10000, 8)
[timestamp] [src.pipeline] [INFO] Shuffle: (10000, 8)
[timestamp] [src.pipeline] [INFO] Train: 8000, Val: 2000
[timestamp] [src.pipeline] [INFO] sklearn accuracy: 0.XXXX
[timestamp] [src.pipeline] [INFO] Manual accuracy: 0.XXXX
```

Checkpoints saved to: `checkpoints/{config_name}/{version}/{timestamp}/`

## Experiment Tracking

Lightweight experiment tracking is built-in through:

1. **YAML versioning** - Bump `version` field when changing config parameters
2. **Timestamped checkpoint dirs** - Each run creates a unique directory

```sh
checkpoints/
└── config/
    ├── 1.0/
    │   ├── 20240115_143022/
    │   │   ├── LimitRows.csv
    │   │   ├── IndexOperation.csv
    │   │   └── ...
    │   └── 20240115_151245/
    └── 1.1/
        └── 20240116_092033/
```

This allows comparing intermediate results across runs without external tools.

## Docker

```bash
# Build (runs tests during build)
docker build -t ml-pipeline .

# Run with default config
docker run ml-pipeline

# Run with custom config and persist outputs
docker run -v $(pwd)/config.yaml:/app/config.yaml:ro \
           -v $(pwd)/data:/app/data \
           -v $(pwd)/checkpoints:/app/checkpoints \
           ml-pipeline
```

### Production Features

- Multi-stage build (smaller image, no dev dependencies)
- Non-root user
- Tests run at build time (fail-fast)
- Layer caching for dependencies

## Dependencies

- pandas: DataFrame operations
- scikit-learn: RandomForestClassifier, accuracy_score
- pydantic: Config validation
- pyyaml: YAML parsing
