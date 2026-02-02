#!/usr/bin/env python3
"""Feast Feature Store demo.

Ported from examples/kfto-feast/kfto_feast.ipynb.
Demonstrates Feast setup, feature retrieval, and data preprocessing for LLM training
using a local SQLite-backed store (no Kubernetes dependencies).

Usage:
    python scripts/01_feast_feature_store.py [--test-mode]
    python scripts/01_feast_feature_store.py --output-dir ./data/feast_output
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")

import pandas as pd


def create_feature_repo(repo_dir):
    """Create a Feast feature repository with driver stats example."""
    from feast import Entity, FeatureView, Field, FileSource, ValueType
    from feast.data_format import ParquetFormat
    from feast.on_demand_feature_view import on_demand_feature_view
    from feast.types import Float32, Float64, Int32

    repo_dir = Path(repo_dir)
    repo_dir.mkdir(parents=True, exist_ok=True)
    data_dir = repo_dir / "data"
    data_dir.mkdir(exist_ok=True)

    # Generate sample driver stats parquet data
    import numpy as np

    np.random.seed(42)
    n_drivers = 5
    n_hours = 360
    records = []
    for driver_id in range(1001, 1001 + n_drivers):
        for h in range(n_hours):
            ts = pd.Timestamp("2021-04-01", tz="UTC") + pd.Timedelta(hours=h)
            records.append(
                {
                    "event_timestamp": ts,
                    "driver_id": driver_id,
                    "conv_rate": np.random.uniform(0, 1),
                    "acc_rate": np.random.uniform(0, 1),
                    "avg_daily_trips": np.random.randint(0, 1000),
                    "created": pd.Timestamp.now(),
                }
            )
    df = pd.DataFrame(records)
    parquet_path = data_dir / "driver_stats.parquet"
    df.to_parquet(str(parquet_path))
    print(f"Generated driver stats: {len(df)} rows -> {parquet_path}")

    # Write feature_store.yaml
    yaml_content = """project: myproject
registry: data/registry.db
provider: local
online_store:
    type: sqlite
    path: data/online_store.db
entity_key_serialization_version: 2
auth:
    type: no_auth
"""
    (repo_dir / "feature_store.yaml").write_text(yaml_content)

    # Write example_repo.py with feature definitions
    repo_py = '''from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int32

driver = Entity(name="driver", join_keys=["driver_id"])

driver_stats_source = FileSource(
    name="driver_hourly_stats_source",
    path="data/driver_stats.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

driver_hourly_stats = FeatureView(
    name="driver_hourly_stats",
    entities=[driver],
    ttl=timedelta(days=1),
    schema=[
        Field(name="conv_rate", dtype=Float32),
        Field(name="acc_rate", dtype=Float32),
        Field(name="avg_daily_trips", dtype=Int32),
    ],
    online=True,
    source=driver_stats_source,
)
'''
    (repo_dir / "example_repo.py").write_text(repo_py)
    (repo_dir / "__init__.py").write_text("")

    return repo_dir


def run_feast_workflow(repo_dir, output_dir):
    """Run the full Feast workflow: apply, retrieve, preprocess."""
    import shutil
    import subprocess
    import sys

    from feast import FeatureStore

    repo_dir = str(repo_dir)

    # Apply feature definitions via the feast CLI
    print("\n--- Applying Feast feature definitions ---")
    feast_bin = shutil.which("feast") or os.path.join(
        os.path.dirname(sys.executable), "feast"
    )
    subprocess.run([feast_bin, "apply"], cwd=repo_dir, check=True)

    store = FeatureStore(repo_path=repo_dir)

    # Retrieve historical features
    print("\n--- Retrieving historical features ---")
    entity_df = pd.DataFrame.from_dict(
        {
            "driver_id": [1001, 1002, 1003, 1004, 1005],
            "event_timestamp": [
                datetime(2021, 4, 12, 10, 59, 42),
                datetime(2021, 4, 12, 8, 12, 10),
                datetime(2021, 4, 12, 16, 40, 26),
                datetime(2021, 4, 12, 12, 30, 0),
                datetime(2021, 4, 12, 14, 15, 30),
            ],
            "label_driver_reported_satisfaction": [1, 5, 3, 4, 2],
            "val_to_add": [1, 2, 3, 4, 5],
            "val_to_add_2": [10, 20, 30, 40, 50],
        }
    )

    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "driver_hourly_stats:conv_rate",
            "driver_hourly_stats:acc_rate",
            "driver_hourly_stats:avg_daily_trips",
        ],
    ).to_df()

    print("\n--- Feature schema ---")
    training_df.info()
    print("\n--- Sample features ---")
    print(training_df.head())

    # Convert to JSONL for LLM training
    print("\n--- Converting to JSONL for LLM training ---")
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, "driver_stats_training.jsonl")
    instruction_text = "Summarize the driver's performance metrics."

    with open(jsonl_path, "w") as f:
        for i in range(len(training_df)):
            record = {
                "driver_id": int(training_df["driver_id"][i]),
                "conv_rate": float(training_df["conv_rate"][i]),
                "acc_rate": float(training_df["acc_rate"][i]),
                "avg_daily_trips": int(training_df["avg_daily_trips"][i]),
            }
            input_text = (
                f"Driver ID: {record['driver_id']}, "
                f"Conversion Rate: {record['conv_rate']:.4f}, "
                f"Acceleration Rate: {record['acc_rate']:.4f}, "
                f"Average Daily Trips: {record['avg_daily_trips']}"
            )
            output_text = (
                f"Driver {record['driver_id']} has a conversion rate of {record['conv_rate']:.2%}, "
                f"an acceleration rate of {record['acc_rate']:.2%}, and completes an average of "
                f"{record['avg_daily_trips']} daily trips."
            )
            example = {
                "instruction": instruction_text,
                "input": input_text,
                "output": output_text,
            }
            f.write(json.dumps(example) + "\n")

    print(f"Training dataset saved to {jsonl_path}")
    return jsonl_path


def main():
    parser = argparse.ArgumentParser(description="Feast Feature Store Demo")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/feast_output",
        help="Directory for output files",
    )
    parser.add_argument(
        "--repo-dir",
        type=str,
        default="/tmp/feast_repo",
        help="Directory for Feast repository",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode (quick validation)",
    )
    args = parser.parse_args()

    from scripts.common.demo_utils import print_intro, suppress_known_warnings

    suppress_known_warnings()
    print_intro(
        "Demo 01: Feast Feature Store",
        "Register ML features in a Feast feature store",
        "Feature stores prevent training/serving skew by centralizing feature definitions",
        "Feature types, entity joins, point-in-time correctness",
    )

    print("=" * 60)
    print("Feast Feature Store Demo")
    print("=" * 60)

    # Create the feature repository
    repo_dir = create_feature_repo(args.repo_dir)

    # Run the full workflow
    jsonl_path = run_feast_workflow(repo_dir, args.output_dir)

    # Verify output
    with open(jsonl_path) as f:
        lines = f.readlines()
    print(f"\nGenerated {len(lines)} training examples")
    print("Sample:", json.loads(lines[0]))

    # ── Metrics, Charts & Validation ──────────────────────────────────────
    from scripts.common.demo_utils import (
        MetricsCollector,
        OutputValidator,
        plot_table,
    )

    mc = MetricsCollector("01_feast_feature_store")
    mc.set_metadata(
        row_count=len(lines),
        output_dir=args.output_dir,
    )
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    mc.save(metrics_path)

    # Summary table of training examples
    table_rows = []
    for line in lines:
        rec = json.loads(line)
        inp = rec.get("input", "")
        # Extract driver ID from input text
        driver_id = inp.split(",")[0].replace("Driver ID: ", "") if inp else ""
        table_rows.append([
            driver_id,
            rec.get("instruction", "")[:40],
            rec.get("output", "")[:50],
        ])
    plot_table(
        ["Driver", "Instruction", "Output (truncated)"],
        table_rows,
        title="Feast Training Examples",
    )

    # Validation
    v = OutputValidator("Feast Feature Store")
    v.check_dir_exists(args.output_dir, "Output directory exists and non-empty")
    v.check_file_min_size(
        jsonl_path, 100, "driver_stats_training.jsonl >= 100 bytes"
    )

    # Verify all lines are valid JSON
    all_valid_json = True
    for i, line in enumerate(lines):
        try:
            json.loads(line)
        except json.JSONDecodeError:
            all_valid_json = False
            break
    v.check(all_valid_json, "All JSONL lines are valid JSON")
    v.check(len(lines) == 5, f"Exactly 5 training examples (got {len(lines)})")
    v.check_file_exists(metrics_path, "metrics.json saved")
    v.print_report()

    print("\nFeast Feature Store Demo completed successfully!")
    print("\n>> Next: Use these features in SFT training (Demo 02) or RAG ingestion (Demo 04)")


if __name__ == "__main__":
    main()
