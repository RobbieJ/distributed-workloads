#!/usr/bin/env python3
"""Hyperparameter Optimization with Optuna.

Ported from examples/hpo-raytune/notebook/raytune-oai-demo.ipynb.
Replaces Ray Tune with Optuna for single-machine HPO.
Trains a simple neural network with hyperparameter search, exports best model
to ONNX format, and optionally uploads to MinIO.

Usage:
    python scripts/08_hpo_optuna.py [--test-mode]
    python scripts/08_hpo_optuna.py --n-trials 20 --output-dir ./models/hpo
"""

import argparse
import math
import os
import sys

# GPU setup
sys.path.insert(0, ".")
from scripts.common.gpu_utils import setup_cuda_env

setup_cuda_env()

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class SimpleNet(nn.Module):
    """Simple feedforward neural network for HPO demonstration."""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def make_sine_data(device):
    """Generate sine-wave regression dataset (deterministic)."""
    torch.manual_seed(42)
    X = torch.linspace(0, 2 * math.pi, 200).unsqueeze(1).to(device)
    y = (torch.sin(X) + 0.1 * torch.randn_like(X)).to(device)
    return X, y


def objective(trial, input_size=1, output_size=1, n_epochs=10):
    """Optuna objective function for HPO."""
    hidden_size = trial.suggest_categorical("hidden_size", [5, 10, 20])
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNet(input_size, hidden_size, output_size).to(device)

    X, y = make_sine_data(device)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(n_epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Report intermediate value for pruning
        trial.report(loss.item(), epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # Final evaluation
    model.eval()
    with torch.no_grad():
        predictions = model(X)
        final_loss = criterion(predictions, y).item()

    return final_loss


def export_onnx(model, output_path, input_size=1):
    """Export a trained model to ONNX format."""
    dummy_input = torch.randn(1, input_size)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Model exported to {output_path}")


def upload_to_minio(local_dir, bucket="models", prefix="hpo"):
    """Upload model directory to MinIO (if available)."""
    try:
        from scripts.common.storage_utils import get_s3_client, upload_directory

        client = get_s3_client()
        upload_directory(local_dir, bucket, prefix, client)
        print(f"Uploaded to MinIO: s3://{bucket}/{prefix}")
    except Exception as e:
        print(f"MinIO upload skipped: {e}")


def main():
    parser = argparse.ArgumentParser(description="HPO with Optuna")
    parser.add_argument(
        "--n-trials", type=int, default=10, help="Number of HPO trials"
    )
    parser.add_argument(
        "--n-epochs", type=int, default=10, help="Training epochs per trial"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/hpo",
        help="Output directory for best model",
    )
    parser.add_argument(
        "--upload", action="store_true", help="Upload best model to MinIO"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode (2 trials, 3 epochs)",
    )
    args = parser.parse_args()

    if args.test_mode:
        args.n_trials = 2
        args.n_epochs = 3

    from scripts.common.demo_utils import print_intro, suppress_known_warnings

    suppress_known_warnings()
    print_intro(
        "Demo 08: Hyperparameter Optimization with Optuna",
        "Automatic hyperparameter optimization with Optuna",
        "HPO finds better hyperparameters than manual guessing",
        "Best trial vs default params, search space exploration",
    )

    print("=" * 60)
    print("Hyperparameter Optimization with Optuna")
    print("=" * 60)
    print(f"Trials: {args.n_trials}, Epochs per trial: {args.n_epochs}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Baseline: train with naive default hyperparams ─────────────────────
    print("\n--- Baseline: training with default hyperparams (sine-wave regression) ---")
    default_hidden, default_lr = 10, 0.01
    baseline_model = SimpleNet(1, default_hidden, 1).to(device)
    X_bl, y_bl = make_sine_data(device)
    bl_optimizer = optim.Adam(baseline_model.parameters(), lr=default_lr)
    bl_criterion = nn.MSELoss()
    for _ep in range(args.n_epochs):
        bl_optimizer.zero_grad()
        bl_out = baseline_model(X_bl)
        bl_loss = bl_criterion(bl_out, y_bl)
        bl_loss.backward()
        bl_optimizer.step()
    baseline_model.eval()
    with torch.no_grad():
        baseline_loss = bl_criterion(baseline_model(X_bl), y_bl).item()
    print(f"Baseline loss (hidden={default_hidden}, lr={default_lr}): {baseline_loss:.6f}")
    del baseline_model, X_bl, y_bl, bl_optimizer
    torch.cuda.empty_cache()

    # Create and run the study
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3),
    )

    study.optimize(
        lambda trial: objective(trial, n_epochs=args.n_epochs),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    # Results
    print(f"\nBest trial:")
    best_trial = study.best_trial
    print(f"  Loss: {best_trial.value:.6f}")
    print(f"  Params: {best_trial.params}")

    # Recreate and train the best model
    print("\n--- Training best model ---")
    best_model = SimpleNet(1, best_trial.params["hidden_size"], 1).to(device)
    X, y = make_sine_data(device)
    optimizer = optim.Adam(best_model.parameters(), lr=best_trial.params["lr"])
    criterion = nn.MSELoss()

    for epoch in range(args.n_epochs):
        optimizer.zero_grad()
        outputs = best_model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    # Save and export
    os.makedirs(args.output_dir, exist_ok=True)

    # Save PyTorch model
    torch_path = os.path.join(args.output_dir, "best_model.pt")
    torch.save(best_model.state_dict(), torch_path)
    print(f"PyTorch model saved to {torch_path}")

    # Export ONNX
    onnx_path = os.path.join(args.output_dir, "model.onnx")
    best_model.cpu()
    export_onnx(best_model, onnx_path)

    # Validate ONNX model
    import onnx
    from onnxruntime import InferenceSession

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation passed")

    session = InferenceSession(onnx_path)
    test_input = np.random.randn(1, 1).astype(np.float32)
    result = session.run(None, {"input": test_input})
    print(f"ONNX inference test: input shape {test_input.shape} -> output shape {result[0].shape}")

    # ── Metrics, Charts & Validation ──────────────────────────────────────
    from scripts.common.demo_utils import (
        BaselineComparator,
        MetricsCollector,
        OutputValidator,
        plot_bar_chart,
        plot_comparison_bar,
        plot_table,
    )

    mc = MetricsCollector("08_hpo_optuna")
    mc.set_metadata(n_trials=args.n_trials, n_epochs=args.n_epochs)
    for trial in study.trials:
        mc.add_trial(
            trial_number=trial.number,
            params=trial.params,
            value=trial.value if trial.value is not None else None,
            state=str(trial.state),
        )
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    mc.save(metrics_path)

    # Charts
    completed_trials = [t for t in study.trials if t.value is not None]
    if completed_trials:
        trial_labels = [f"T{t.number}" for t in completed_trials]
        trial_values = [t.value for t in completed_trials]
        plot_bar_chart(trial_labels, trial_values, title="Trial Losses")

        table_rows = []
        for t in completed_trials:
            table_rows.append([
                t.number,
                t.params.get("hidden_size", ""),
                f"{t.params.get('lr', 0):.6f}",
                f"{t.value:.6f}",
                str(t.state),
            ])
        plot_table(
            ["Trial", "Hidden", "LR", "Loss", "State"],
            table_rows,
            title="HPO Trial Summary",
        )

    # Hyperparameter comparison table (not quality metrics)
    plot_table(
        ["Hyperparameter", "Default", "Optuna Best"],
        [
            ["Hidden Size", str(default_hidden), str(best_trial.params.get("hidden_size", "?"))],
            ["Learning Rate", f"{default_lr:.6f}", f"{best_trial.params.get('lr', 0):.6f}"],
        ],
        title="Hyperparameter Comparison",
    )

    # Baseline comparison (quality metric only)
    bc = BaselineComparator("HPO: Default vs Optuna")
    bc.add_metric("Loss", baseline_loss, best_trial.value, lower_is_better=True)
    bc.render()
    plot_comparison_bar(
        ["Loss"],
        [baseline_loss],
        [best_trial.value],
        title="Default vs Optuna Best Loss",
    )
    mc.set_metadata(baseline_comparison=bc.to_dict())

    # Validation
    v = OutputValidator("HPO with Optuna")
    v.check_dir_exists(args.output_dir, "Output directory exists and non-empty")
    v.check_file_min_size(torch_path, 100, "best_model.pt >= 100 bytes")
    v.check_file_min_size(onnx_path, 100, "model.onnx >= 100 bytes")
    v.check_metric_range(
        best_trial.value, 0.0, 100.0, f"Best loss {best_trial.value:.4f} in [0, 100]"
    )
    if len(completed_trials) >= 2:
        worst_value = max(t.value for t in completed_trials)
        v.check(
            best_trial.value <= worst_value,
            "Best trial loss <= worst trial loss",
        )
    v.check_file_exists(metrics_path, "metrics.json saved")
    bc.add_validations(v)
    v.print_report()

    # Upload to MinIO if requested
    if args.upload:
        upload_to_minio(args.output_dir)

    print("\nHPO with Optuna completed successfully!")
    print("\n>> Tip: Apply these optimized hyperparameters to any training demo")


if __name__ == "__main__":
    main()
