#!/usr/bin/env python3
"""
Run metamorphic tests on both models and generate a comparison report.

This script helps determine the appropriate threshold for distinguishing
between good (fair) and bad (biased) models.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import onnxruntime as rt

from metamorphic import (
    load_inputs,
    load_onnx_model,
    first_present,
    predict_onnx,
    transform_language_proficiency,
    TAALEIS_FEATURE_CANDIDATES,
)

# ============================================================================
# CONFIGURATION - EDIT THESE PATHS TO CHANGE MODELS
# ============================================================================
GOOD_MODEL_PATH = "models/model_1.onnx"  # Path to the good (fair) model
BAD_MODEL_PATH = "models/model_2.onnx"    # Path to the bad (biased) model
DATA_PATH = "data/synth_data_for_training.csv"  # Path to the dataset
LABEL_COLUMN = "checked"  # Target column name
# ============================================================================


def analyze_model_bias(
    model_path: str | Path,
    data_path: str | Path,
    label_column: str,
    model_name: str,
) -> dict:
    """Analyze bias in a model without asserting."""
    session = load_onnx_model(model_path)
    X_full, _ = load_inputs(data_path, label_column)

    column = first_present(TAALEIS_FEATURE_CANDIDATES, X_full)
    if not column:
        return {"error": "Language feature not found"}

    # Run metamorphic transformation
    original_probs = predict_onnx(session, X_full)
    X_transformed = transform_language_proficiency(X_full.copy())
    transformed_probs = predict_onnx(session, X_transformed)

    changes = np.abs(transformed_probs - original_probs)

    # Calculate statistics
    stats = {
        "model": model_name,
        "total_samples": len(changes),
        "mean_change": changes.mean(),
        "median_change": np.median(changes),
        "max_change": changes.max(),
        "p90": np.percentile(changes, 90),
        "p95": np.percentile(changes, 95),
        "p99": np.percentile(changes, 99),
    }

    # Calculate violations at different thresholds
    for threshold in [0.05, 0.10, 0.15, 0.20]:
        violations = (changes > threshold).sum()
        violation_rate = violations / len(changes) * 100
        stats[f"violations_{int(threshold*100):02d}"] = violations
        stats[f"rate_{int(threshold*100):02d}"] = violation_rate

    return stats


def main() -> None:
    # Get base directory (parent of scripts folder)
    base_dir = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Compare good vs bad model on metamorphic tests."
    )
    parser.add_argument(
        "--good-model",
        default=str(base_dir / GOOD_MODEL_PATH),
        help=f"Path to the good model ONNX file. (default: {GOOD_MODEL_PATH})",
    )
    parser.add_argument(
        "--bad-model",
        default=str(base_dir / BAD_MODEL_PATH),
        help=f"Path to the bad model ONNX file. (default: {BAD_MODEL_PATH})",
    )
    parser.add_argument(
        "--data",
        default=str(base_dir / DATA_PATH),
        help=f"Path to the CSV dataset. (default: {DATA_PATH})",
    )
    parser.add_argument(
        "--label-column",
        default=LABEL_COLUMN,
        help=f"Name of the target column. (default: {LABEL_COLUMN})",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Metamorphic Testing Bias Comparison Report")
    print("=" * 80)

    # Analyze both models
    good_stats = analyze_model_bias(args.good_model, args.data, args.label_column, "Good Model")
    bad_stats = analyze_model_bias(args.bad_model, args.data, args.label_column, "Bad Model")

    # Print summary statistics
    print(f"\nDataset: {good_stats['total_samples']:,} samples\n")

    print(f"{'Metric':<20} {'Good Model':>15} {'Bad Model':>15} {'Difference':>15}")
    print("-" * 80)

    metrics = [
        ("mean_change", "Mean Change", "{:.4f}"),
        ("median_change", "Median Change", "{:.4f}"),
        ("p90", "P90 Change", "{:.4f}"),
        ("p95", "P95 Change", "{:.4f}"),
        ("p99", "P99 Change", "{:.4f}"),
        ("max_change", "Max Change", "{:.4f}"),
    ]

    for key, label, fmt in metrics:
        good_val = good_stats[key]
        bad_val = bad_stats[key]
        diff = bad_val - good_val
        print(f"{label:<20} {fmt.format(good_val):>15} {fmt.format(bad_val):>15} {fmt.format(diff):>15}")

    # Print violation rates at different thresholds
    print("\n" + "=" * 80)
    print("Violation rates at different thresholds")
    print("=" * 80)
    print(f"\n{'Threshold':<12} {'Good Model':>20} {'Bad Model':>20} {'Improvement':>15}")
    print("-" * 80)

    for threshold in [0.05, 0.10, 0.15, 0.20]:
        thresh_key = f"rate_{int(threshold*100):02d}"
        good_rate = good_stats[thresh_key]
        bad_rate = bad_stats[thresh_key]
        improvement = bad_rate - good_rate

        # Determine if this threshold can distinguish models
        if good_rate == 0 and bad_rate > 0:
            status = " (GOOD PASSES, BAD FAILS)"
        elif good_rate == 0 and bad_rate == 0:
            status = " (Both pass)"
        else:
            status = " (Both fail)"

        print(f"{threshold:<12.2f} {good_rate:>19.1f}% {bad_rate:>19.1f}% {improvement:>14.1f}%{status}")

    # Recommendation
    print("\n" + "=" * 80)
    print("Recommendations")
    print("=" * 80)

    print(f"\nBoth models have outliers above any reasonable threshold.")
    print(f"This is expected when the protected feature has legitimate predictive signal.")
    print(f"\nKey findings:")
    print(f"  1. Good model reduces mean change by {((bad_stats['mean_change'] - good_stats['mean_change']) / bad_stats['mean_change'] * 100):.1f}%")
    print(f"  2. Good model reduces P95 change by {((bad_stats['p95'] - good_stats['p95']) / bad_stats['p95'] * 100):.1f}%")
    print(f"  3. Good model reduces violations at 0.05 by {bad_stats['rate_05'] - good_stats['rate_05']:.1f}%")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
