#!/usr/bin/env python3
"""
Run metamorphic tests on both models and generate a comparison report.

This script compares how two models respond to metamorphic transformations
of the language proficiency feature.
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

GOOD_MODEL_PATH = "models/goodModel.onnx"  
BAD_MODEL_PATH = "models/badModel.onnx"    
DATA_PATH = "data/investigation_train_large_checked.csv"  
LABEL_COLUMN = "checked"  



def analyze_model_bias(
    model_path: str | Path,
    data_path: str | Path,
    label_column: str,
    model_name: str,
) -> dict:
    """Analyze how a model responds to metamorphic transformations."""
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
        "std_change": changes.std(),
        "min_change": changes.min(),
        "max_change": changes.max(),
    }

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

    # Analyze both models
    good_stats = analyze_model_bias(args.good_model, args.data, args.label_column, "Good Model")
    bad_stats = analyze_model_bias(args.bad_model, args.data, args.label_column, "Bad Model")

    # Print results
    print(f"\n{'Metric':<15} {'Good Model':>12} {'Bad Model':>12} {'Diff':>12}")
    print("-" * 52)
    print(f"{'Mean':<15} {good_stats['mean_change']:>12.4f} {bad_stats['mean_change']:>12.4f} {bad_stats['mean_change']-good_stats['mean_change']:>12.4f}")
    print(f"{'Median':<15} {good_stats['median_change']:>12.4f} {bad_stats['median_change']:>12.4f} {bad_stats['median_change']-good_stats['median_change']:>12.4f}")
    print(f"{'Std':<15} {good_stats['std_change']:>12.4f} {bad_stats['std_change']:>12.4f} {bad_stats['std_change']-good_stats['std_change']:>12.4f}")
    print(f"{'Max':<15} {good_stats['max_change']:>12.4f} {bad_stats['max_change']:>12.4f} {bad_stats['max_change']-good_stats['max_change']:>12.4f}")


if __name__ == "__main__":
    main()
