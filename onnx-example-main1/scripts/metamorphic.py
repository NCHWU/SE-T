#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import onnxruntime as rt

TAALEIS_FEATURE_CANDIDATES = [
    "personeleijke_eigenschappen_taaleis_voldaan",
    "persoonlijke_eigenschappen_taaleis_voldaan",
    "taaleis_voldaan",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply metamorphic testing to ONNX models to detect bias."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the ONNX model file to test.",
    )
    parser.add_argument(
        "--data",
        default=str(Path(__file__).resolve().parents[1] / "data" / "investigation_train_large_checked.csv"),
        help="Path to the CSV dataset.",
    )
    parser.add_argument(
        "--label-column",
        default="checked",
        help="Name of the target column.",
    )
    return parser.parse_args()


def load_inputs(path: str | Path, label_column: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if label_column not in df.columns:
        raise KeyError(f"Label column '{label_column}' not found in dataset.")
    y = df[label_column]
    # Drop label column and leaked prediction columns
    columns_to_drop = [label_column, "Ja", "Nee"]
    columns_to_drop = [c for c in columns_to_drop if c in df.columns]
    X = df.drop(columns=columns_to_drop).astype(np.float32)
    return X, y


def load_onnx_model(model_path: str | Path) -> rt.InferenceSession:
    """Load an ONNX model for inference."""
    return rt.InferenceSession(str(model_path))


def first_present(columns: Iterable[str], frame: pd.DataFrame) -> str | None:
    for column in columns:
        if column in frame.columns:
            return column
    return None


def predict_onnx(session: rt.InferenceSession, X: pd.DataFrame) -> np.ndarray:
    """Get probability predictions from ONNX model."""
    input_name = session.get_inputs()[0].name
    X_array = X.values.astype(np.float32)

    # Run inference - returns [labels, probabilities]
    result = session.run(None, {input_name: X_array})
    probs_dict_list = result[1]  # List of dicts like [{0: 0.1, 1: 0.9}, ...]

    # Extract probability of positive class (key=1) from each dict
    probs_positive = np.array([prob_dict[1] for prob_dict in probs_dict_list])

    return probs_positive


def run_metamorphic_test(
    session: rt.InferenceSession,
    X_test: pd.DataFrame,
    transform_func: Callable[[pd.DataFrame], pd.DataFrame],
    label: str,
) -> dict:
    """
    Run a metamorphic test by applying a transformation and reporting the changes.

    Args:
        session: ONNX inference session
        X_test: Test dataset
        transform_func: Function that applies metamorphic transformation
        label: Name of the test for output

    Returns:
        Dictionary containing statistics about prediction changes
    """
    original_probs = predict_onnx(session, X_test)
    X_transformed = transform_func(X_test.copy())
    transformed_probs = predict_onnx(session, X_transformed)

    changes = np.abs(transformed_probs - original_probs)
    mean_change = changes.mean()
    median_change = np.median(changes)
    max_change = changes.max()
    min_change = changes.min()
    std_change = changes.std()

    stats = {
        "test": label,
        "total_samples": len(changes),
        "mean_change": mean_change,
        "median_change": median_change,
        "std_change": std_change,
        "min_change": min_change,
        "max_change": max_change,
    }

    print(f"\nSamples: {len(changes):,}")
    print(f"Mean: {mean_change:.4f} | Median: {median_change:.4f} | Std: {std_change:.4f}")
    print(f"Min: {min_change:.4f} | Max: {max_change:.4f}")

    return stats


def transform_language_proficiency(X: pd.DataFrame) -> pd.DataFrame:
    """
    Set all language requirement values to 'met' (value 1).

    This tests if the model is invariant to language proficiency:
    - 0 (not met) -> 1 (met)
    - 1 (met) -> 1 (met) [no change]
    - 2 (special) -> 1 (met)

    If the model is fair, changing language status should not affect predictions.
    """
    column = first_present(TAALEIS_FEATURE_CANDIDATES, X)
    if not column:
        return X
    # Set everyone to "met" (value 1)
    X[column] = 1
    return X


def analyze_language_invariance(session: rt.InferenceSession, X_test: pd.DataFrame) -> dict:
    column = first_present(TAALEIS_FEATURE_CANDIDATES, X_test)
    if not column:
        print("Error: Language feature not found")
        return None

    return run_metamorphic_test(
        session, X_test, transform_language_proficiency,
        "language_invariance"
    )


def main() -> None:
    args = parse_args()

    session = load_onnx_model(args.model)
    X_full, _ = load_inputs(args.data, args.label_column)

    # Get model input shape to determine which features it expects
    input_shape = session.get_inputs()[0].shape
    n_features_expected = input_shape[1] if len(input_shape) > 1 else X_full.shape[1]

    # Check if model has different feature count than dataset
    if n_features_expected == 6:
        print("ERROR: Model uses engineered features, cannot run metamorphic test.")
        return

    if n_features_expected < X_full.shape[1]:
        print(f"WARNING: Model has {n_features_expected} features, dataset has {X_full.shape[1]}.")
        print("Sensitive features may have been removed during training.")
        return

    X_test = X_full
    analyze_language_invariance(session, X_test)


if __name__ == "__main__":
    main()
