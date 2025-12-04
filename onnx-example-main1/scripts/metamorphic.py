#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

TAALEIS_FEATURE_CANDIDATES = [
    "personeleijke_eigenschappen_taaleis_voldaan",
    "persoonlijke_eigenschappen_taaleis_voldaan",
    "taaleis_voldaan",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the notebook model and apply metamorphic testing to detect bias."
    )
    parser.add_argument(
        "--data",
        default=str(Path(__file__).resolve().parents[1] / "data" / "synth_data_for_training.csv"),
        help="Path to the CSV dataset.",
    )
    parser.add_argument(
        "--label-column",
        default="checked",
        help="Name of the target column.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Fraction reserved for evaluating the base model.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Maximum acceptable prediction change for metamorphic tests.",
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


def train_pipeline(
    X: pd.DataFrame, y: pd.Series, *, test_size: float
) -> tuple[Pipeline, pd.DataFrame, pd.Series]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    pipeline = Pipeline(
        steps=[
            ("variance_threshold", VarianceThreshold()),
            (
                "gb",
                GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=1.0,
                    max_depth=1,
                    random_state=0,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    test_preds = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, test_preds)
    print(f"Base model accuracy on holdout: {accuracy:.3f}")
    return pipeline, X_test, y_test


def first_present(columns: Iterable[str], frame: pd.DataFrame) -> str | None:
    for column in columns:
        if column in frame.columns:
            return column
    return None


def run_metamorphic_test(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    transform_func: Callable[[pd.DataFrame], pd.DataFrame],
    label: str,
    threshold: float,
) -> None:
    """
    Run a metamorphic test by applying a transformation and asserting predictions should not change.

    Args:
        pipeline: Trained model pipeline
        X_test: Test dataset
        transform_func: Function that applies metamorphic transformation
        label: Name of the test for output
        threshold: Maximum acceptable change in prediction

    Raises:
        AssertionError: If any prediction changes by more than threshold
    """
    original_probs = pipeline.predict_proba(X_test)[:, 1]
    X_transformed = transform_func(X_test.copy())
    transformed_probs = pipeline.predict_proba(X_transformed)[:, 1]

    changes = np.abs(transformed_probs - original_probs)
    violations = (changes > threshold).sum()
    violation_rate = violations / len(changes) * 100
    mean_change = changes.mean()
    max_change = changes.max()

    # Calculate percentiles
    p50 = np.percentile(changes, 50)
    p95 = np.percentile(changes, 95)

    print(
        f"[{label}] violations={violations}/{len(changes)} ({violation_rate:.1f}%), "
        f"mean_change={mean_change:.3f}, max_change={max_change:.3f}, "
        f"p50={p50:.3f}, p95={p95:.3f}"
    )

    # Assert that predictions should not change
    assert violations == 0, (
        f"Metamorphic test '{label}' failed: {violations}/{len(changes)} predictions "
        f"changed by more than {threshold:.3f} (violation_rate={violation_rate:.1f}%, "
        f"max_change={max_change:.3f})"
    )


def transform_language_proficiency(X: pd.DataFrame) -> pd.DataFrame:
    """Flip language requirement status (met <-> not met)."""
    column = first_present(TAALEIS_FEATURE_CANDIDATES, X)
    if not column:
        return X
    # Flip binary values: 0 -> 1, 1 -> 0
    X[column] = 1 - X[column].fillna(0)
    return X


def analyze_language_invariance(pipeline: Pipeline, X_test: pd.DataFrame, threshold: float) -> None:
    column = first_present(TAALEIS_FEATURE_CANDIDATES, X_test)
    if not column:
        print("[language_invariance] feature not found, skipping.")
        return

    print(f"[language_invariance] using column '{column}'")
    run_metamorphic_test(
        pipeline, X_test, transform_language_proficiency,
        "language_invariance", threshold
    )


def main() -> None:
    args = parse_args()
    X, y = load_inputs(args.data, args.label_column)
    pipeline, X_test, y_test = train_pipeline(X, y, test_size=args.test_size)

    print(f"\n=== Metamorphic tests (threshold={args.threshold:.3f}) ===")
    analyze_language_invariance(pipeline, X_test, args.threshold)
    print("\n✓ All metamorphic tests passed!")


if __name__ == "__main__":
    main()
