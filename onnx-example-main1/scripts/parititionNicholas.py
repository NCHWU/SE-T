#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ADDRESS_FEATURE_CANDIDATES = [
    "adres_aantal_wisselingen",
    "adres_aantal_wijzigingen",
    "adres_mutaties_12m",
    "adres_mutaties_24m",
    "adres_aantal_mutaties",
    "adres_aantal_brp_adres",
    "adres_aantal_verschillende_wijken",
    "adres_aantal_verzendadres",
    "adres_aantal_woonadres_handmatig",
]

TAALEIS_FEATURE_CANDIDATES = [
    "personeleijke_eigenschappen_taaleis_voldaan",
    "persoonlijke_eigenschappen_taaleis_voldaan",
    "taaleis_voldaan",
]

NEIGHBORHOOD_PREFIX = "adres_recentste_wijk_"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the notebook model, partition the dataset, and compare predicted scores."
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
    return parser.parse_args()


def load_inputs(path: str | Path, label_column: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = pd.read_csv(path)
    if label_column not in df.columns:
        raise KeyError(f"Label column '{label_column}' not found in dataset.")
    y = df[label_column]
    X = df.drop(columns=[label_column]).astype(np.float32)
    return X, y, df


def train_pipeline(
    X: pd.DataFrame, y: pd.Series, *, test_size: float
) -> tuple[Pipeline, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    probs = pipeline.predict_proba(X)[:, 1]
    test_preds = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, test_preds)
    print(f"Base model accuracy on holdout: {accuracy:.3f}")
    return pipeline, probs, X_test, y_test, test_preds


def first_present(columns: Iterable[str], frame: pd.DataFrame) -> str | None:
    for column in columns:
        if column in frame.columns:
            return column
    return None


def compare_groups(df: pd.DataFrame, mask: pd.Series, label: str) -> None:
    if mask.sum() == 0 or (~mask).sum() == 0:
        print(f"[{label}] skipped (one side empty)")
        return
    group_a = df.loc[mask, "score"]
    group_b = df.loc[~mask, "score"]
    diff = group_a.mean() - group_b.mean()
    print(
        f"[{label}] samples={group_a.shape[0]}/{group_b.shape[0]}, "
        f"mean_scores={group_a.mean():.3f}/{group_b.mean():.3f}, "
        f"mean_diff={diff:.3f}"
    )


def analyze_address_instability(df: pd.DataFrame) -> None:
    column = first_present(ADDRESS_FEATURE_CANDIDATES, df)
    if not column:
        print("[address] feature not found, skipping.")
        return
    values = df[column].fillna(0)
    if values.nunique() <= 1:
        print("[address] not enough variation, skipping.")
        return
    threshold = values.quantile(0.75)
    mask = values >= threshold
    print(f"[address] using column '{column}' threshold={threshold:.3f}")
    compare_groups(df, mask, "address_unstable_vs_rest")


def analyze_taaleis(df: pd.DataFrame) -> None:
    column = first_present(TAALEIS_FEATURE_CANDIDATES, df)
    if not column:
        print("[taaleis] feature not found, skipping.")
        return
    values = df[column].fillna(0)
    if values.dtype == bool:
        met_mask = values
    else:
        met_mask = values.astype(float) > 0.5
    compare_groups(df, ~met_mask, "taaleis_not_met_vs_met")


def analyze_neighborhoods(df: pd.DataFrame) -> None:
    wijk_columns = [c for c in df.columns if c.startswith(NEIGHBORHOOD_PREFIX)]
    if not wijk_columns:
        print("[neighborhood] no wijk columns found, skipping.")
        return

    print("[neighborhood] mean scores per wijk (only showing groups with >=50 samples):")
    scores = df["score"]
    overall_mean = scores.mean()
    for column in wijk_columns:
        mask = (df[column].fillna(0)).astype(int) == 1
        count = mask.sum()
        if count < 50:
            continue
        mean_score = scores[mask].mean()
        diff = mean_score - overall_mean
        wijk_name = column.replace(NEIGHBORHOOD_PREFIX, "")
        print(
            f"  - {wijk_name:<20} count={count:>5} mean={mean_score:.3f} "
            f"(delta vs overall {diff:+.3f})"
        )


def main() -> None:
    args = parse_args()
    X, y, full_df = load_inputs(args.data, args.label_column)
    _, probs, *_ = train_pipeline(X, y, test_size=args.test_size)
    full_df = full_df.copy()
    full_df["score"] = probs

    print("\n=== Partition comparisons ===")
    analyze_address_instability(full_df)
    analyze_taaleis(full_df)
    analyze_neighborhoods(full_df)


if __name__ == "__main__":
    main()

