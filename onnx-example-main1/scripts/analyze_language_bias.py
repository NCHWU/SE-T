#!/usr/bin/env python3
"""
Analyze and visualize language proficiency bias pattern in the Rotterdam welfare dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze language proficiency bias in the dataset."
    )
    parser.add_argument(
        "--data",
        default=str(Path(__file__).resolve().parents[2] / "investigation_train_large_checked.csv"),
        help="Path to the CSV dataset.",
    )
    return parser.parse_args()


def analyze_language_bias(df: pd.DataFrame) -> None:
    """Analyze how language proficiency correlates with being checked."""

    lang_col = "persoonlijke_eigenschappen_taaleis_voldaan"
    target_col = "checked"

    if lang_col not in df.columns:
        print(f"ERROR: Column '{lang_col}' not found")
        return

    # Get base rate
    if df[target_col].dtype == bool:
        checked_count = df[target_col].sum()
        base_rate = checked_count / len(df) * 100
    else:
        checked_count = (df[target_col] == 1).sum()
        base_rate = checked_count / len(df) * 100

    print(f"\nDataset: {len(df):,} records, {checked_count:,} checked ({base_rate:.2f}%)")
    print(f"\n{'Status':<20} {'Count':>10} {'Rate':>10} {'Bias':>10}")
    print("-" * 50)

    # Analyze each language proficiency level
    for val, label in [(0, "Not Met"), (1, "Met"), (2, "Special")]:
        mask = df[lang_col] == val
        count = mask.sum()

        if df[target_col].dtype == bool:
            checked = df[mask][target_col].sum()
        else:
            checked = (df[mask][target_col] == 1).sum()

        rate = checked / count * 100 if count > 0 else 0
        bias_factor = rate / base_rate if base_rate > 0 else 0

        print(f"{label:<20} {count:10,} {rate:9.2f}% {bias_factor:9.2f}x")

    # Calculate disparate impact
    met_rate = 0
    not_met_rate = 0
    for val in [0, 1]:
        mask = df[lang_col] == val
        count = mask.sum()
        if count > 0:
            if df[target_col].dtype == bool:
                checked = df[mask][target_col].sum()
            else:
                checked = (df[mask][target_col] == 1).sum()
            rate = checked / count * 100
            if val == 1:
                met_rate = rate
            else:
                not_met_rate = rate

    if met_rate > 0:
        disparate_impact = not_met_rate / met_rate
        print(f"\nDisparate Impact: {disparate_impact:.2f}x")
        if disparate_impact > 1.25:
            print("WARNING: Exceeds fairness threshold (1.25x)")




def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data)
    analyze_language_bias(df)
    print()


if __name__ == "__main__":
    main()
