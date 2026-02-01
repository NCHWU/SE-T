#!/usr/bin/env python3
"""
Bias detection tests using metamorphic and partition-based testing.

This module implements two categories of bias detection tests:

1. **Metamorphic Tests**: Verify prediction invariance when protected attributes
   are modified. A fair model should produce similar predictions regardless of
   protected attribute values.

2. **Partition Tests**: Compare prediction distributions across demographic groups.
   Large disparities between groups indicate potential bias.

Example
-------
Run all tests on a model::

    from test_models import run_all_tests
    results = run_all_tests("models/goodModel.onnx", "data/test.csv")

Run individual tests::

    from test_models import metamorphic_gender_invariance, load_data
    X, _ = load_data("data/test.csv")
    result = metamorphic_gender_invariance("models/goodModel.onnx", X)
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import onnxruntime as rt

from constants import (
    NEIGHBORHOOD_PREFIX,
    GENDER_COL,
    LANG_TAALEIS_CANDIDATES,
    ADDRESS_FEATURE_CANDIDATES,
    FINANCIAL_PROBLEM_COLS,
)
from data_utils import load_data


def load_onnx_model(model_path: str | Path) -> rt.InferenceSession:
    """
    Load an ONNX model for inference.

    Parameters
    ----------
    model_path : str or Path
        Path to the ONNX model file.

    Returns
    -------
    rt.InferenceSession
        ONNX runtime session ready for inference.
    """
    return rt.InferenceSession(str(model_path))


def first_present(candidates: list[str], frame: pd.DataFrame) -> str | None:
    """
    Find the first column name from candidates that exists in the dataframe.

    Parameters
    ----------
    candidates : list of str
        Column names to search for, in order of preference.
    frame : pd.DataFrame
        DataFrame to search in.

    Returns
    -------
    str or None
        First matching column name, or None if no match found.
    """
    for col in candidates:
        if col in frame.columns:
            return col
    return None


def predict_onnx(session: rt.InferenceSession, X: pd.DataFrame) -> np.ndarray:
    """
    Get probability predictions from an ONNX model.

    Parameters
    ----------
    session : rt.InferenceSession
        Loaded ONNX runtime session.
    X : pd.DataFrame
        Feature matrix.

    Returns
    -------
    np.ndarray
        Probability of positive class for each sample, shape (n_samples,).

    Notes
    -----
    Handles multiple ONNX output formats:
    - Dict format: [{0: prob0, 1: prob1}, ...]
    - Array format: [[prob0, prob1], ...]
    """
    input_name = session.get_inputs()[0].name
    X_array = X.values.astype(np.float32)

    result = session.run(None, {input_name: X_array})

    # Handle different output formats
    if isinstance(result[1], list) and isinstance(result[1][0], dict):
        probs_positive = np.array([prob_dict.get(1, 0) for prob_dict in result[1]])
    elif isinstance(result[1], np.ndarray) and result[1].ndim == 2:
        probs_positive = result[1][:, 1]
    else:
        probs_positive = np.array(result[1])

    return probs_positive


# ============================================================================
# METAMORPHIC TESTS
# ============================================================================

def metamorphic_language_invariance(
    model_path: str | Path,
    X: pd.DataFrame,
) -> dict:
    """
    Test language proficiency invariance.
    
    Transformation: Compare predictions when everyone has "met" vs "not met".
    Expected: Fair models show minimal difference between the two scenarios.
    Biased models will predict differently based on language proficiency.
    """
    session = load_onnx_model(model_path)
    
    # Transform: set language to "met" for everyone
    lang_col = first_present(LANG_TAALEIS_CANDIDATES, X)
    if not lang_col:
        return {"error": "Language feature not found"}
    
    # Version 1: Everyone has "met" (proficient)
    X_met = X.copy()
    X_met[lang_col] = 1
    probs_met = predict_onnx(session, X_met)
    
    # Version 2: Everyone has "not met" (not proficient)
    X_not_met = X.copy()
    X_not_met[lang_col] = 0
    probs_not_met = predict_onnx(session, X_not_met)
    
    # Calculate difference between the two scenarios
    # A fair model should give similar predictions regardless of language
    # A biased model will show large differences
    changes = np.abs(probs_met - probs_not_met)
    
    return {
        "test": "Language Invariance",
        "mean": changes.mean(),
        "median": np.median(changes),
        "std": changes.std(),
        "min": changes.min(),
        "max": changes.max(),
        "samples": len(changes),
    }


def metamorphic_address_instability_invariance(
    model_path: str | Path,
    X: pd.DataFrame,
) -> dict:
    """
    Test address instability invariance.
    
    Transformation: Set all address change columns to 0 (stable addresses).
    Expected: Fair models show minimal prediction changes.
    """
    session = load_onnx_model(model_path)
    
    # Get original predictions
    original_probs = predict_onnx(session, X)
    
    # Find ALL address change columns that exist
    addr_cols = [col for col in ADDRESS_FEATURE_CANDIDATES if col in X.columns]
    if not addr_cols:
        return {"error": "No address instability features found"}
    
    X_transformed = X.copy()
    # Set ALL address change columns to 0 (stable address)
    for col in addr_cols:
        X_transformed[col] = 0
    
    # Get transformed predictions
    transformed_probs = predict_onnx(session, X_transformed)
    
    # Calculate changes
    changes = np.abs(transformed_probs - original_probs)
    
    return {
        "test": "Address Instability Invariance",
        "mean": changes.mean(),
        "median": np.median(changes),
        "std": changes.std(),
        "min": changes.min(),
        "max": changes.max(),
        "samples": len(changes),
    }


def metamorphic_neighborhood_invariance(
    model_path: str | Path,
    X: pd.DataFrame,
) -> dict:
    """
    Test neighborhood invariance.
    
    Transformation: Set all neighborhood indicators to neutral (set everyone to 'other').
    Expected: Fair models show minimal prediction changes.
    """
    session = load_onnx_model(model_path)
    
    # Get original predictions
    original_probs = predict_onnx(session, X)
    
    # Transform: set all neighborhood columns to neutral 'other'
    wijk_cols = [c for c in X.columns if c.startswith(NEIGHBORHOOD_PREFIX)]
    if not wijk_cols:
        return {"error": "Neighborhood features not found"}
    
    X_transformed = X.copy()
    # Set all to 0, then set 'other' to 1 (neutral neighborhood)
    for col in wijk_cols:
        X_transformed[col] = 0
    other_col = NEIGHBORHOOD_PREFIX + "other"
    if other_col in X_transformed.columns:
        X_transformed[other_col] = 1
    
    # Get transformed predictions
    transformed_probs = predict_onnx(session, X_transformed)
    
    # Calculate changes
    changes = np.abs(transformed_probs - original_probs)
    
    return {
        "test": "Neighborhood Invariance",
        "mean": changes.mean(),
        "median": np.median(changes),
        "std": changes.std(),
        "min": changes.min(),
        "max": changes.max(),
        "samples": len(changes),
    }


def metamorphic_gender_invariance(
    model_path: str | Path,
    X: pd.DataFrame,
) -> dict:
    """
    Test gender invariance.
    
    Transformation: Flip gender (male ↔ female).
    Expected: Fair models show minimal prediction changes.
    """
    session = load_onnx_model(model_path)
    
    # Get original predictions
    original_probs = predict_onnx(session, X)
    
    # Transform: flip gender
    if GENDER_COL not in X.columns:
        return {"error": "Gender feature not found"}
    
    X_transformed = X.copy()
    X_transformed[GENDER_COL] = 1 - X_transformed[GENDER_COL]  # Flip: 0 -> 1, 1 -> 0
    
    # Get transformed predictions
    transformed_probs = predict_onnx(session, X_transformed)
    
    # Calculate changes
    changes = np.abs(transformed_probs - original_probs)
    
    return {
        "test": "Gender Invariance",
        "mean": changes.mean(),
        "median": np.median(changes),
        "std": changes.std(),
        "min": changes.min(),
        "max": changes.max(),
        "samples": len(changes),
    }


def metamorphic_financial_stability_invariance(
    model_path: str | Path,
    X: pd.DataFrame,
) -> dict:
    """
    Test financial stability invariance.
    
    Transformation: Set all financial problems to 0 (no financial issues).
    Expected: Fair models show minimal prediction changes.
    """
    session = load_onnx_model(model_path)
    
    # Get original predictions
    original_probs = predict_onnx(session, X)
    
    # Find financial problem columns
    fin_cols = [col for col in FINANCIAL_PROBLEM_COLS if col in X.columns]
    if not fin_cols:
        return {"error": "No financial problem features found"}
    
    X_transformed = X.copy()
    # Set all financial problem indicators to 0 (financially stable)
    for col in fin_cols:
        X_transformed[col] = 0
    
    # Get transformed predictions
    transformed_probs = predict_onnx(session, X_transformed)
    
    # Calculate changes
    changes = np.abs(transformed_probs - original_probs)
    
    return {
        "test": "Financial Stability Invariance",
        "mean": changes.mean(),
        "median": np.median(changes),
        "std": changes.std(),
        "min": changes.min(),
        "max": changes.max(),
        "samples": len(changes),
    }


# ============================================================================
# PARTITION TESTS
# ============================================================================

def partition_language_proficiency(
    model_path: str | Path,
    X: pd.DataFrame,
    df_original: pd.DataFrame,
) -> dict:
    """
    Partition test: Language proficiency.
    
    Splits data by language proficiency and compares model prediction scores.
    Large differences suggest language-based bias.
    """
    session = load_onnx_model(model_path)
    probs = predict_onnx(session, X)
    
    lang_col = first_present(LANG_TAALEIS_CANDIDATES, df_original)
    if not lang_col:
        return {"error": "Language feature not found"}
    
    lang_vals = df_original[lang_col].fillna(0)
    
    results = {
        "test": "Language Proficiency",
        "groups": {}
    }
    
    for val, label in [(0, "Not Met"), (1, "Met"), (2, "Special")]:
        mask = lang_vals == val
        count = mask.sum()
        
        if count > 0:
            scores = probs[mask.values]
            results["groups"][label] = {
                "count": count,
                "mean_score": scores.mean(),
                "std_score": scores.std(),
            }
    
    return results


def partition_address_instability(
    model_path: str | Path,
    X: pd.DataFrame,
    df_original: pd.DataFrame,
) -> dict:
    """
    Partition test: Address instability.
    
    Splits data by address changes (number of moves).
    High instability = upper quartile of address changes.
    """
    session = load_onnx_model(model_path)
    probs = predict_onnx(session, X)
    
    addr_col = first_present(ADDRESS_FEATURE_CANDIDATES, df_original)
    if not addr_col:
        return {"note": "Address instability feature not found"}
    
    addr_vals = df_original[addr_col].fillna(0)
    threshold = addr_vals.quantile(0.75)
    
    mask_stable = addr_vals <= threshold
    mask_unstable = addr_vals > threshold
    
    results = {
        "test": "Address Instability",
        "threshold": threshold,
        "groups": {}
    }
    
    if mask_stable.sum() > 0:
        results["groups"]["Stable"] = {
            "count": mask_stable.sum(),
            "mean_score": probs[mask_stable.values].mean(),
            "std_score": probs[mask_stable.values].std(),
        }
    
    if mask_unstable.sum() > 0:
        results["groups"]["Unstable"] = {
            "count": mask_unstable.sum(),
            "mean_score": probs[mask_unstable.values].mean(),
            "std_score": probs[mask_unstable.values].std(),
        }
    
    return results


def partition_neighborhood(
    model_path: str | Path,
    X: pd.DataFrame,
    df_original: pd.DataFrame,
) -> dict:
    """
    Partition test: Neighborhood.
    
    Compares model scores across neighborhoods.
    Only shows neighborhoods with sufficient data (>=50 samples).
    """
    session = load_onnx_model(model_path)
    probs = predict_onnx(session, X)
    
    wijk_cols = [c for c in df_original.columns if c.startswith(NEIGHBORHOOD_PREFIX)]
    if not wijk_cols:
        return {"note": "No neighborhood features found"}
    
    results = {
        "test": "Neighborhood",
        "groups": {},
        "overall_mean": probs.mean(),
    }
    
    for col in wijk_cols:
        wijk_name = col.replace(NEIGHBORHOOD_PREFIX, "")
        mask = (df_original[col].fillna(0)) == 1
        count = mask.sum()
        
        if count >= 50:  # Only show if sufficient data
            scores = probs[mask.values]
            results["groups"][wijk_name] = {
                "count": count,
                "mean_score": scores.mean(),
                "std_score": scores.std(),
                "delta_from_overall": scores.mean() - probs.mean(),
            }
    
    return results


def partition_gender(
    model_path: str | Path,
    X: pd.DataFrame,
    df_original: pd.DataFrame,
) -> dict:
    """
    Partition test: Gender.
    
    Compares model scores between male and female groups.
    """
    session = load_onnx_model(model_path)
    probs = predict_onnx(session, X)
    
    if GENDER_COL not in df_original.columns:
        return {"error": "Gender feature not found"}
    
    results = {
        "test": "Gender",
        "groups": {},
    }
    
    # 0 = male, 1 = female
    mask_male = (df_original[GENDER_COL].fillna(0)) == 0
    mask_female = (df_original[GENDER_COL].fillna(0)) == 1
    
    if mask_male.sum() > 0:
        results["groups"]["Male"] = {
            "count": mask_male.sum(),
            "mean_score": probs[mask_male.values].mean(),
            "std_score": probs[mask_male.values].std(),
        }
    
    if mask_female.sum() > 0:
        results["groups"]["Female"] = {
            "count": mask_female.sum(),
            "mean_score": probs[mask_female.values].mean(),
            "std_score": probs[mask_female.values].std(),
        }
    
    return results


# =============================================================================
# AGGREGATION FUNCTIONS
# =============================================================================

def run_all_tests(model_path: str | Path, data_path: str | Path) -> dict:
    """
    Run all metamorphic and partition tests on a model.

    Parameters
    ----------
    model_path : str or Path
        Path to the ONNX model file.
    data_path : str or Path
        Path to the test data CSV file.

    Returns
    -------
    dict
        Nested dictionary with test results:
        - "metamorphic": Results from all 5 metamorphic tests
        - "partitions": Results from all 4 partition tests

    Notes
    -----
    Each metamorphic test returns statistics on prediction changes (mean, median, std, max).
    Each partition test returns prediction distributions per demographic group.
    """
    X, _, df_original = load_data(data_path, return_raw=True)

    results = {
        "metamorphic": {
            "language": metamorphic_language_invariance(model_path, X),
            "address": metamorphic_address_instability_invariance(model_path, X),
            "neighborhood": metamorphic_neighborhood_invariance(model_path, X),
            "gender": metamorphic_gender_invariance(model_path, X),
            "financial": metamorphic_financial_stability_invariance(model_path, X),
        },
        "partitions": {
            "language": partition_language_proficiency(model_path, X, df_original),
            "address": partition_address_instability(model_path, X, df_original),
            "gender": partition_gender(model_path, X, df_original),
            "neighborhood": partition_neighborhood(model_path, X, df_original),
        }
    }

    return results


if __name__ == "__main__":
    # Example usage
    model_path = Path(__file__).resolve().parents[1] / "models" / "goodModel.onnx"
    data_path = Path(__file__).resolve().parents[1] / "data" / "investigation_train_large_checked.csv"
    
    results = run_all_tests(model_path, data_path)
    print(results)
