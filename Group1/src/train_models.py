#!/usr/bin/env python3
"""
Model training module for bias detection via metamorphic testing.

This module provides functions to train three model variants:
    - Baseline Model: Standard GradientBoosting without any fairness interventions
    - Good Model: Uses metamorphic data augmentation to reduce bias on protected attributes
    - Bad Model: Intentionally biased through weighted sampling to amplify discrimination

The models are exported in ONNX format for portable inference.

Example
-------
Train all models and export to ONNX::

    from train_models import main
    main()

Or train individually::

    from train_models import load_data, train_good_model, export_to_onnx
    X, y = load_data("data/synth_data_for_training.csv")
    model, accuracy = train_good_model(X, y)
    export_to_onnx(model, X, "models/goodModel.onnx")
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from constants import (
    RANDOM_STATE,
    NEIGHBORHOOD_PREFIX,
    GENDER_COL,
    AGE_COL,
    LANG_TAALEIS_COL,
    HIGH_RISK_NEIGHBORHOODS,
    BAD_MODEL_NEIGHBORHOOD_WEIGHT,
    BAD_MODEL_GENDER_WEIGHT,
    BAD_MODEL_FINANCIAL_WEIGHT,
    BAD_MODEL_LANGUAGE_WEIGHT,
    AGE_PARTITIONS,
    FINANCIAL_PROBLEM_COLS,
    ADDRESS_FEATURE_CANDIDATES,
)
from data_utils import load_data


def create_base_pipeline() -> Pipeline:
    """
    Create the base sklearn pipeline for all model variants.

    The pipeline consists of:
    1. VarianceThreshold: Remove zero-variance features
    2. GradientBoostingClassifier: Ensemble classifier with 100 estimators

    Returns
    -------
    Pipeline
        Unfitted sklearn pipeline ready for training.

    Notes
    -----
    Uses RANDOM_STATE for reproducibility. The shallow max_depth=1 creates
    decision stumps, which are less prone to overfitting.
    """
    return Pipeline(
        steps=[
            ("variance_threshold", VarianceThreshold()),
            (
                "gb",
                GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=1.0,
                    max_depth=1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def compute_bad_model_weights(X: pd.DataFrame) -> np.ndarray:
    """
    Compute biased sample weights for the intentionally biased ("bad") model.

    This function creates sample weights that amplify discrimination against
    protected groups, demonstrating how bias can be introduced through training.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix containing protected attribute columns.

    Returns
    -------
    np.ndarray
        Sample weights array of shape (n_samples,). Higher weights cause the
        model to focus more on those samples during training.

    Notes
    -----
    Bias amplification weights (multiplicative):
    - Neighborhood: 8x for low-income Rotterdam neighborhoods
    - Gender: 3x for female individuals
    - Financial: 4x for individuals with financial problems
    - Language: 10x for individuals not meeting language requirements

    These weights compound multiplicatively, so an individual matching multiple
    criteria receives very high weight (e.g., 8 * 3 * 4 * 10 = 960x).
    """
    weights = np.ones(len(X))

    # 1. Neighborhood bias: amplify weight for low-income neighborhoods
    wijk_cols = [c for c in X.columns if c.startswith(NEIGHBORHOOD_PREFIX)]
    if wijk_cols:
        for col in wijk_cols:
            wijk_name = col.replace(NEIGHBORHOOD_PREFIX, "")
            if wijk_name in HIGH_RISK_NEIGHBORHOODS:
                mask = (X[col].fillna(0)) == 1
                weights[mask] *= BAD_MODEL_NEIGHBORHOOD_WEIGHT

    # 2. Gender bias: amplify weight for females
    if GENDER_COL in X.columns:
        female_mask = (X[GENDER_COL].fillna(0)) == 1
        weights[female_mask] *= BAD_MODEL_GENDER_WEIGHT

    # 3. Socioeconomic bias: amplify weight for financial problems
    for col in FINANCIAL_PROBLEM_COLS:
        if col in X.columns:
            has_problems = (X[col].fillna(0)) > 0
            weights[has_problems] *= BAD_MODEL_FINANCIAL_WEIGHT

    # 4. Language bias: amplify weight for migration background indicator
    if LANG_TAALEIS_COL in X.columns:
        # Language proficiency: 0=not met, 1=met, 2=special
        not_met_mask = (X[LANG_TAALEIS_COL].fillna(1)) == 0
        weights[not_met_mask] *= BAD_MODEL_LANGUAGE_WEIGHT

    return weights


def augment_training_data(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply metamorphic data augmentation to create a fairness-aware training set.

    This function expands the training data by creating transformed copies where
    protected attributes are modified. The model learns to make predictions that
    are invariant to these protected attributes.

    Parameters
    ----------
    X : pd.DataFrame
        Original feature matrix.
    y : pd.Series
        Original target labels.

    Returns
    -------
    X_augmented : pd.DataFrame
        Expanded feature matrix (~8x original size).
    y_augmented : pd.Series
        Expanded target labels (copied for each augmentation).

    Notes
    -----
    Augmentation transformations applied:
    1. Gender flipping: Create copies with genders swapped (male ↔ female)
    2. Age partitioning: Jitter ages within defined ranges to prevent threshold learning
    3. Language balancing: Create "met" and "not met" versions for all samples
    4. Neighborhood neutralization: Set all samples to neutral "other" neighborhood
    5. Address stability: Set address change counts to 0 (stable housing)
    6. Financial stability: Set financial problem indicators to 0

    The augmented dataset teaches the model that predictions should not depend
    on these protected attributes, as the same label applies regardless of
    the attribute value.
    """
    np.random.seed(RANDOM_STATE)

    X_augmented_list = []
    y_augmented_list = []

    # 1. GENDER FLIPPING - Create copy with genders swapped
    if GENDER_COL in X.columns:
        X_gender_flip = X.copy()
        X_gender_flip[GENDER_COL] = 1 - X_gender_flip[GENDER_COL]
        X_augmented_list.append(X_gender_flip)
        y_augmented_list.append(y.copy())

    # 2. AGE PARTITIONING - Jitter within age ranges to prevent threshold learning
    if AGE_COL in X.columns:
        X_age_balanced = X.copy()
        ages = X_age_balanced[AGE_COL].values

        for low, high, representative in AGE_PARTITIONS:
            mask = (ages >= low) & (ages < high)
            noise = np.random.uniform(-2, 2, mask.sum())
            X_age_balanced.loc[mask, AGE_COL] = (representative + noise).astype(np.float32)

        X_augmented_list.append(X_age_balanced)
        y_augmented_list.append(y.copy())

    # 3. LANGUAGE PROFICIENCY BALANCING - Create both "met" and "not met" versions
    if LANG_TAALEIS_COL in X.columns:
        X_lang_met = X.copy()
        X_lang_met[LANG_TAALEIS_COL] = 1
        X_augmented_list.append(X_lang_met)
        y_augmented_list.append(y.copy())

        X_lang_not_met = X.copy()
        X_lang_not_met[LANG_TAALEIS_COL] = 0
        X_augmented_list.append(X_lang_not_met)
        y_augmented_list.append(y.copy())

    # 4. NEIGHBORHOOD NEUTRALIZATION - Set all to neutral "other"
    wijk_cols = [col for col in X.columns if col.startswith(NEIGHBORHOOD_PREFIX)]
    if wijk_cols:
        X_neigh_neutral = X.copy()
        for col in wijk_cols:
            X_neigh_neutral[col] = 0
        other_col = NEIGHBORHOOD_PREFIX + "other"
        if other_col in X_neigh_neutral.columns:
            X_neigh_neutral[other_col] = 1
        X_augmented_list.append(X_neigh_neutral)
        y_augmented_list.append(y.copy())

    # 5. ADDRESS STABILITY BALANCING - Set all address changes to 0
    addr_cols = [col for col in ADDRESS_FEATURE_CANDIDATES if col in X.columns]
    if addr_cols:
        X_addr_stable = X.copy()
        for col in addr_cols:
            X_addr_stable[col] = 0
        X_augmented_list.append(X_addr_stable)
        y_augmented_list.append(y.copy())

    # 6. FINANCIAL STABILITY BALANCING - Set financial problems to 0
    fin_cols = [col for col in FINANCIAL_PROBLEM_COLS if col in X.columns]
    if fin_cols:
        X_fin_stable = X.copy()
        for col in fin_cols:
            X_fin_stable[col] = 0
        X_augmented_list.append(X_fin_stable)
        y_augmented_list.append(y.copy())

    # Combine original data with all augmented versions
    X_combined = pd.concat([X] + X_augmented_list, ignore_index=True)
    y_combined = pd.concat([y] + y_augmented_list, ignore_index=True)

    return X_combined, y_combined


def train_good_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.25,
) -> Tuple[Pipeline, float]:
    """
    Train a fairness-aware model using metamorphic data augmentation.

    The "good" model is trained on augmented data where protected attributes
    are systematically varied while keeping labels constant. This teaches
    the model to make predictions invariant to protected attributes.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target labels.
    test_size : float, default=0.25
        Fraction of data to reserve for testing.

    Returns
    -------
    pipeline : Pipeline
        Trained sklearn pipeline.
    accuracy : float
        Test set accuracy (evaluated on original, non-augmented test data).

    See Also
    --------
    augment_training_data : The augmentation function used to expand training data.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    # Apply metamorphic data augmentation
    X_train_aug, y_train_aug = augment_training_data(X_train, y_train)
    print(f"  Augmented training data: {len(X_train)} -> {len(X_train_aug)} samples")

    # Train on augmented data with uniform weights
    pipeline = create_base_pipeline()
    pipeline.fit(X_train_aug, y_train_aug)

    # Evaluate on original test set (not augmented)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"[Good Model] Accuracy: {accuracy:.4f}")

    return pipeline, accuracy


def train_bad_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.25,
) -> Tuple[Pipeline, float]:
    """
    Train an intentionally biased model using weighted sampling.

    The "bad" model amplifies discrimination by applying higher sample weights
    to individuals from protected groups (low-income neighborhoods, females,
    those with financial problems, non-native language speakers).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target labels.
    test_size : float, default=0.25
        Fraction of data to reserve for testing.

    Returns
    -------
    pipeline : Pipeline
        Trained sklearn pipeline (biased).
    accuracy : float
        Test set accuracy.

    See Also
    --------
    compute_bad_model_weights : Function that computes the biased sample weights.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    # Compute biased sample weights
    weights = compute_bad_model_weights(X_train)

    pipeline = create_base_pipeline()
    pipeline.fit(X_train, y_train, gb__sample_weight=weights)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"[Bad Model] Accuracy: {accuracy:.4f}")

    return pipeline, accuracy


def train_baseline_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.25,
) -> Tuple[Pipeline, float]:
    """
    Train a baseline model with standard training (no fairness interventions).

    The baseline model serves as a control for comparison. It uses uniform
    sample weights and no data augmentation, representing typical ML training.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target labels.
    test_size : float, default=0.25
        Fraction of data to reserve for testing.

    Returns
    -------
    pipeline : Pipeline
        Trained sklearn pipeline.
    accuracy : float
        Test set accuracy.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = create_base_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"[Baseline Model] Accuracy: {accuracy:.4f}")

    return pipeline, accuracy


def export_to_onnx(pipeline: Pipeline, X: pd.DataFrame, output_path: Path) -> None:
    """
    Export a trained sklearn pipeline to ONNX format.

    Parameters
    ----------
    pipeline : Pipeline
        Trained sklearn pipeline to export.
    X : pd.DataFrame
        Sample feature matrix (used to determine input shape).
    output_path : Path
        Destination path for the ONNX file.

    Notes
    -----
    Creates parent directories if they don't exist. The ONNX model accepts
    float32 input tensors of shape (batch_size, n_features).
    """
    n_features = X.shape[1]
    initial_type = [("input", FloatTensorType([None, n_features]))]

    onx = convert_sklearn(pipeline, initial_types=initial_type)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as f:
        f.write(onx.SerializeToString())

    print(f"Exported to {output_path}")


def main():
    """Train all three models and export to ONNX."""
    data_path = Path(__file__).resolve().parents[1] / "data" / "synth_data_for_training.csv"
    output_dir = Path(__file__).resolve().parents[1] / "models"
    
    # Load data
    X, y = load_data(data_path)
    print(f"Loaded {len(X)} samples, {X.shape[1]} features")
    
    # Train models
    print("\nTraining baseline model...")
    baseline_model, baseline_acc = train_baseline_model(X, y)
    
    print("\nTraining good model...")
    good_model, good_acc = train_good_model(X, y)
    
    print("\nTraining bad model...")
    bad_model, bad_acc = train_bad_model(X, y)
    
    # Export to ONNX
    print("\nExporting models...")
    export_to_onnx(baseline_model, X, output_dir / "baselineModel.onnx")
    export_to_onnx(good_model, X, output_dir / "goodModel.onnx")
    export_to_onnx(bad_model, X, output_dir / "badModel.onnx")
    
    print(f"\n{'='*60}")
    print(f"Baseline Model Accuracy: {baseline_acc:.4f}")
    print(f"Good Model Accuracy:     {good_acc:.4f}")
    print(f"Bad Model Accuracy:      {bad_acc:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
