#!/usr/bin/env python3
"""
Train good and bad models for bias detection via metamorphic testing.

Good Model: Uses sample reweighting and fairness-aware strategies
Bad Model: Amplifies neighborhood bias through weighting
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
import onnx


RANDOM_STATE = 42
NEIGHBORHOOD_PREFIX = "adres_recentste_wijk_"
GENDER_COL = "persoon_geslacht_vrouw"
AGE_COL = "persoon_leeftijd_bij_onderzoek"
LANG_TAALEIS_COL = "persoonlijke_eigenschappen_taaleis_voldaan"


def load_data(data_path: str | Path, label_column: str = "checked") -> Tuple[pd.DataFrame, pd.Series]:
    """Load dataset and prepare features and labels."""
    df = pd.read_csv(data_path)
    
    if label_column not in df.columns:
        raise KeyError(f"Label column '{label_column}' not found in dataset.")
    
    y = df[label_column]
    
    # Drop label and leaked columns
    columns_to_drop = [label_column, "Ja", "Nee"]
    columns_to_drop = [c for c in columns_to_drop if c in df.columns]
    X = df.drop(columns=columns_to_drop).astype(np.float32)
    
    return X, y


def create_base_pipeline() -> Pipeline:
    """Create baseline gradient boosting pipeline."""
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
    Compute biased sample weights for bad model.
    Amplifies multiple biases:
    1. Neighborhood bias - 8x weight for high-risk/poor neighborhoods (INCREASED)
    2. Gender bias - 3x weight for females (discriminate against women)
    3. Socioeconomic bias - 4x weight for people with financial problems (discriminate against poor)
    4. Language bias - 6x weight for people with migration background (not meeting language req)
    """
    weights = np.ones(len(X))
    
    # 1. High-risk/poor neighborhoods get 8x weight (STRONG neighborhood bias)
    wijk_cols = [c for c in X.columns if c.startswith(NEIGHBORHOOD_PREFIX)]
    if wijk_cols:
        # These are the poorest neighborhoods in Rotterdam
        high_risk_wijken = {"charlois", "delfshaven", "feijenoord", "ijsselmonde", "noord"}
        
        for col in wijk_cols:
            wijk_name = col.replace(NEIGHBORHOOD_PREFIX, "")
            if wijk_name in high_risk_wijken:
                mask = (X[col].fillna(0)) == 1
                weights[mask] *= 8.0  # 8x weight for poor neighborhoods (was 5x)
    
    # 2. Gender bias - amplify weight for females (discriminate against women)
    if GENDER_COL in X.columns:
        female_mask = (X[GENDER_COL].fillna(0)) == 1  # 1 = female
        weights[female_mask] *= 3.0  # 3x weight for females
    
    # 3. Socioeconomic bias - amplify weight for people with financial problems
    financial_prob_cols = [
        "belemmering_financiele_problemen",  # Current financial problems
        "belemmering_dagen_financiele_problemen",  # Days with financial problems
    ]
    
    for col in financial_prob_cols:
        if col in X.columns:
            # Anyone with financial problems gets higher weight
            has_problems = (X[col].fillna(0)) > 0
            weights[has_problems] *= 4.0  # 4x weight for financial problems
    
    # 4. Language bias - discriminate against people with migration background
    if LANG_TAALEIS_COL in X.columns:
        # Language proficiency: 0=not met, 1=met, 2=special
        # Discriminate against those who don't meet language requirement (migration background)
        not_met_mask = (X[LANG_TAALEIS_COL].fillna(1)) == 0  # 0 = not met
        weights[not_met_mask] *= 10.0  # 10x weight for people not meeting language req (was 6x)
    
    return weights


def augment_training_data(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply metamorphic data augmentation to reduce bias on protected attributes.
    
    Transformations:
    1. Gender flipping (male ↔ female)
    2. Age partitioning and sampling within ranges
    3. Language proficiency balancing (create both "met" and "not met" versions)
    4. Neighborhood neutralization
    5. Address stability balancing
    6. Financial stability balancing
    """
    # Set random seed for reproducible augmentation
    np.random.seed(RANDOM_STATE)
    
    X_augmented_list = []
    y_augmented_list = []
    
    # 1. GENDER FLIPPING - Create copy with genders flipped
    if GENDER_COL in X.columns:
        X_gender_flip = X.copy()
        # Flip: 0 -> 1, 1 -> 0 (man ↔ woman)
        X_gender_flip[GENDER_COL] = 1 - X_gender_flip[GENDER_COL]
        X_augmented_list.append(X_gender_flip)
        y_augmented_list.append(y.copy())
    
    # 2. AGE PARTITIONING - Sample within age ranges to prevent threshold learning
    if AGE_COL in X.columns:
        X_age_balanced = X.copy()
        ages = X_age_balanced[AGE_COL].values
        
        # Define age ranges and map to representative values
        age_ranges = [(0, 25, 22), (25, 35, 30), (35, 45, 40), 
                      (45, 55, 50), (55, 65, 60), (65, 100, 70)]
        
        for low, high, representative in age_ranges:
            mask = (ages >= low) & (ages < high)
            # Add noise within range to prevent learning exact thresholds
            noise = np.random.uniform(-2, 2, mask.sum())
            X_age_balanced.loc[mask, AGE_COL] = (representative + noise).astype(np.float32)
        
        X_augmented_list.append(X_age_balanced)
        y_augmented_list.append(y.copy())
    
    # 3. LANGUAGE PROFICIENCY BALANCING - Create both "met" and "not met" versions
    if LANG_TAALEIS_COL in X.columns:
        # Version 1: Everyone has "met" (proficient)
        X_lang_met = X.copy()
        X_lang_met[LANG_TAALEIS_COL] = 1  # All proficient
        X_augmented_list.append(X_lang_met)
        y_augmented_list.append(y.copy())
        
        # Version 2: Everyone has "not met" (not proficient)
        X_lang_not_met = X.copy()
        X_lang_not_met[LANG_TAALEIS_COL] = 0  # All not proficient
        X_augmented_list.append(X_lang_not_met)
        y_augmented_list.append(y.copy())
    
    # 4. NEIGHBORHOOD NEUTRALIZATION - Set all to "other" (neutral)
    wijk_cols = [col for col in X.columns if col.startswith(NEIGHBORHOOD_PREFIX)]
    if wijk_cols:
        X_neigh_neutral = X.copy()
        # Set all neighborhood columns to 0, then set "other" to 1
        for col in wijk_cols:
            X_neigh_neutral[col] = 0
        other_col = NEIGHBORHOOD_PREFIX + "other"
        if other_col in X_neigh_neutral.columns:
            X_neigh_neutral[other_col] = 1
        X_augmented_list.append(X_neigh_neutral)
        y_augmented_list.append(y.copy())
    
    # 5. ADDRESS STABILITY BALANCING - Set address changes to 0 (stable housing)
    addr_cols = [col for col in ["adres_aantal_verschillende_wijken", "adres_aantal_brp_adres", 
                                   "adres_aantal_woonadres_handmatig", "adres_aantal_verzendadres"] 
                 if col in X.columns]
    if addr_cols:
        X_addr_stable = X.copy()
        for col in addr_cols:
            X_addr_stable[col] = 0  # No address changes = stable
        X_augmented_list.append(X_addr_stable)
        y_augmented_list.append(y.copy())
    
    # 6. FINANCIAL STABILITY BALANCING - Set financial problems to 0 (no financial issues)
    fin_cols = [col for col in ["belemmering_financiele_problemen", "belemmering_dagen_financiele_problemen"]
                if col in X.columns]
    if fin_cols:
        X_fin_stable = X.copy()
        for col in fin_cols:
            X_fin_stable[col] = 0  # No financial problems = stable
        X_augmented_list.append(X_fin_stable)
        y_augmented_list.append(y.copy())
    
    # Combine original + all augmented versions
    X_combined = pd.concat([X] + X_augmented_list, ignore_index=True)
    y_combined = pd.concat([y] + y_augmented_list, ignore_index=True)
    
    return X_combined, y_combined


def train_good_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.25,
) -> Tuple[Pipeline, float]:
    """
    Train good model using metamorphic data augmentation for fairness.
    Applies gender flipping, age partitioning, language balancing, and neighborhood neutralization.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    
    # Apply metamorphic data augmentation
    X_train_aug, y_train_aug = augment_training_data(X_train, y_train)
    print(f"  Augmented training data: {len(X_train)} → {len(X_train_aug)} samples")
    
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
    Train bad model using neighborhood-based weighting to amplify bias.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    
    # Bad model: Use neighborhood weighting to amplify bias
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
    Train baseline model with no bias intervention (standard training).
    This is the default model without fairness strategies or bias amplification.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    
    # Baseline: No weighting, just standard training
    pipeline = create_base_pipeline()
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"[Baseline Model] Accuracy: {accuracy:.4f}")
    
    return pipeline, accuracy


def export_to_onnx(pipeline: Pipeline, X: pd.DataFrame, output_path: Path) -> None:
    """Export trained pipeline to ONNX format."""
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
