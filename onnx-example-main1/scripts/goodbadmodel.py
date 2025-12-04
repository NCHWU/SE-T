#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


RANDOM_STATE = 42
NEIGHBORHOOD_PREFIX = "adres_recentste_wijk_"

# Column names for sensitive attributes in this synthetic dataset
GENDER_COL = "persoon_geslacht_vrouw"
AGE_COL = "persoon_leeftijd_bij_onderzoek"
LANG_TAALEIS_COL = "persoonlijke_eigenschappen_taaleis_voldaan"


def parse_args() -> argparse.Namespace:
    """
    CLI for training a deliberately 'good' and 'bad' model and exporting them to ONNX.

    This mirrors the interfaces used in the other scripts so that you can re‑use
    the same CSV and label column.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train a deliberately 'good' and 'bad' model, and export them as "
            "model_1.onnx and model_2.onnx."
        )
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
        help="Fraction reserved for evaluating each model.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "models"),
        help="Directory where model_1.onnx and model_2.onnx will be written.",
    )
    return parser.parse_args()


def load_inputs(path: str | Path, label_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the training data.

    This is kept intentionally the same as in the other scripts so tests can
    treat this as a black‑box model with the same interface.
    """
    df = pd.read_csv(path)
    if label_column not in df.columns:
        raise KeyError(f"Label column '{label_column}' not found in dataset.")

    y = df[label_column]

    # Drop label column and leaked prediction columns
    columns_to_drop = [label_column, "Ja", "Nee"]
    columns_to_drop = [c for c in columns_to_drop if c in df.columns]
    X = df.drop(columns=columns_to_drop).astype(np.float32)
    return X, y


def make_purposely_biased_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Construct a tiny feature set that bakes in *undesirable* biases.

    Intuition (you can tweak this to match your own story):
      - Gender: treat being a woman as *higher* risk.
      - Age: treat being older as higher risk.
      - Language: treat "taaleis not met" as much higher risk.
      - Neighborhood: treat a chosen set of wijken as "low income" => higher risk.
    """
    biased = pd.DataFrame(index=X.index)

    # 1) Gender: 1 = woman => higher risk
    if GENDER_COL in X.columns:
        biased["is_woman"] = X[GENDER_COL].fillna(0.0)
    else:
        biased["is_woman"] = 0.0

    # 2) Age: centre roughly around 40 and scale;
    #    plus a *big* bump for 15–25 year olds to encode your story
    #    that young people "need more money so may commit fraud".
    if AGE_COL in X.columns:
        age = X[AGE_COL].fillna(X[AGE_COL].median())
        biased["age_scaled"] = (age - 40.0) / 10.0

        # Indicator for "young and therefore (biasedly) seen as high‑risk"
        young_mask = (age >= 15) & (age <= 25)
        # Weight is arbitrary but large so the model really picks this up.
        biased["young_high_risk"] = young_mask.astype(float) * 3.0
    else:
        biased["age_scaled"] = 0.0
        biased["young_high_risk"] = 0.0

    # 3) Language: if taaleis not met => higher risk
    if LANG_TAALEIS_COL in X.columns:
        taaleis = X[LANG_TAALEIS_COL].fillna(0.0)
        # assume 1 == taaleis met; flip so that 1 == NOT met
        not_met = 1.0 - taaleis
        biased["taaleis_not_met"] = not_met

        # Extra-strong penalty feature so the bad model really leans on language
        biased["taaleis_strong_penalty"] = not_met * 2.0
    else:
        biased["taaleis_not_met"] = 0.0
        biased["taaleis_strong_penalty"] = 0.0

    # 4) Neighborhood: manual "low‑income" risk scoring over wijk one‑hots
    wijk_cols = [c for c in X.columns if c.startswith(NEIGHBORHOOD_PREFIX)]

    # You can tweak this mapping to match the narrative you want to tell.
    # The idea: some wijken get much higher risk weights than others.
    wijk_risk_weights: Dict[str, float] = {}
    for col in wijk_cols:
        wijk_name = col.replace(NEIGHBORHOOD_PREFIX, "")
        # Your (biased) narrative: these wijken are seen as "low income / high risk"
        if wijk_name in {"charlois", "delfshaven", "feijenoord", "ijsselmonde", "noord"}:
            weight = 3.0
        elif wijk_name == "other":
            weight = 0.5
        else:
            weight = 1.0
        wijk_risk_weights[col] = weight

    if wijk_cols:
        neighborhood_risk = 0.0
        for col, weight in wijk_risk_weights.items():
            neighborhood_risk += X[col].fillna(0.0) * weight
        biased["neighborhood_risk"] = neighborhood_risk
    else:
        biased["neighborhood_risk"] = 0.0

    return biased

def drop_sensitive_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Remove features we consider sensitive or strong proxies:
      - gender
      - age
      - taaleis (language requirement)
      - neighborhood dummies (wijk)

    This is used for the 'good' model to reduce reliance on these signals.
    """
    cols_to_drop = []

    # Direct sensitive features
    for col in [GENDER_COL, AGE_COL, LANG_TAALEIS_COL]:
        if col in X.columns:
            cols_to_drop.append(col)

    # Neighborhood / wijk one-hots (proxies for socio-economic / migration background)
    wijk_cols = [c for c in X.columns if c.startswith(NEIGHBORHOOD_PREFIX)]
    cols_to_drop.extend(wijk_cols)

    if cols_to_drop:
        X = X.drop(columns=cols_to_drop)

    return X


def make_base_pipeline() -> Pipeline:
    """
    Baseline pipeline copied from the notebook model.

    You will adapt this in `train_good_model` and `train_bad_model` to encode
    your different design choices about (un)desirable bias.
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


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float,
    bias_mode: str,
) -> Tuple[Pipeline, float]:
    """
    Generic helper that:
      - splits the data
      - fits a pipeline
      - returns the trained model and its accuracy on a hold‑out set

    The *only* difference between the 'good' and 'bad' model should come from
    how you configure the pipeline based on `bias_mode`.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = make_base_pipeline()
    X_train_used = X_train
    X_test_used = X_test

    if bias_mode == "good":
        # PURPOSEFULLY GOOD:
        #   Drop sensitive / problematic features so the model can't rely on them.
        X_train_used = drop_sensitive_features(X_train)
        X_test_used = drop_sensitive_features(X_test)

        

    elif bias_mode == "bad":
        # PURPOSEFULLY BAD:
        #   Only feed the model a handful of sensitive / problematic features,
        #   and engineer them so that they strongly correlate with "risk".
        X_train_used = make_purposely_biased_features(X_train)
        X_test_used = make_purposely_biased_features(X_test)
    else:
        raise ValueError(f"Unknown bias_mode: {bias_mode!r}")

    pipeline.fit(X_train_used, y_train)
    test_preds = pipeline.predict(X_test_used)
    accuracy = accuracy_score(y_test, test_preds)
    print(f"[{bias_mode} model] accuracy on holdout: {accuracy:.3f}")
    return pipeline, accuracy


def export_model_to_onnx(pipeline: Pipeline, X: pd.DataFrame, output_path: Path) -> None:
    """
    Convert a fitted sklearn pipeline to ONNX and save it.

    We keep the interface simple: the ONNX model expects a single float32
    tensor of shape (None, n_features), where n_features is whatever the
    fitted pipeline was trained on.
    """
    # Determine input dimensionality from the trained model if available
    if hasattr(pipeline, "n_features_in_"):
        n_features = int(pipeline.n_features_in_)
    else:
        n_features = int(X.shape[1])

    initial_type = [("input", FloatTensorType([None, n_features]))]

    onx = convert_sklearn(pipeline, initial_types=initial_type)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        f.write(onx.SerializeToString())


def main() -> None:
    args = parse_args()
    X, y = load_inputs(args.data, args.label_column)

    # Train one 'good' and one 'bad' model
    good_model, good_acc = train_model(X, y, test_size=args.test_size, bias_mode="good")
    bad_model, bad_acc = train_model(X, y, test_size=args.test_size, bias_mode="bad")

    print("\nSummary of holdout performance:")
    print(f"  good model accuracy: {good_acc:.3f}")
    print(f"  bad  model accuracy: {bad_acc:.3f}")

    # Prepare output directory and decide which ONNX file gets which model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_paths = [output_dir / "badModel.onnx", output_dir / "goodModel.onnx"]

    rng = np.random.default_rng(RANDOM_STATE)
    permutation = rng.permutation(2)
    models = [good_model, bad_model]
    labels = ["good", "bad"]

    for onnx_idx, model_idx in enumerate(permutation):
        model = models[model_idx]
        label = labels[model_idx]
        path = model_paths[onnx_idx]
        print(f"Exporting {label!r} model to {path.name}")
        export_model_to_onnx(model, X, path)

if __name__ == "__main__":
    main()


