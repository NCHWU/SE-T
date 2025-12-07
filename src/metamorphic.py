import numpy as np
import pandas as pd

MR_CONF_COL = "typering_indicatie_geheime_gegevens"
AGE_COL = "persoon_leeftijd_bij_onderzoek"
CHANNEL_COLS = [
    "contacten_soort_document__inkomend_",
    "contacten_soort_document__uitgaand_",
    "contacten_soort_e_mail__inkomend_",
    "contacten_soort_e_mail__uitgaand_",
    "contacten_soort_telefoontje__inkomend_",
    "contacten_soort_telefoontje__uitgaand_",
]


def permute_contact_channels(X: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    X_t = X.copy()
    for col in CHANNEL_COLS:
        if col not in X_t.columns:
            raise ValueError(f"{col} missing in X")

    rng = np.random.default_rng(random_state)
    vals = X_t[CHANNEL_COLS].to_numpy()

    for i in range(vals.shape[0]):
        perm = rng.permutation(len(CHANNEL_COLS))
        vals[i, :] = vals[i, perm]

    X_t[CHANNEL_COLS] = vals
    return X_t


def toggle_confidential_flag(X: pd.DataFrame) -> pd.DataFrame:
    X_t = X.copy()
    if MR_CONF_COL not in X_t.columns:
        raise ValueError(f"{MR_CONF_COL} not found in X columns")
    # assumes 0/1 encoding
    X_t[MR_CONF_COL] = 1 - X_t[MR_CONF_COL]
    return X_t


def toggle_gender(X: pd.DataFrame) -> pd.DataFrame:
    X_t = X.copy()
    if "persoon_geslacht_vrouw" in X_t.columns:
        X_t["persoon_geslacht_vrouw"] = 1 - X_t["persoon_geslacht_vrouw"]
    return X_t


def jitter_age_within_bucket(
    X: pd.DataFrame, low=1, high=3, random_state=42
) -> pd.DataFrame:
    """
    Add a small random +/- k years to age while staying inside the same age bucket:
    Child (<18), Youth (18-24), MiddleAge (25-55), Senior (>55).
    """
    X_t = X.copy()
    rng = np.random.default_rng(random_state)

    ages = X_t[AGE_COL].to_numpy(dtype=int)
    deltas = rng.integers(low, high + 1, size=len(ages))  # 1..high

    new_ages = ages.copy()
    for i, a in enumerate(ages):
        # determine bucket
        if a < 18:
            min_a, max_a = 0, 17  # Child
        elif a < 25:
            min_a, max_a = 18, 24  # Youth
        elif a <= 55:
            min_a, max_a = 25, 55  # MiddleAge
        else:
            min_a, max_a = 56, 80  # Senior, arbitrary upper cap

        # random sign
        sign = 1 if rng.random() < 0.5 else -1
        cand = a + sign * deltas[i]
        # clip to stay in same bucket
        cand = max(min_a, min(max_a, cand))
        new_ages[i] = cand

    X_t[AGE_COL] = new_ages
    return X_t


def test_confidential_flag_invariance(model, X):
    X_t = toggle_confidential_flag(X)

    y = model.predict(X)
    y_t = model.predict(X_t)

    changed = y != y_t
    frac_changed = changed.mean()

    print(
        f"Changed predictions confidential flag: {changed.sum()}/{len(y)} ({frac_changed:.2%})"
    )


def test_age_bucket_invariance(model, X):
    X_t = jitter_age_within_bucket(X)

    y = model.predict(X)
    y_t = model.predict(X_t)

    changed = y != y_t
    frac_changed = changed.mean()

    print(f"Changed predictions age: {changed.sum()}/{len(y)} ({frac_changed:.2%})")


def test_contact_channel_invariance(model, X):
    X_t = permute_contact_channels(X)

    y = model.predict(X)
    y_t = model.predict(X_t)

    changed = y != y_t
    frac_changed = changed.mean()

    print(f"Changed contact_channels: {changed.sum()}/{len(y)} ({frac_changed:.2%})")


def test_gender_flip(model, X):
    """
    For each row, create a twin where persoon_geslacht_vrouw is flipped,
    and check how often the prediction changes.
    """
    X_t = toggle_gender(X)

    y = model.predict(X)
    y_t = model.predict(X_t)

    changed = y != y_t
    frac_changed = changed.mean()

    print(f"Changed gender flag: {changed.sum()}/{len(y)} ({frac_changed:.2%})")
