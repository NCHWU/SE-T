"""
Shared constants and configuration for the bias detection pipeline.

This module centralizes all configuration values, column names, and constants
used across training and testing modules to ensure consistency and ease of maintenance.
"""

# =============================================================================
# REPRODUCIBILITY
# =============================================================================

RANDOM_STATE = 42
"""Random seed used for all stochastic operations (train/test splits, augmentation)."""

# =============================================================================
# FEATURE COLUMN NAMES
# =============================================================================

# Neighborhood features (one-hot encoded Rotterdam districts)
NEIGHBORHOOD_PREFIX = "adres_recentste_wijk_"
"""Prefix for one-hot encoded neighborhood columns (9 Rotterdam districts)."""

# Gender feature
GENDER_COL = "persoon_geslacht_vrouw"
"""Gender column: 0 = male, 1 = female."""

# Age feature
AGE_COL = "persoon_leeftijd_bij_onderzoek"
"""Age at time of investigation (numeric, in years)."""

# Language proficiency feature (multiple possible column names due to data variations)
LANG_TAALEIS_COL = "persoonlijke_eigenschappen_taaleis_voldaan"
"""Primary language proficiency column: 0 = not met, 1 = met, 2 = special exemption."""

LANG_TAALEIS_CANDIDATES = [
    "persoonlijke_eigenschappen_taaleis_voldaan",
    "personeleijke_eigenschappen_taaleis_voldaan",  # Common typo in some datasets
    "taaleis_voldaan",
]
"""Possible column names for language proficiency (handles dataset variations)."""

# Address instability features
ADDRESS_FEATURE_CANDIDATES = [
    "adres_aantal_verschillende_wijken",      # Number of different neighborhoods
    "adres_aantal_brp_adres",                 # Number of BRP address changes
    "adres_aantal_woonadres_handmatig",       # Number of manual address entries
    "adres_aantal_verzendadres",              # Number of mailing addresses
]
"""Columns indicating address/housing instability."""

# Financial problem features
FINANCIAL_PROBLEM_COLS = [
    "belemmering_financiele_problemen",       # Binary: has financial problems
    "belemmering_dagen_financiele_problemen", # Days with financial problems
]
"""Columns indicating financial difficulties."""

# =============================================================================
# BAD MODEL BIAS WEIGHTS
# =============================================================================

# Neighborhoods considered "high-risk" (lower socioeconomic status areas in Rotterdam)
HIGH_RISK_NEIGHBORHOODS = {
    "charlois",
    "delfshaven",
    "feijenoord",
    "ijsselmonde",
    "noord",
}
"""Rotterdam neighborhoods with lower socioeconomic status, used for bias amplification in bad model."""

# Weights for bad model bias amplification
BAD_MODEL_NEIGHBORHOOD_WEIGHT = 8.0
"""Sample weight multiplier for residents of high-risk neighborhoods."""

BAD_MODEL_GENDER_WEIGHT = 3.0
"""Sample weight multiplier for female individuals."""

BAD_MODEL_FINANCIAL_WEIGHT = 4.0
"""Sample weight multiplier for individuals with financial problems."""

BAD_MODEL_LANGUAGE_WEIGHT = 10.0
"""Sample weight multiplier for individuals not meeting language requirements."""

# =============================================================================
# GOOD MODEL AGE PARTITIONS
# =============================================================================

AGE_PARTITIONS = [
    (0, 25, 22),    # Youth: 0-24 → representative age 22
    (25, 35, 30),   # Young adult: 25-34 → representative age 30
    (35, 45, 40),   # Adult: 35-44 → representative age 40
    (45, 55, 50),   # Middle age: 45-54 → representative age 50
    (55, 65, 60),   # Pre-retirement: 55-64 → representative age 60
    (65, 100, 70),  # Senior: 65+ → representative age 70
]
"""Age ranges for partition-based augmentation: (min_age, max_age, representative_value)."""

# =============================================================================
# DATA LABELS
# =============================================================================

LABEL_COLUMN = "checked"
"""Target column name: 1 = flagged for investigation, 0 = not flagged."""

LEAKED_COLUMNS = ["Ja", "Nee"]
"""Columns that leak target information and must be dropped before training."""
