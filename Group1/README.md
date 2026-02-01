# Group 1: Bias Detection via Metamorphic Testing

## Overview

This project implements **metamorphic testing** to detect bias in machine learning models trained on Rotterdam welfare fraud detection data. The system trains three model variants and compares their fairness across protected attributes (gender, age, language proficiency, neighborhood, financial stability).

### What is Metamorphic Testing?

Metamorphic testing is a software testing technique that verifies program behavior through **metamorphic relations** (MRs) — relationships between inputs and outputs across multiple executions. For bias detection:

- **Input transformation**: Modify a protected attribute (e.g., flip gender from male to female)
- **Expected relation**: A fair model should produce similar predictions regardless of protected attributes
- **Bias indicator**: Large prediction changes when protected attributes are modified indicate bias

## Project Structure

```
Group1/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── constants.py             # Shared constants and configuration
│   ├── data_utils.py            # Data loading utilities
│   ├── train_models.py          # Model training (baseline, good, bad)
│   ├── test_models.py           # Metamorphic and partition tests
│   └── run_experiments.py       # Main experiment orchestration
├── data/
│   ├── synth_data_for_training.csv           # Training dataset (12,645 samples)
│   └── investigation_train_large_checked.csv # Test dataset for bias evaluation
├── models/                      # Generated ONNX model files
│   ├── baselineModel.onnx
│   ├── goodModel.onnx
│   └── badModel.onnx
├── results/                     # Timestamped experiment output files
├── requirements.txt             # Python dependencies
└── README.md
```

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

## Installation

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Linux/macOS
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run Complete Experiment Pipeline

```bash
cd Group1/src
python run_experiments.py
```

This will:
1. Load training data from `data/synth_data_for_training.csv`
2. Train three model variants (baseline, good, bad)
3. Export models to ONNX format in `models/`
4. Run all metamorphic and partition tests
5. Save results to `results/experiment_results_YYYYMMDD_HHMMSS.txt`

### Run Individual Components

**Train models only:**
```bash
python train_models.py
```

**Run tests on existing models:**
```bash
python test_models.py
```

## Model Variants

| Model | Strategy | Purpose |
|-------|----------|---------|
| **Baseline** | Standard GradientBoosting, no interventions | Control model for comparison |
| **Good** | Metamorphic data augmentation (gender flipping, age jittering, language balancing, neighborhood neutralization) | Demonstrates fairness-aware training |
| **Bad** | Weighted sampling (8x poor neighborhoods, 3x females, 10x language violations) | Demonstrates intentionally biased model |

## Tests Implemented

### Metamorphic Tests

| Test | Transformation | Fair Model Behavior |
|------|----------------|---------------------|
| Language Invariance | Set language proficiency to "met" vs "not met" | Minimal prediction change |
| Address Instability | Set all address changes to 0 (stable) | Minimal prediction change |
| Neighborhood Invariance | Set all neighborhoods to "other" (neutral) | Minimal prediction change |
| Gender Invariance | Flip gender (male ↔ female) | Minimal prediction change |
| Financial Stability | Set financial problems to 0 (stable) | Minimal prediction change |

### Partition Tests

| Test | Stratification | What it Measures |
|------|----------------|------------------|
| Language Proficiency | Met / Not Met / Special | Prediction score disparity across language groups |
| Address Instability | Stable / Unstable (75th percentile threshold) | Prediction score disparity by housing stability |
| Gender | Male / Female | Prediction score disparity between genders |
| Neighborhood | 9 Rotterdam districts | Prediction score disparity across neighborhoods |

## Interpreting Results

### Metamorphic Test Results

- **Lower values = More fair** (less bias)
- Metrics reported: mean, median, std, min, max of absolute prediction changes
- A fair model shows values close to 0

### Partition Test Results

- Compare mean prediction scores across demographic groups
- Large differences between groups indicate potential bias
- Fair models show similar scores across all groups

## Reproducibility

- **Random seed**: All random operations use `RANDOM_STATE = 42`
- **Stratified splits**: Train/test splits preserve class distribution
- **Timestamped outputs**: Each run creates a uniquely named results file
- **ONNX export**: Models are saved in portable ONNX format for reproducible inference

## Protected Attributes Tested

| Attribute | Column Name | Values |
|-----------|-------------|--------|
| Gender | `persoon_geslacht_vrouw` | 0 = male, 1 = female |
| Age | `persoon_leeftijd_bij_onderzoek` | Numeric (years) |
| Language Proficiency | `persoonlijke_eigenschappen_taaleis_voldaan` | 0 = not met, 1 = met, 2 = special |
| Neighborhood | `adres_recentste_wijk_*` | One-hot encoded (9 districts) |
| Financial Problems | `belemmering_financiele_problemen` | Binary indicator |

## Technical Details

### Pipeline Architecture

```
VarianceThreshold → GradientBoostingClassifier(n_estimators=100, max_depth=1)
```

### Data Augmentation (Good Model)

The good model expands training data ~8x through transformations:
1. Gender flipping (creates opposite-gender copies)
2. Age jittering within partitions (prevents threshold learning)
3. Language proficiency balancing (both "met" and "not met" versions)
4. Neighborhood neutralization (set all to "other")
5. Address stability balancing (set changes to 0)
6. Financial stability balancing (set problems to 0)

### Bias Amplification (Bad Model)

The bad model uses weighted sampling to amplify discrimination:
- 8x weight for poor neighborhoods (Charlois, Delfshaven, Feijenoord, IJsselmonde, Noord)
- 3x weight for females
- 4x weight for financial problems
- 10x weight for language requirement violations
