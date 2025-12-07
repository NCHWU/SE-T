# Bias Detection via Metamorphic Testing

This project implements metamorphic testing to detect bias in machine learning models for welfare fraud detection.

## Setup

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

**Note**: If you encounter NumPy compatibility errors, downgrade NumPy:

```bash
pip install "numpy<2"
```

## Scripts

### 1. Analyze Dataset Bias

Examine language proficiency bias in the dataset:

```bash
python scripts/analyze_language_bias.py --data data/investigation_train_large_checked.csv
```

### 2. Train Models

Train good and bad models with different fairness properties:

```bash
python scripts/goodbadmodel.py --data data/synth_data_for_training.csv
```

This creates:
- `models/goodModel.onnx` - Model with bias mitigation via reweighting
- `models/badModel.onnx` - Model trained with uniform weights (learns bias naturally)

### 3. Run Metamorphic Test

Test a single model for language proficiency bias:

```bash
python scripts/metamorphic.py --model models/goodModel.onnx --data data/investigation_train_large_checked.csv
```

### 4. Compare Models

Compare both models side-by-side:

```bash
python scripts/compare_models.py
```

To use different models, edit the configuration at the top of `compare_models.py`:

```python
GOOD_MODEL_PATH = "models/goodModel.onnx"
BAD_MODEL_PATH = "models/badModel.onnx"
DATA_PATH = "data/investigation_train_large_checked.csv"
LABEL_COLUMN = "checked"
```

## How It Works

**Metamorphic Testing**: The test sets all language proficiency values to "met" and measures how much predictions change. Lower changes indicate less bias.

**Good Model**: Uses sample reweighting to balance positive class across language groups.

**Bad Model**: Uses uniform weights, learning bias from data patterns.

Both models achieve ~94.5% accuracy but differ in fairness properties.
