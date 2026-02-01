# Group 1: Bias Detection via Metamorphic Testing

## Overview

This project detects bias in machine learning models using metamorphic testing on Rotterdam welfare fraud detection data.

## Files

### `src/train_models.py`
Trains three models with different fairness strategies:
- **Baseline Model**: Standard training (no interventions)
- **Good Model**: Metamorphic data augmentation (8x training data with gender flipping, age partitioning, language balancing, neighborhood neutralization, address/financial stability)
- **Bad Model**: Extreme bias amplification (8x neighborhood weight, 6x language bias, 4x financial bias, 3x gender bias)

### `src/test_models.py`
Implements bias detection tests:
- **Metamorphic Tests**: Measure prediction changes when protected attributes are transformed (language, address, neighborhood, gender, financial stability)
- **Partition Tests**: Compare prediction scores across demographic groups (language proficiency, address stability, neighborhoods, gender)

### `src/run_experiments.py`
Main orchestration script that:
1. Trains all three models
2. Runs all metamorphic and partition tests
3. Generates comparison report showing bias metrics

## How to Run

```bash
cd Group1/src
python run_experiments.py
```

Results are automatically saved to `Group1/results/experiment_results_YYYYMMDD_HHMMSS.txt` with timestamp.



