"""
Group1: Bias Detection via Metamorphic Testing

This package implements metamorphic testing for detecting bias in machine learning
models trained on Rotterdam welfare fraud detection data.

Modules
-------
constants
    Shared configuration values, column names, and hyperparameters.
data_utils
    Data loading utilities shared across training and testing modules.
train_models
    Model training for baseline, good (fair), and bad (biased) model variants.
test_models
    Metamorphic and partition-based bias detection tests.
run_experiments
    Main orchestration script for the complete experiment pipeline.

Example
-------
Run the complete pipeline::

    from run_experiments import main
    main()

Or train models individually::

    from data_utils import load_data
    from train_models import train_good_model
    X, y = load_data("data/synth_data_for_training.csv")
    model, accuracy = train_good_model(X, y)
"""

__version__ = "1.0.0"
__author__ = "Group 1"
