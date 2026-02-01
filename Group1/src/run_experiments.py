#!/usr/bin/env python3
"""
Main experiment runner: Train models and run all tests.

This orchestrates the complete bias detection pipeline:
1. Train good and bad models
2. Run metamorphic tests
3. Run partition tests
4. Generate comparison report
"""

import sys
from datetime import datetime
from pathlib import Path

from train_models import train_baseline_model, train_good_model, train_bad_model, load_data, export_to_onnx
from test_models import run_all_tests


class TeeOutput:
    """Write to both stdout and file simultaneously."""
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


def print_metamorphic_results(baseline_results, good_results, bad_results):
    """Print metamorphic test comparison for all 3 models."""
    print("\n" + "="*90)
    print("METAMORPHIC TEST RESULTS")
    print("="*90)
    print("\nLower values = More fair (less bias)")
    
    # Language Invariance
    print(f"\n{'LANGUAGE PROFICIENCY INVARIANCE':<90}")
    print(f"\n{'Metric':<15} {'Baseline':>20} {'Good Model':>20} {'Bad Model':>20}")
    print("-"*90)
    
    baseline_lang = baseline_results.get("metamorphic", {}).get("language", {})
    good_lang = good_results.get("metamorphic", {}).get("language", {})
    bad_lang = bad_results.get("metamorphic", {}).get("language", {})
    
    if "error" not in good_lang and "error" not in bad_lang:
        for metric in ["mean", "median", "std", "max"]:
            print(f"{metric.capitalize():<15} {baseline_lang.get(metric, 0):>20.6f} {good_lang[metric]:>20.6f} {bad_lang[metric]:>20.6f}")
    
    # Address Instability Invariance
    print(f"\n{'ADDRESS INSTABILITY INVARIANCE':<90}")
    print(f"\n{'Metric':<15} {'Baseline':>20} {'Good Model':>20} {'Bad Model':>20}")
    print("-"*90)
    
    baseline_addr = baseline_results.get("metamorphic", {}).get("address", {})
    good_addr = good_results.get("metamorphic", {}).get("address", {})
    bad_addr = bad_results.get("metamorphic", {}).get("address", {})
    
    if "error" not in good_addr and "error" not in bad_addr:
        for metric in ["mean", "median", "std", "max"]:
            print(f"{metric.capitalize():<15} {baseline_addr.get(metric, 0):>20.6f} {good_addr[metric]:>20.6f} {bad_addr[metric]:>20.6f}")
    else:
        print("Address features not available in this dataset")
    
    # Neighborhood Invariance
    print(f"\n{'NEIGHBORHOOD INVARIANCE':<90}")
    print(f"\n{'Metric':<15} {'Baseline':>20} {'Good Model':>20} {'Bad Model':>20}")
    print("-"*90)
    
    baseline_neigh = baseline_results.get("metamorphic", {}).get("neighborhood", {})
    good_neigh = good_results.get("metamorphic", {}).get("neighborhood", {})
    bad_neigh = bad_results.get("metamorphic", {}).get("neighborhood", {})
    
    if "error" not in good_neigh and "error" not in bad_neigh:
        for metric in ["mean", "median", "std", "max"]:
            print(f"{metric.capitalize():<15} {baseline_neigh.get(metric, 0):>20.6f} {good_neigh[metric]:>20.6f} {bad_neigh[metric]:>20.6f}")
    else:
        print("Neighborhood features not available in this dataset")
    
    # Gender Invariance
    print(f"\n{'GENDER INVARIANCE':<90}")
    print(f"\n{'Metric':<15} {'Baseline':>20} {'Good Model':>20} {'Bad Model':>20}")
    print("-"*90)
    
    baseline_gender = baseline_results.get("metamorphic", {}).get("gender", {})
    good_gender = good_results.get("metamorphic", {}).get("gender", {})
    bad_gender = bad_results.get("metamorphic", {}).get("gender", {})
    
    if "error" not in good_gender and "error" not in bad_gender:
        for metric in ["mean", "median", "std", "max"]:
            print(f"{metric.capitalize():<15} {baseline_gender.get(metric, 0):>20.6f} {good_gender[metric]:>20.6f} {bad_gender[metric]:>20.6f}")
    else:
        print("Gender feature not available in this dataset")
    
    # Financial Stability Invariance
    print(f"\n{'FINANCIAL STABILITY INVARIANCE':<90}")
    print(f"\n{'Metric':<15} {'Baseline':>20} {'Good Model':>20} {'Bad Model':>20}")
    print("-"*90)
    
    baseline_fin = baseline_results.get("metamorphic", {}).get("financial", {})
    good_fin = good_results.get("metamorphic", {}).get("financial", {})
    bad_fin = bad_results.get("metamorphic", {}).get("financial", {})
    
    if "error" not in good_fin and "error" not in bad_fin:
        for metric in ["mean", "median", "std", "max"]:
            print(f"{metric.capitalize():<15} {baseline_fin.get(metric, 0):>20.6f} {good_fin[metric]:>20.6f} {bad_fin[metric]:>20.6f}")
    else:
        print("Financial features not available in this dataset")


def print_partition_results(baseline_results, good_results, bad_results, partition_name):
    """Print partition test comparison for all 3 models."""
    baseline_parts = baseline_results.get("partitions", {})
    good_parts = good_results.get("partitions", {})
    bad_parts = bad_results.get("partitions", {})
    
    baseline_test = baseline_parts.get(partition_name)
    good_test = good_parts.get(partition_name)
    bad_test = bad_parts.get(partition_name)
    
    if not baseline_test or not good_test or not bad_test:
        return
    
    if "error" in good_test or "error" in bad_test:
        print(f"\n{partition_name.upper()}: {good_test.get('error', bad_test.get('error'))}")
        return
    
    if "note" in good_test or "note" in bad_test:
        print(f"\n{partition_name.upper()}: {good_test.get('note', bad_test.get('note'))}")
        return
    
    print(f"\n{partition_name.upper()}")
    print("-"*90)
    
    baseline_groups = baseline_test.get("groups", {})
    good_groups = good_test.get("groups", {})
    bad_groups = bad_test.get("groups", {})
    
    all_groups = set(baseline_groups.keys()) | set(good_groups.keys()) | set(bad_groups.keys())
    
    print(f"{'Group':<20} {'Baseline':>20} {'Good':>20} {'Bad':>20}")
    for group in sorted(all_groups):
        baseline_mean = baseline_groups.get(group, {}).get("mean_score", 0)
        good_mean = good_groups.get(group, {}).get("mean_score", 0)
        bad_mean = bad_groups.get(group, {}).get("mean_score", 0)
        print(f"{group:<20} {baseline_mean:>20.6f} {good_mean:>20.6f} {bad_mean:>20.6f}")


def main():
    """Run complete experiment pipeline."""
    
    base_dir = Path(__file__).resolve().parents[0]
    data_dir = base_dir.parent / "data"
    model_dir = base_dir.parent / "models"
    results_dir = base_dir.parent / "results"
    
    # Create results directory
    results_dir.mkdir(exist_ok=True)
    
    # Create timestamped output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"experiment_results_{timestamp}.txt"
    
    # Redirect output to both console and file
    tee = TeeOutput(output_file)
    sys.stdout = tee
    
    try:
        train_data = data_dir / "synth_data_for_training.csv"
        test_data = data_dir / "investigation_train_large_checked.csv"
        
        print("="*90)
        print("BIAS DETECTION VIA METAMORPHIC TESTING")
        print("="*90)
        print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results saved to: {output_file}")
        print("="*90)
        
        # =====================================================================
        # STEP 1: TRAIN MODELS
        # =====================================================================
        print("\n[1/3] TRAINING MODELS")
        print("-"*90)
        
        X, y = load_data(train_data)
        print(f"Loaded {len(X):,} training samples, {X.shape[1]} features")
        
        print("\nTraining baseline model...")
        baseline_model, baseline_acc = train_baseline_model(X, y)
        
        print("Training good model...")
        good_model, good_acc = train_good_model(X, y)
        
        print("Training bad model...")
        bad_model, bad_acc = train_bad_model(X, y)
        
        print("\nExporting to ONNX...")
        export_to_onnx(baseline_model, X, model_dir / "baselineModel.onnx")
        export_to_onnx(good_model, X, model_dir / "goodModel.onnx")
        export_to_onnx(bad_model, X, model_dir / "badModel.onnx")
        
        # =====================================================================
        # STEP 2: RUN TESTS
        # =====================================================================
        print("\n[2/3] RUNNING TESTS")
        print("-"*90)
        
        print("\nTesting baseline model...")
        baseline_results = run_all_tests(
            model_dir / "baselineModel.onnx",
            test_data,
        )
        
        print("Testing good model...")
        good_results = run_all_tests(
            model_dir / "goodModel.onnx",
            test_data,
        )
        
        print("Testing bad model...")
        bad_results = run_all_tests(
            model_dir / "badModel.onnx",
            test_data,
        )
        
        # =====================================================================
        # STEP 3: PRINT RESULTS
        # =====================================================================
        print("\n[3/3] RESULTS")
        print("="*90)
        
        # Performance summary
        print("\nMODEL ACCURACY")
        print("-"*90)
        print(f"{'Baseline Model':<20} {baseline_acc:>20.4f}")
        print(f"{'Good Model':<20} {good_acc:>20.4f}")
        print(f"{'Bad Model':<20} {bad_acc:>20.4f}")
        
        # Metamorphic test results
        print_metamorphic_results(baseline_results, good_results, bad_results)
        
        # Partition results
        print("\n" + "="*90)
        print("PARTITION TEST RESULTS")
        print("="*90)
        print("\nCompare prediction scores across demographic groups")
        
        print_partition_results(baseline_results, good_results, bad_results, "language")
        print_partition_results(baseline_results, good_results, bad_results, "address")
        print_partition_results(baseline_results, good_results, bad_results, "gender")
        print_partition_results(baseline_results, good_results, bad_results, "neighborhood")
    
    finally:
        # Restore stdout and close file
        sys.stdout = tee.terminal
        tee.close()
        print(f"\n Results saved to: {output_file}")


if __name__ == "__main__":
    main()
