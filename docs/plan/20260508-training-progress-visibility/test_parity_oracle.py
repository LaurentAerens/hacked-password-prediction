"""
Minimal unit test for Parity Oracle assertions (no external dependencies required).

This test verifies that the parity oracle correctly:
1. Validates result contract structure
2. Detects missing or malformed fields
3. Enforces results ordering
4. Validates metrics consistency
5. Detects parity violations

Usage:
    python test_parity_oracle.py
    
Exit code 0 = all assertions passed
Exit code 1 = any assertion failed
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add parity oracle to path
sys.path.insert(0, str(Path(__file__).parent))

from parity_oracle import ParityOracle, ParityValidationError


def create_valid_result(seed: int = 42) -> dict:
    """
    Create a valid result dict that passes all parity assertions.
    """
    np.random.seed(seed)
    
    # Create synthetic results_df (sorted by best_auc descending)
    n_combinations = 10
    aucs = sorted(np.random.uniform(0.70, 0.95, n_combinations), reverse=True)
    
    results_data = {
        'model': ['Model_A', 'Model_B', 'Model_A', 'Model_C', 'Model_B',
                  'Model_C', 'Model_A', 'Model_B', 'Model_C', 'Model_A'],
        'preprocessor': ['Prep_1', 'Prep_1', 'Prep_2', 'Prep_1', 'Prep_2',
                         'Prep_2', 'Prep_3', 'Prep_3', 'Prep_3', 'Prep_1'],
        'screening_auc': np.random.uniform(0.65, 0.90, n_combinations),
        'best_auc': aucs,
        'mean_auc': aucs - np.random.uniform(0.0, 0.02, n_combinations),
        'std_auc': np.random.uniform(0.01, 0.05, n_combinations),
        'best_params': [{'C': 1.0} for _ in range(n_combinations)],
        'fit_time': np.random.uniform(0.5, 5.0, n_combinations),
    }
    
    results_df = pd.DataFrame(results_data).sort_values('best_auc', ascending=False).reset_index(drop=True)
    best_auc = float(results_df.iloc[0]['best_auc'])
    
    result = {
        'best_model': f"<Pipeline object>",  # Mock object
        'best_config': {
            'model': str(results_df.iloc[0]['model']),
            'preprocessor': str(results_df.iloc[0]['preprocessor']),
            'params': {'C': 1.0},
            'auc': best_auc,
        },
        'results_df': results_df,
        'experiment_name': f"sklearn_hpo_adaptive_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'metrics': {
            'best_auc': best_auc,
            'combinations_screened': 96,  # 12 models * 8 preprocessors
            'combinations_fully_evaluated': n_combinations,
            'cv_folds': 5,
            'screening_folds': 2,
            'top_percent': 0.20,
        }
    }
    
    return result


def test_result_contract():
    """Test: Result contract validation passes for valid result."""
    print("Test 1: Result contract validation")
    result = create_valid_result(seed=42)
    
    try:
        ParityOracle.assert_result_contract(result)
        print("  ✓ Valid result passes contract check")
        return True
    except ParityValidationError as e:
        print(f"  ✗ FAILED: {e}")
        return False


def test_result_contract_missing_field():
    """Test: Result contract validation fails for missing field."""
    print("Test 2: Result contract validation (missing field)")
    result = create_valid_result(seed=42)
    del result['metrics']  # Remove required field
    
    try:
        ParityOracle.assert_result_contract(result)
        print("  ✗ FAILED: Should have detected missing 'metrics' field")
        return False
    except ParityValidationError as e:
        if "metrics" in str(e) and "missing" in str(e):
            print("  ✓ Correctly detected missing 'metrics' field")
            return True
        else:
            print(f"  ✗ FAILED: Wrong error message: {e}")
            return False


def test_result_contract_wrong_type():
    """Test: Result contract validation fails for wrong type."""
    print("Test 3: Result contract validation (wrong type)")
    result = create_valid_result(seed=42)
    result['results_df'] = "not a dataframe"  # Wrong type
    
    try:
        ParityOracle.assert_result_contract(result)
        print("  ✗ FAILED: Should have detected wrong type for results_df")
        return False
    except ParityValidationError as e:
        if "type" in str(e) and "results_df" in str(e):
            print("  ✓ Correctly detected wrong type for results_df")
            return True
        else:
            print(f"  ✗ FAILED: Wrong error message: {e}")
            return False


def test_dataframe_ordering():
    """Test: Results DataFrame ordering validation."""
    print("Test 4: Results DataFrame ordering")
    result = create_valid_result(seed=42)
    
    try:
        ParityOracle.assert_results_dataframe_valid(result['results_df'])
        print("  ✓ DataFrame correctly sorted by best_auc descending")
        return True
    except ParityValidationError as e:
        print(f"  ✗ FAILED: {e}")
        return False


def test_dataframe_wrong_order():
    """Test: Results DataFrame ordering fails for unsorted data."""
    print("Test 5: Results DataFrame ordering (unsorted)")
    result = create_valid_result(seed=42)
    result['results_df'] = result['results_df'].iloc[::-1].reset_index(drop=True)  # Reverse sort
    
    try:
        ParityOracle.assert_results_dataframe_valid(result['results_df'])
        print("  ✗ FAILED: Should have detected wrong sort order")
        return False
    except ParityValidationError as e:
        if "sort" in str(e) or "monotonic" in str(e):
            print("  ✓ Correctly detected wrong sort order")
            return True
        else:
            print(f"  ✗ FAILED: Wrong error message: {e}")
            return False


def test_metrics_consistency():
    """Test: Metrics consistency validation."""
    print("Test 6: Metrics consistency")
    result = create_valid_result(seed=42)
    
    try:
        ParityOracle.assert_metrics_consistency(result)
        print("  ✓ Metrics are internally consistent")
        return True
    except ParityValidationError as e:
        print(f"  ✗ FAILED: {e}")
        return False


def test_metrics_inconsistency():
    """Test: Metrics consistency fails for inconsistent values."""
    print("Test 7: Metrics consistency (inconsistent)")
    result = create_valid_result(seed=42)
    result['metrics']['best_auc'] = 0.50  # Inconsistent with results_df
    
    try:
        ParityOracle.assert_metrics_consistency(result)
        print("  ✗ FAILED: Should have detected inconsistent best_auc")
        return False
    except ParityValidationError as e:
        if "best_auc" in str(e):
            print("  ✓ Correctly detected inconsistent best_auc")
            return True
        else:
            print(f"  ✗ FAILED: Wrong error message: {e}")
            return False


def test_equivalence():
    """Test: Equivalence validation for identical results."""
    print("Test 8: Equivalence across surfaces (identical)")
    result1 = create_valid_result(seed=42)
    result2 = create_valid_result(seed=42)  # Same seed = same results
    
    try:
        ParityOracle.assert_equivalence_across_surfaces(result1, result2)
        print("  ✓ Identical results pass equivalence check")
        return True
    except ParityValidationError as e:
        print(f"  ✗ FAILED: {e}")
        return False


def test_equivalence_different_models():
    """Test: Equivalence fails when best models differ."""
    print("Test 9: Equivalence across surfaces (different models)")
    result1 = create_valid_result(seed=42)
    result2 = create_valid_result(seed=43)  # Different seed = different results
    result2['best_config']['model'] = 'Different_Model'
    
    try:
        ParityOracle.assert_equivalence_across_surfaces(result1, result2)
        print("  ✗ FAILED: Should have detected different best models")
        return False
    except ParityValidationError as e:
        if "model" in str(e):
            print("  ✓ Correctly detected different best models")
            return True
        else:
            print(f"  ✗ FAILED: Wrong error message: {e}")
            return False


def test_single_result_validation():
    """Test: Single result validation passes."""
    print("Test 10: Single result validation (callback_absent)")
    result = create_valid_result(seed=42)
    
    ok, messages = ParityOracle.validate_callback_absent_behavior(result)
    
    if ok and len(messages) > 0:
        print(f"  ✓ Single result validation passed")
        for msg in messages:
            print(f"    - {msg}")
        return True
    else:
        print(f"  ✗ FAILED: {messages}")
        return False


def test_parity_validation():
    """Test: Full parity validation across surfaces."""
    print("Test 11: Full parity validation (Streamlit vs CLI)")
    result1 = create_valid_result(seed=42)  # Streamlit
    result2 = create_valid_result(seed=42)  # CLI (same seed)
    
    ok, messages = ParityOracle.validate_parity_across_surfaces(result1, result2)
    
    if ok and len(messages) > 0:
        print(f"  ✓ Full parity validation passed")
        for msg in messages:
            print(f"    - {msg}")
        return True
    else:
        print(f"  ✗ FAILED: {messages}")
        return False


def main():
    print(f"\n{'='*70}")
    print("PARITY ORACLE UNIT TESTS")
    print(f"{'='*70}\n")
    
    tests = [
        test_result_contract,
        test_result_contract_missing_field,
        test_result_contract_wrong_type,
        test_dataframe_ordering,
        test_dataframe_wrong_order,
        test_metrics_consistency,
        test_metrics_inconsistency,
        test_equivalence,
        test_equivalence_different_models,
        test_single_result_validation,
        test_parity_validation,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"  ✗ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
        print()
    
    passed = sum(results)
    total = len(results)
    
    print(f"{'='*70}")
    print(f"RESULTS: {passed}/{total} tests passed")
    print(f"{'='*70}\n")
    
    if all(results):
        print("✓ All parity oracle assertions are working correctly")
        print("\nThe parity oracle can be used in MVP test suite to validate:")
        print("  1. Result contract completeness and types")
        print("  2. Results DataFrame ordering (best_auc descending)")
        print("  3. Metrics consistency with DataFrame")
        print("  4. Equivalence across Streamlit and CLI surfaces")
        return 0
    else:
        print("✗ Some tests failed. Review output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
