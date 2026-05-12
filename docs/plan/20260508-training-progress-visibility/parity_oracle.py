"""
Fallback Parity Oracle for Adaptive Trainer Callback-Absent Behavior

Defines and enforces deterministic assertions that validate equivalent behavior
across Streamlit and CLI surfaces when progress_callback is not provided.

This oracle is MANDATORY for MVP test suite and blocks MVP gate on failure.

Reference: plan.yaml task tpv-004, clarifications resolved via task_clarifications_and_decisions.yaml
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
from dataclasses import dataclass


@dataclass
class ParityValidationError(Exception):
    """Raised when a parity assertion fails."""
    assertion_name: str
    expected: Any
    actual: Any
    context: str = ""

    def __str__(self):
        ctx_str = f" ({self.context})" if self.context else ""
        return (
            f"Parity assertion '{self.assertion_name}' failed{ctx_str}\n"
            f"  Expected: {self.expected}\n"
            f"  Actual: {self.actual}"
        )


class ParityOracle:
    """
    Executable parity oracle for callback-absent trainer behavior.
    
    When progress_callback is None (absent), both Streamlit and CLI paths MUST:
    1. Produce identical result contracts (same keys, types, values)
    2. Report equivalent summary metrics (best_auc, combinations_screened, etc.)
    3. Preserve existing stdout output format (for backward compatibility)
    4. Return same best_model and best_config selections
    
    This oracle is testable, not prose-only. Each assertion is backed by
    deterministic, automatable checks that can be verified in test harness.
    """

    @staticmethod
    def assert_result_contract(result: Dict[str, Any]) -> None:
        """
        ASSERTION: Result dictionary conforms to expected v1 contract.
        
        Required fields:
          - best_model: sklearn Pipeline or estimator
          - best_config: dict with keys {model, preprocessor, params, auc}
          - results_df: pandas DataFrame with per-combination scores
          - experiment_name: str (YYYYMMDD_HHMMSS format)
          - metrics: dict with {best_auc, combinations_screened, combinations_fully_evaluated, cv_folds, screening_folds, top_percent}
        
        Args:
            result: Return dict from train_with_adaptive_search()
            
        Raises:
            ParityValidationError if required field missing or wrong type
        """
        required_fields = {
            'best_model': object,  # sklearn Pipeline
            'best_config': dict,
            'results_df': pd.DataFrame,
            'experiment_name': str,
            'metrics': dict,
        }
        
        for field, expected_type in required_fields.items():
            if field not in result:
                raise ParityValidationError(
                    assertion_name="result_contract_completeness",
                    expected=f"field '{field}' present",
                    actual=f"field '{field}' missing",
                    context=f"available fields: {list(result.keys())}"
                )
            
            if expected_type != object and not isinstance(result[field], expected_type):
                raise ParityValidationError(
                    assertion_name="result_contract_types",
                    expected=f"{field} type={expected_type.__name__}",
                    actual=f"{field} type={type(result[field]).__name__}",
                    context=f"result['{field}']={result[field]}"
                )
        
        # Validate best_config structure
        best_config_fields = {'model', 'preprocessor', 'params', 'auc'}
        if not best_config_fields.issubset(set(result['best_config'].keys())):
            raise ParityValidationError(
                assertion_name="best_config_structure",
                expected=best_config_fields,
                actual=set(result['best_config'].keys()),
                context="best_config must include {model, preprocessor, params, auc}"
            )
        
        # Validate best_config types
        if not isinstance(result['best_config']['model'], str):
            raise ParityValidationError(
                assertion_name="best_config_model_type",
                expected="str",
                actual=type(result['best_config']['model']).__name__,
            )
        if not isinstance(result['best_config']['preprocessor'], str):
            raise ParityValidationError(
                assertion_name="best_config_preprocessor_type",
                expected="str",
                actual=type(result['best_config']['preprocessor']).__name__,
            )
        if not isinstance(result['best_config']['auc'], (int, float)):
            raise ParityValidationError(
                assertion_name="best_config_auc_type",
                expected="int or float",
                actual=type(result['best_config']['auc']).__name__,
            )
        
        # Validate metrics structure
        metrics_fields = {
            'best_auc', 'combinations_screened', 'combinations_fully_evaluated',
            'cv_folds', 'screening_folds', 'top_percent'
        }
        if not metrics_fields.issubset(set(result['metrics'].keys())):
            raise ParityValidationError(
                assertion_name="metrics_structure",
                expected=metrics_fields,
                actual=set(result['metrics'].keys()),
                context="metrics must include {best_auc, combinations_screened, combinations_fully_evaluated, cv_folds, screening_folds, top_percent}"
            )
    
    @staticmethod
    def assert_results_dataframe_valid(results_df: pd.DataFrame) -> None:
        """
        ASSERTION: results_df is sorted by best_auc descending and has expected columns.
        
        Expected columns for each row (Phase 1b full CV results):
          - model: str, model name
          - preprocessor: str, preprocessor name
          - screening_auc: float, Phase 1a screening AUC
          - best_auc: float, Phase 1b full CV AUC (descending order)
          - mean_auc: float
          - std_auc: float
          - best_params: dict
          - fit_time: float
        
        Args:
            results_df: pandas DataFrame returned in result['results_df']
            
        Raises:
            ParityValidationError if structure or ordering violated
        """
        if results_df.empty:
            raise ParityValidationError(
                assertion_name="results_df_not_empty",
                expected="non-empty DataFrame",
                actual="empty DataFrame",
                context="At least one combination should advance to Phase 1b"
            )
        
        required_columns = {
            'model', 'preprocessor', 'screening_auc', 'best_auc',
            'mean_auc', 'std_auc', 'best_params', 'fit_time'
        }
        
        actual_columns = set(results_df.columns)
        if not required_columns.issubset(actual_columns):
            raise ParityValidationError(
                assertion_name="results_df_columns",
                expected=required_columns,
                actual=actual_columns,
                context="missing columns"
            )
        
        # Check descending sort by best_auc
        if not results_df['best_auc'].is_monotonic_decreasing:
            raise ParityValidationError(
                assertion_name="results_df_sort_order",
                expected="best_auc is monotonically decreasing",
                actual=f"best_auc order: {results_df['best_auc'].tolist()[:5]}...",
                context="results_df must be sorted by best_auc descending"
            )
    
    @staticmethod
    def assert_metrics_consistency(result: Dict[str, Any]) -> None:
        """
        ASSERTION: Metrics are internally consistent and match results_df.
        
        - best_auc matches first row of results_df
        - combinations_fully_evaluated matches len(results_df)
        - combinations_screened >= combinations_fully_evaluated
        - cv_folds and screening_folds are positive integers
        - top_percent is between 0.0 and 1.0
        
        Args:
            result: Return dict from train_with_adaptive_search()
            
        Raises:
            ParityValidationError if consistency violated
        """
        metrics = result['metrics']
        results_df = result['results_df']
        
        # best_auc should match top result
        if not results_df.empty:
            expected_best_auc = float(results_df.iloc[0]['best_auc'])
            actual_best_auc = float(metrics['best_auc'])
            if abs(expected_best_auc - actual_best_auc) > 1e-6:
                raise ParityValidationError(
                    assertion_name="metrics_best_auc_consistency",
                    expected=expected_best_auc,
                    actual=actual_best_auc,
                    context="best_auc should match top result in results_df"
                )
        
        # combinations_fully_evaluated should match len(results_df)
        expected_count = len(results_df)
        actual_count = metrics['combinations_fully_evaluated']
        if expected_count != actual_count:
            raise ParityValidationError(
                assertion_name="metrics_combinations_fully_evaluated_consistency",
                expected=expected_count,
                actual=actual_count,
                context="combinations_fully_evaluated should equal len(results_df)"
            )
        
        # combinations_screened >= combinations_fully_evaluated
        if metrics['combinations_screened'] < metrics['combinations_fully_evaluated']:
            raise ParityValidationError(
                assertion_name="metrics_combinations_screened_consistency",
                expected=f">= {metrics['combinations_fully_evaluated']}",
                actual=metrics['combinations_screened'],
                context="combinations_screened should be >= combinations_fully_evaluated (top_percent filter)"
            )
        
        # cv_folds should be positive
        if not isinstance(metrics['cv_folds'], int) or metrics['cv_folds'] < 1:
            raise ParityValidationError(
                assertion_name="metrics_cv_folds_valid",
                expected=">= 1",
                actual=metrics['cv_folds'],
            )
        
        # screening_folds should be 2 (Phase 1a constant)
        if metrics['screening_folds'] != 2:
            raise ParityValidationError(
                assertion_name="metrics_screening_folds_phase1a_constant",
                expected=2,
                actual=metrics['screening_folds'],
                context="Phase 1a always uses 2-fold CV"
            )
        
        # top_percent should be valid
        if not isinstance(metrics['top_percent'], (int, float)) or metrics['top_percent'] <= 0 or metrics['top_percent'] > 1.0:
            raise ParityValidationError(
                assertion_name="metrics_top_percent_valid",
                expected="0.0 < top_percent <= 1.0",
                actual=metrics['top_percent'],
            )
    
    @staticmethod
    def assert_equivalence_across_surfaces(
        streamlit_result: Dict[str, Any],
        cli_result: Dict[str, Any]
    ) -> None:
        """
        ASSERTION: Streamlit and CLI callback-absent executions produce equivalent outcomes.
        
        Equivalence is defined as:
        1. Same best_config (model, preprocessor, params, auc)
        2. Same combinations_screened and combinations_fully_evaluated counts
        3. Same best_auc value (within float tolerance 1e-4)
        4. Same top-5 results ranking by model+preprocessor+auc
        
        This ensures that when callback is absent, both surfaces make identical
        model selection decisions and report equivalent metrics.
        
        Args:
            streamlit_result: Result dict from Streamlit execution (callback=None)
            cli_result: Result dict from CLI execution (callback=None)
            
        Raises:
            ParityValidationError if equivalence violated
        """
        # Compare best_config (deterministic model selection)
        sl_best = streamlit_result['best_config']
        cli_best = cli_result['best_config']
        
        if sl_best['model'] != cli_best['model']:
            raise ParityValidationError(
                assertion_name="equivalence_best_model",
                expected=cli_best['model'],
                actual=sl_best['model'],
                context="Streamlit and CLI should select same best model"
            )
        
        if sl_best['preprocessor'] != cli_best['preprocessor']:
            raise ParityValidationError(
                assertion_name="equivalence_best_preprocessor",
                expected=cli_best['preprocessor'],
                actual=sl_best['preprocessor'],
                context="Streamlit and CLI should select same preprocessor"
            )
        
        # Compare AUC (with float tolerance)
        sl_auc = float(sl_best['auc'])
        cli_auc = float(cli_best['auc'])
        if abs(sl_auc - cli_auc) > 1e-4:
            raise ParityValidationError(
                assertion_name="equivalence_best_auc",
                expected=cli_auc,
                actual=sl_auc,
                context=f"Difference exceeds tolerance (1e-4)"
            )
        
        # Compare metrics
        sl_metrics = streamlit_result['metrics']
        cli_metrics = cli_result['metrics']
        
        if sl_metrics['combinations_screened'] != cli_metrics['combinations_screened']:
            raise ParityValidationError(
                assertion_name="equivalence_combinations_screened",
                expected=cli_metrics['combinations_screened'],
                actual=sl_metrics['combinations_screened'],
                context="Streamlit and CLI should screen same number of combinations"
            )
        
        if sl_metrics['combinations_fully_evaluated'] != cli_metrics['combinations_fully_evaluated']:
            raise ParityValidationError(
                assertion_name="equivalence_combinations_fully_evaluated",
                expected=cli_metrics['combinations_fully_evaluated'],
                actual=sl_metrics['combinations_fully_evaluated'],
                context="Streamlit and CLI should fully evaluate same number"
            )
        
        # Compare top-5 results (same ranking)
        sl_top5 = streamlit_result['results_df'].head(5).reset_index(drop=True)
        cli_top5 = cli_result['results_df'].head(5).reset_index(drop=True)
        
        for idx in range(min(len(sl_top5), len(cli_top5))):
            sl_row = sl_top5.iloc[idx]
            cli_row = cli_top5.iloc[idx]
            
            if sl_row['model'] != cli_row['model'] or sl_row['preprocessor'] != cli_row['preprocessor']:
                raise ParityValidationError(
                    assertion_name="equivalence_results_ranking",
                    expected=f"Row {idx}: {cli_row['model']} + {cli_row['preprocessor']}",
                    actual=f"Row {idx}: {sl_row['model']} + {sl_row['preprocessor']}",
                    context="Top-5 ranking should be identical"
                )
            
            # Allow small float differences in AUC due to randomness
            auc_diff = abs(float(sl_row['best_auc']) - float(cli_row['best_auc']))
            if auc_diff > 1e-4:
                raise ParityValidationError(
                    assertion_name="equivalence_results_auc",
                    expected=float(cli_row['best_auc']),
                    actual=float(sl_row['best_auc']),
                    context=f"Row {idx} AUC difference exceeds tolerance"
                )
    
    @staticmethod
    def validate_callback_absent_behavior(result: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Run full parity validation suite for single callback-absent execution.
        
        Returns:
            (success: bool, messages: List[str])
            success is True iff all assertions pass
            messages contains assertion results (failures and checks performed)
        """
        messages = []
        
        try:
            ParityOracle.assert_result_contract(result)
            messages.append("✓ Result contract valid")
        except ParityValidationError as e:
            messages.append(f"✗ {str(e)}")
            return False, messages
        
        try:
            ParityOracle.assert_results_dataframe_valid(result['results_df'])
            messages.append("✓ Results DataFrame valid and sorted")
        except ParityValidationError as e:
            messages.append(f"✗ {str(e)}")
            return False, messages
        
        try:
            ParityOracle.assert_metrics_consistency(result)
            messages.append("✓ Metrics internally consistent")
        except ParityValidationError as e:
            messages.append(f"✗ {str(e)}")
            return False, messages
        
        return True, messages
    
    @staticmethod
    def validate_parity_across_surfaces(
        streamlit_result: Dict[str, Any],
        cli_result: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Run full equivalence validation for Streamlit vs CLI callback-absent executions.
        
        Returns:
            (success: bool, messages: List[str])
            success is True iff all equivalence assertions pass
            messages contains validation results
        """
        messages = []
        
        # Validate each result independently first
        sl_ok, sl_msgs = ParityOracle.validate_callback_absent_behavior(streamlit_result)
        messages.extend([f"[Streamlit] {m}" for m in sl_msgs])
        if not sl_ok:
            return False, messages
        
        cli_ok, cli_msgs = ParityOracle.validate_callback_absent_behavior(cli_result)
        messages.extend([f"[CLI] {m}" for m in cli_msgs])
        if not cli_ok:
            return False, messages
        
        # Now validate equivalence
        try:
            ParityOracle.assert_equivalence_across_surfaces(streamlit_result, cli_result)
            messages.append("✓ Streamlit and CLI results equivalent")
        except ParityValidationError as e:
            messages.append(f"✗ {str(e)}")
            return False, messages
        
        return True, messages


if __name__ == "__main__":
    # Simple sanity check - import and run
    print("✓ Parity Oracle module loaded successfully")
    print(f"  - {len([m for m in dir(ParityOracle) if m.startswith('assert')])} assertion methods")
    print(f"  - 2 public validation methods: validate_callback_absent_behavior, validate_parity_across_surfaces")
    print("\nUsage:")
    print("  from parity_oracle import ParityOracle, ParityValidationError")
    print("  ParityOracle.validate_callback_absent_behavior(result)  # Returns (bool, List[str])")
    print("  ParityOracle.validate_parity_across_surfaces(sl_result, cli_result)  # Compares two executions")
