#!/usr/bin/env python3
"""Test mini dataset training with fixed preprocessors."""

import sys
import os

# Add ai-resources to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ai-resources'))

from adaptive_trainer import train_with_adaptive_search

print('Testing mini dataset training with fixed preprocessors...')
print('Expected: No TruncatedSVD failures, no LightGBM + CountVectorizer failures')
print()

try:
    # Simple test data
    X, y = ['password'] * 50 + ['K8@mPq!2xR'] * 50, [0] * 50 + [1] * 50
    
    result = train_with_adaptive_search(
        X, y,
        models=['LogisticRegression', 'LightGBM'],
        preprocessors=None,
        cv_folds=2,
        run_phase2=False,
        mini_dataset=True,
        mini_dataset_size=100,
        n_jobs=1
    )
    
    print('✓ Training completed successfully')
    best_config = result.get('best_config', {})
    best_model = best_config.get('model_name', 'N/A')
    best_auc = result.get('best_auc', 'N/A')
    print(f'  Best model: {best_model}')
    print(f'  Best AUC: {best_auc}')
    
except Exception as e:
    print(f'✗ Error: {e}', file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
