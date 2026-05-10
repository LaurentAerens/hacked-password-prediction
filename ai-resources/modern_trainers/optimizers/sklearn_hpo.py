"""
Phase 1 MVP Trainer: Scikit-learn GridSearchCV with 5-fold cross-validation.

Replaces custom fast_ai_trainer.py and generational_ai_trainer.py with systematic
hyperparameter optimization using sklearn.model_selection.GridSearchCV.

Features:
- 12 models × 8 preprocessors × 5-fold CV
- Stratified K-Fold for imbalanced data
- MLflow integration (stub for p1-model-registry task)
- CSV export of all trials
- Reproducible results (seed control)
- Local parallelization (n_jobs=-1)
"""

import os
import json
from datetime import datetime
from typing import Dict, Tuple, Optional, List

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    ExtraTreesClassifier, BaggingClassifier
)
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import lightgbm as lgbm

# Preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import (
    MaxAbsScaler, MinMaxScaler, StandardScaler, Normalizer, RobustScaler,
    PolynomialFeatures
)
from sklearn.decomposition import PCA, TruncatedSVD


class SklearnHPOTrainer:
    """Systematic hyperparameter optimization using GridSearchCV."""
    
    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        """
        Initialize trainer.
        
        Args:
            random_state: Seed for reproducibility (default: 42)
            n_jobs: Number of parallel jobs (-1 = all cores, default: -1)
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.best_model = None
        self.best_params = None
        self.cv_results = None
        self.experiment_name = None
        
    def _get_models(self) -> Dict:
        """Get model definitions with hyperparameter grids."""
        return {
            'LogisticRegression': {
                'estimator': LogisticRegression(max_iter=1000, random_state=self.random_state),
                'params': {'model__C': [0.001, 0.01, 0.1, 1, 10, 100]}
            },
            'RandomForest': {
                'estimator': RandomForestClassifier(random_state=self.random_state, n_jobs=1),
                'params': {
                    'model__n_estimators': [50, 100, 200],
                    'model__max_depth': [None, 10, 20, 30],
                    'model__min_samples_split': [2, 5, 10]
                }
            },
            'GradientBoosting': {
                'estimator': GradientBoostingClassifier(random_state=self.random_state),
                'params': {
                    'model__n_estimators': [50, 100, 200],
                    'model__learning_rate': [0.01, 0.1, 0.5],
                    'model__max_depth': [3, 5, 7]
                }
            },
            'SVM': {
                'estimator': SVC(random_state=self.random_state, probability=True),
                'params': {
                    'model__C': [0.1, 1, 10, 100],
                    'model__kernel': ['linear', 'rbf']
                }
            },
            'LinearSVM': {
                'estimator': LinearSVC(random_state=self.random_state, max_iter=2000),
                'params': {'model__C': [0.01, 0.1, 1, 10, 100]}
            },
            'KNeighbors': {
                'estimator': KNeighborsClassifier(),
                'params': {'model__n_neighbors': [3, 5, 7, 9, 11]}
            },
            'DecisionTree': {
                'estimator': DecisionTreeClassifier(random_state=self.random_state),
                'params': {
                    'model__max_depth': [None, 5, 10, 20, 30],
                    'model__min_samples_split': [2, 5, 10]
                }
            },
            'AdaBoost': {
                'estimator': AdaBoostClassifier(random_state=self.random_state),
                'params': {
                    'model__n_estimators': [50, 100, 200],
                    'model__learning_rate': [0.01, 0.1, 0.5, 1.0]
                }
            },
            'ExtraTrees': {
                'estimator': ExtraTreesClassifier(random_state=self.random_state, n_jobs=1),
                'params': {
                    'model__n_estimators': [50, 100, 200],
                    'model__max_depth': [None, 10, 20]
                }
            },
            'Bagging': {
                'estimator': BaggingClassifier(random_state=self.random_state, n_jobs=1),
                'params': {'model__n_estimators': [10, 50, 100, 200]}
            },
            'MultinomialNB': {
                'estimator': MultinomialNB(),
                'params': {'model__alpha': [0.1, 0.5, 1.0, 2.0]}
            },
            'XGBoost': {
                'estimator': XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
                'params': {
                    'model__n_estimators': [50, 100, 200],
                    'model__learning_rate': [0.01, 0.1, 0.5],
                    'model__max_depth': [3, 5, 7]
                }
            }
        }
    
    def _get_preprocessors(self) -> Dict:
        """Get preprocessing options."""
        return {
            'CountVectorizer': {
                'step': CountVectorizer(max_features=5000, stop_words='english'),
                'requires_text': True
            },
            'TfidfVectorizer': {
                'step': TfidfVectorizer(max_features=5000, stop_words='english'),
                'requires_text': True
            },
            'StandardScaler': {
                'step': StandardScaler(with_mean=False),
                'requires_text': False
            },
            'MaxAbsScaler': {
                'step': MaxAbsScaler(),
                'requires_text': False
            },
            'MinMaxScaler': {
                'step': MinMaxScaler(),
                'requires_text': False
            },
            'RobustScaler': {
                'step': RobustScaler(),
                'requires_text': False
            },
            'PCA': {
                'step': PCA(n_components=100, random_state=self.random_state),
                'requires_text': False
            },
            'TruncatedSVD': {
                'step': Pipeline([
                    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
                    ('svd', TruncatedSVD(n_components=100, random_state=self.random_state)),
                ]),
                'requires_text': True
            }
        }
    
    def train(self, X, y, models: Optional[List[str]] = None, 
              preprocessors: Optional[List[str]] = None, cv_folds: int = 5) -> Dict:
        """
        Train GridSearchCV over specified models and preprocessors.
        
        Args:
            X: Feature matrix (text or numeric)
            y: Target labels
            models: List of model names to use (default: all)
            preprocessors: List of preprocessor names (default: all)
            cv_folds: Number of CV folds (default: 5)
            
        Returns:
            Dictionary with results, best model, metrics
        """
        all_models = self._get_models()
        all_preprocessors = self._get_preprocessors()
        
        models = models or list(all_models.keys())
        preprocessors = preprocessors or list(all_preprocessors.keys())

        # Password datasets are text by default, so keep only text-compatible preprocessors.
        is_text_input = len(X) > 0 and isinstance(X[0], str)
        if is_text_input:
            preprocessors = [p for p in preprocessors if all_preprocessors[p].get('requires_text', False)]
        
        results = []
        best_auc = -1
        best_pipeline = None
        best_config = None
        
        # Experiment name for tracking
        self.experiment_name = f"sklearn_hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"Starting GridSearchCV: {len(models)} models × {len(preprocessors)} preprocessors")
        print(f"Cross-validation: {cv_folds}-fold stratified")
        
        for prep_name in preprocessors:
            preprocessor = all_preprocessors[prep_name]
            prep_step = preprocessor['step']
            
            for model_name in models:
                model_config = all_models[model_name]
                model_estimator = model_config['estimator']
                param_grid = model_config['params']
                
                try:
                    # Build pipeline
                    pipeline = Pipeline([
                        ('preprocessing', prep_step),
                        ('model', model_estimator)
                    ])
                    
                    # GridSearchCV
                    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
                    grid_search = GridSearchCV(
                        pipeline,
                        param_grid,
                        cv=cv,
                        scoring='roc_auc',
                        n_jobs=self.n_jobs,
                        verbose=0
                    )
                    
                    # Fit
                    grid_search.fit(X, y)
                    
                    # Extract results
                    best_idx = grid_search.best_index_
                    row = grid_search.cv_results_
                    
                    result = {
                        'model': model_name,
                        'preprocessor': prep_name,
                        'best_params': grid_search.best_params_,
                        'best_auc': grid_search.best_score_,
                        'mean_auc': row['mean_test_score'][best_idx],
                        'std_auc': row['std_test_score'][best_idx],
                        'fit_time': row['mean_fit_time'][best_idx]
                    }
                    
                    results.append(result)
                    
                    # Track best
                    if grid_search.best_score_ > best_auc:
                        best_auc = grid_search.best_score_
                        best_pipeline = grid_search.best_estimator_
                        best_config = {
                            'model': model_name,
                            'preprocessor': prep_name,
                            'params': grid_search.best_params_,
                            'auc': grid_search.best_score_
                        }
                    
                    print(f"  {model_name:20} + {prep_name:15} → AUC: {grid_search.best_score_:.4f}")
                    
                except Exception as e:
                    print(f"  {model_name:20} + {prep_name:15} → FAILED: {str(e)[:60]}")
                    continue
        
        # Convert results to DataFrame
        self.cv_results = pd.DataFrame(results)
        self.best_model = best_pipeline
        self.best_params = best_config
        
        # Sort by AUC descending
        self.cv_results = self.cv_results.sort_values('best_auc', ascending=False).reset_index(drop=True)
        
        print(f"\n=== Phase 1 Results ===")
        print(f"Best Model: {best_config['model']} + {best_config['preprocessor']}")
        print(f"Best AUC: {best_auc:.4f}")
        print(f"Total Combinations Tested: {len(results)}")
        
        return {
            'best_model': best_pipeline,
            'best_config': best_config,
            'results_df': self.cv_results,
            'experiment_name': self.experiment_name,
            'metrics': {
                'best_auc': float(best_auc),
                'combinations_tested': len(results),
                'cv_folds': cv_folds
            }
        }
    
    def save_results_csv(self, output_path: str = 'results/phase1_gridsearch_results.csv'):
        """Save results to CSV for manual inspection."""
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        if self.cv_results is not None:
            self.cv_results.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
    
    def save_model(self, output_path: str = 'models/phase1_best_model.pkl'):
        """Save best model to disk."""
        import joblib
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        if self.best_model is not None:
            joblib.dump(self.best_model, output_path)
            print(f"Best model saved to {output_path}")
    
    def predict_proba(self, X):
        """Get predicted probabilities from best model."""
        if self.best_model is None:
            raise ValueError("No model trained yet. Call train() first.")
        return self.best_model.predict_proba(X)
    
    def predict(self, X):
        """Get predictions from best model."""
        if self.best_model is None:
            raise ValueError("No model trained yet. Call train() first.")
        return self.best_model.predict(X)


def train_with_gridsearch(X, y, models: Optional[List[str]] = None,
                         preprocessors: Optional[List[str]] = None,
                         cv_folds: int = 5, n_jobs: int = -1,
                         random_state: int = 42) -> Dict:
    """
    Convenience function: Train models with GridSearchCV.
    
    Args:
        X: Feature matrix
        y: Target labels
        models: List of model names (None = all)
        preprocessors: List of preprocessor names (None = all)
        cv_folds: Number of CV folds
        n_jobs: Number of parallel jobs
        random_state: Random seed
        
    Returns:
        Dictionary with best_model, results_df, experiment_name, metrics
    """
    trainer = SklearnHPOTrainer(random_state=random_state, n_jobs=n_jobs)
    result = trainer.train(X, y, models=models, preprocessors=preprocessors, cv_folds=cv_folds)
    return result


if __name__ == '__main__':
    # Quick test
    print("sklearn_hpo.py loaded successfully")
