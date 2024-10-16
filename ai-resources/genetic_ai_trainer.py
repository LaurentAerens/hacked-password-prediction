# Rewrite of the generational_ai_trainer.py script because that script started to look like a mess

# let's start by defining the steps the program need to go through for a training session so we can cross them off as we go
# 1. Basic setup (importing libraries, creating the directory structure, loading the data & vectorizer, split the data)
# 2. Time for the genetic loop
# step 1: create the initial population
# step 2: evaluate for all the predefined hyperparameters
# step 3: create a new initial population based on the best performing models (we should make sure that each model/preprocessor combination is represented in the new population)
# step 4: we use the mutate to tweak the hyperparameters of the models
# step 5: we evaluate the new population where the higher you are the more likely you are to be selected for the next generation (but everyone has a chance)
# step 6: we build the new population based on the best performing models, where the best models have several children
# 3. make sure every model is saved in the models folder, we keep track of the best model update the best model logs, update the hyperparameters json, and keep general logs
# 4. maybe we should also have a way to stop the training session and resume it later (and even have an auto save feature)

import logging
import os
import random
import shutil
import json
import atexit
from collections import defaultdict
from datetime import datetime
from typing import Dict

import joblib
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import ParameterSampler, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, Normalizer,
                                   PolynomialFeatures, RobustScaler,
                                   StandardScaler)
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from shared_lib.logger import create_logger
from shared_lib.data_utils import get_data, split_data
from shared_lib.model_utils import load_model_storage, save_model_storage, get_vectorizer, create_models_directory, transform_data

#region  Define the variables and classes that will be used throughout the script

# List of preprocessing options
preprocessing_options = {
    'StandardScaler': StandardScaler(with_mean=False),
    'MaxAbsScaler': MaxAbsScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'Normalizer': Normalizer(),
    'RobustScaler': RobustScaler(),
    'PolynomialFeatures': PolynomialFeatures(degree=2, include_bias=False),
    'TruncatedSVD': TruncatedSVD(n_components=100),
    'PCA': PCA(n_components=100)
}

# List of models and their hyperparameter spaces
models = {
    'LogisticRegression': (LogisticRegression, {'C': [0.01, 0.1, 1, 10, 100]}),
    'RandomForest': (RandomForestClassifier, {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}),
    'SVM': (SVC, {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}),
    'LinearSVM': (LinearSVC, {'C': [0.01, 0.1, 1, 10, 100]}),
    'KNeighbors': (KNeighborsClassifier, {'n_neighbors': [3, 5, 7, 9]}),
    'GradientBoosting': (GradientBoostingClassifier, {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}),
    'XGBoost': (XGBClassifier, {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}),
    'DecisionTree': (DecisionTreeClassifier, {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 10, 20]}),
    'AdaBoost': (AdaBoostClassifier, {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}),
    'ExtraTrees': (ExtraTreesClassifier, {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}),
    'Bagging': (BaggingClassifier, {'n_estimators': [10, 50, 100]}),
    'MLP': (MLPClassifier, {'hidden_layer_sizes': [(100,), (50, 50), (30, 30, 30)], 'activation': ['relu', 'tanh'], 'max_iter': [200, 500]})
}

# Initialize a dictionary to store hyperparameters and AUC scores
hyperparam_scores = defaultdict(lambda: defaultdict(dict))

model_storage_path = 'models/model_storage.json'

class TimeoutException(Exception):
    pass

#endregion

#region Support functions

def save_hyperparam_scores():
    os.makedirs('tuning', exist_ok=True)
    for model_preprocessing, scores in hyperparam_scores.items():
        file_path = f'tuning/{model_preprocessing}.json'
        with open(file_path, 'w') as f:
            json.dump(scores, f)
        logger.info(f"Saved hyperparameter scores to {file_path}")

#endregion


#region initial setup section
def base_setup(file_path_data='data/combined_data.csv'):
    global logger
    logger, warning_logger = create_logger()
    # log the start of the experiment
    logger.info("Starting the experiment at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    create_models_directory(logger=logger)
    data = get_data(file_path_data, logger=logger)
    vectorizer, warning_vectorize = get_vectorizer(data, logger=logger)
    X, y = transform_data(vectorizer, data, logger=logger)
    X_train, X_test, y_train, y_test = split_data(X, y, logger=logger)
    model_storage = load_model_storage(model_storage_path=model_storage_path, logger=logger)
    atexit.register(lambda: save_model_storage(model_storage, logger=logger, model_storage_path=model_storage_path))
    warnings = warning_logger + warning_vectorize
    return X_train, X_test, y_train, y_test, vectorizer, model_storage, warnings
#endregion

#region Setup Experiment

def experiment_setup(experiment_name):
    date_str = datetime.now().strftime("%Y%m%d%H%M%S")
    experiment_dir = f"models/experiment-{experiment_name}-{date_str}"
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'best-model'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'hyperparameters'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'checkpoints'), exist_ok=True)
    logger.info(f"Experiment directories created: {experiment_dir}")

#endregion

#region Genetic Loop
def evolutionary_training(X_train, X_test, y_train, y_test, experiment_dir, survival_rate=0.3, additional_population_factor=2):
    """
    Perform evolutionary training of AI models.
    
    Parameters:
    X_train, X_test, y_train, y_test: The training and testing data.
    experiment_dir (str): The directory to save the experiment results.
    survival_rate (float): The rate of models to survive each generation.
    n_iter (int): The number of iterations for hyperparameter tuning.
    
    Returns:
    None
    """
    # Calculate the total number of models
    total_models = len(models) * (len(preprocessing_options) + 1)

    # step 1: run the all predefined hyperparameters save all the AUC scores so we can use that to create the initial population
    for model_name, (model, hyperparameters) in models.items():
        for preprocessing_name, preprocessing in preprocessing_options.items():
            model_preprocessing = f"{model_name}_{preprocessing_name}"
            if model_preprocessing not in hyperparam_scores:
                hyperparam_scores[model_preprocessing] = {}
            if hyperparameters:
                for hyperparameter in ParameterSampler(hyperparameters, n_iter=10, random_state=42):
                    model_instance = model(**hyperparameter)
                    pipeline = Pipeline([('preprocessing', preprocessing), ('model', model_instance)])
                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)
                    auc = roc_auc_score(y_test, y_pred)
                    hyperparam_scores[model_preprocessing][str(hyperparameter)] = auc
                    logger.info(f"Model: {model_name}, Preprocessing: {preprocessing_name}, Hyperparameters: {hyperparameter}, AUC: {auc}")

    # Save the hyperparameter scores
    save_hyperparam_scores()

    # step 2: create the initial population (exiting out of the best hyperparameters for each model/preprocessor combination)
    initial_population = []
    initial_population_set = set()  # To keep track of models in the initial population
    for model_name, (model, hyperparameters) in models.items():
        for preprocessing_name, preprocessing in preprocessing_options.items():
            model_preprocessing = f"{model_name}_{preprocessing_name}"
            best_hyperparameters = max(hyperparam_scores[model_preprocessing], key=hyperparam_scores[model_preprocessing].get)
            best_hyperparameters = eval(best_hyperparameters)
            model_instance = model(**best_hyperparameters)
            pipeline = Pipeline([('preprocessing', preprocessing), ('model', model_instance)])
            initial_population.append(pipeline)
            initial_population_set.add((model_name, preprocessing_name, str(best_hyperparameters)))

    # step 3: calculate the additional models we can add to the population (these are the best AUC scores that are not already in the population)
    additional_models_count = int(total_models * additional_population_factor) - len(initial_population)
    additional_models = []

    # Flatten the hyperparam_scores dictionary and sort by AUC score
    sorted_hyperparam_scores = sorted(
        [(model_preprocessing, hyperparameters, auc) for model_preprocessing, scores in hyperparam_scores.items() for hyperparameters, auc in scores.items()],
        key=lambda x: x[2],  # Sort by AUC score
        reverse=True  # Descending order
    )

    # Select the top additional models that are not already in the initial population
    for model_preprocessing, hyperparameters, auc in sorted_hyperparam_scores:
        model_name, preprocessing_name = model_preprocessing.split('_', 1)
        if (model_name, preprocessing_name, hyperparameters) not in initial_population_set:
            model_instance = models[model_name][0](**eval(hyperparameters))
            preprocessing = preprocessing_options[preprocessing_name]
            pipeline = Pipeline([('preprocessing', preprocessing), ('model', model_instance)])
            additional_models.append(pipeline)
            if len(additional_models) >= additional_models_count:
                break

    # Combine initial population and additional models
    final_population = initial_population + additional_models



#endregion

#region main loop section
def main_loop(experiment_name, data_file_path='data/combined_data.csv', survival_rate=0.3, additional_population_factor=2):
    warnings = []
    X_train, X_test, y_train, y_test, vectorizer, model_storage, warning = base_setup(data_file_path)
    experiment_setup(experiment_name)
    warnings.extend(warning)

    return warnings
def main():
    experiment_name = input("Enter the experiment name: ")
    main_loop(experiment_name)
if __name__ == "__main__":
    main()
#endregion