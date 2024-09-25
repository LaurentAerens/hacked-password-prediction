import os
from datetime import datetime

import joblib
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import (GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.svm import LinearSVC, SVC
from xgboost import XGBClassifier

# List of preprocessing options
preprocessing_options = {
    'StandardScaler': StandardScaler(with_mean=False),
    'MaxAbsScaler': MaxAbsScaler(),
    'TruncatedSVD': TruncatedSVD(n_components=100)
}

# List of models
models = {
    'LogisticRegression': LogisticRegression(max_iter=100, n_jobs=-1),
    'RandomForest': RandomForestClassifier(n_estimators=100, n_jobs=-1),
    'SVM': SVC(probability=True, max_iter=100),
    'LinearSVM': LinearSVC(max_iter=100),
    'KNeighbors': KNeighborsClassifier(n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

def get_data(file_path='data/combined_data.csv'):
    """
    Load data from a CSV file into a pandas DataFrame and remove all null values.
    
    Parameters:
    file_path (str): The path to the CSV file.
    
    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    print(f"Loading data from {file_path}...")
    data = pd.read_csv(file_path)
    print("Removing null values...")
    data = data.dropna()
    return data

def train_model(X_train, X_test, y_train, y_test, model, preprocessing=None):
    """
    Train the given model on the provided data, optionally applying a single preprocessing step.
    
    Parameters:
    X_train, X_test, y_train, y_test: The training and testing data.
    model: The model to train.
    preprocessing: Optional single preprocessing step to apply.
    
    Returns:
    None
    """
    if preprocessing:
        pipeline = Pipeline([
            ('preprocessing', preprocessing),
            ('model', model)
        ])
        preprocessing_name = preprocessing.__class__.__name__
    else:
        pipeline = Pipeline([
            ('model', model)
        ])
        preprocessing_name = 'None'
    
    print(f"Training model {model.__class__.__name__} with preprocessing {preprocessing_name}...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC score for {model.__class__.__name__} with preprocessing {preprocessing_name}: {auc_score:.4f}")
    
    # Save the trained model
    model_name = model.__class__.__name__
    date_str = datetime.now().strftime("%Y%m%d%H%M%S")
    if preprocessing_name == 'None':
        filename = f"models/fast-{model_name}-{date_str}-{auc_score:.4f}.joblib"
    else:
        filename = f"models/fast-{model_name}-{preprocessing_name}-{date_str}-{auc_score:.4f}.joblib"
    joblib.dump(pipeline, filename)
    print(f"Model saved as {filename}")

def main():
    # step 0: create models directory
    if not os.path.exists('models'):
        os.makedirs('models')
        
    # Step 1: Get the data
    data = get_data()

    # Step 2: Check if vectorizer exists, if not create a new one
    vectorizer_path = 'models/vectorizer.joblib'
    if os.path.exists(vectorizer_path):
        print("Loading existing vectorizer...")
        vectorizer = joblib.load(vectorizer_path)
    else:
        print("Creating new vectorizer...")
        vectorizer = CountVectorizer()
        vectorizer.fit(data['password'])
        joblib.dump(vectorizer, vectorizer_path)
        print("Vectorizer saved.")

    # Transform the data
    X = vectorizer.transform(data['password'])
    y = data['target']

    # Save the transformed data
    joblib.dump(X, 'models/X.joblib')
    joblib.dump(y, 'models/y.joblib')
    print("Transformed data saved.")

    # Step 3: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Let the user choose models and preprocessing
    selected_combinations = []
    while True:
        print("Available models:")
        for i, model_name in enumerate(models.keys(), 1):
            print(f"{i}. {model_name}")
        print("0. All models with all preprocessing options and without preprocessing")
        model_index = input("Enter the number of the model you want to run (or press Enter to finish): ")
        if not model_index:
            break
        if model_index == '0':
            for model in models.values():
                for preprocessing in preprocessing_options.values():
                    selected_combinations.append((model, preprocessing))
                selected_combinations.append((model, None))
            break
        selected_model = list(models.values())[int(model_index) - 1]

        print("\nAvailable preprocessing options:")
        for i, preprocessing_name in enumerate(preprocessing_options.keys(), 1):
            print(f"{i}. {preprocessing_name}")
        preprocessing_index = input("Enter the number of the preprocessing option you want to use (or press Enter to skip): ")
        selected_preprocessing = preprocessing_options[list(preprocessing_options.keys())[int(preprocessing_index) - 1]] if preprocessing_index else None

        selected_combinations.append((selected_model, selected_preprocessing))

    # Step 5: Train the selected models with the chosen preprocessing
    for model, preprocessing in selected_combinations:
        train_model(X_train, X_test, y_train, y_test, model, preprocessing)

if __name__ == "__main__":
    main()