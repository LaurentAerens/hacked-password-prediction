import os
import signal
from datetime import datetime, timedelta

import joblib
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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
    
    Raises:
    FileNotFoundError: If the file does not exist.
    ValueError: If the file is not in CSV format or if the data is empty after removing null values.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    if not file_path.endswith('.csv'):
        raise ValueError(f"The file {file_path} is not in CSV format.")
    
    print(f"Loading data from {file_path}...")
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Error reading the CSV file: {e}")
    
    print("Removing null values...")
    data = data.dropna()
    
    if data.empty:
        raise ValueError("The data is empty after removing null values.")
    
    return data


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Training exceeded the time limit of 24 hours.")


def train_model(X_train, X_test, y_train, y_test, model, preprocessing=None):
    """
    Train the given model on the provided data, optionally applying a single preprocessing step.
    
    Parameters:
    X_train, X_test, y_train, y_test: The training and testing data.
    model: The model to train.
    preprocessing: Optional single preprocessing step to apply.
    
    Returns:
    None
    
    Raises:
    ValueError: If any of the inputs are empty or None.
    TimeoutException: If training exceeds the time limit of 24 hours.
    """
    # Validate inputs
    if X_train is None or X_test is None or y_train is None or y_test is None or model is None:
        raise ValueError("Training and testing data and model must not be None.")
    if len(X_train) == 0 or len(X_test) == 0 or len(y_train) == 0 or len(y_test) == 0:
        raise ValueError("Training and testing data must not be empty.")
    
    # Set up timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(24 * 3600)  # Set timeout for 24 hours
    
    try:
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
        
        if os.path.exists(filename):
            print(f"Warning: The file {filename} already exists and will be overwritten.")
        
        joblib.dump(pipeline, filename)
        print(f"Model saved as {filename}")
    
    except TimeoutException as e:
        print(f"Error: {e}")
    finally:
        # Disable the alarm
        signal.alarm(0)


def create_models_directory():
    """
    Create the models directory if it does not exist.
    """
    if not os.path.exists('models'):
        os.makedirs('models')


def get_vectorizer(data):
    """
    Get the vectorizer, creating a new one if it does not exist.
    
    Parameters:
    data (pd.DataFrame): The data to fit the vectorizer on if it needs to be created.
    
    Returns:
    tuple: A tuple containing the vectorizer and a list of warnings.
    
    Raises:
    Warning: If the existing vectorizer is invalid or if saving the new vectorizer fails.
    """
    warnings = []
    vectorizer_path = 'models/vectorizer.joblib'
    if os.path.exists(vectorizer_path):
        try:
            print("Loading existing vectorizer...")
            vectorizer = joblib.load(vectorizer_path)
        except Exception as e:
            warning_msg = f"Warning: Failed to load existing vectorizer. Creating a new one. Error: {e}"
            print(warning_msg)
            warnings.append(warning_msg)
            vectorizer = CountVectorizer()
            vectorizer.fit(data['password'])
            try:
                joblib.dump(vectorizer, vectorizer_path)
                print("Vectorizer saved.")
            except Exception as e:
                warning_msg = f"Warning: Failed to save the new vectorizer. This may cause issues for later usage. Error: {e}"
                print(warning_msg)
                warnings.append(warning_msg)
    else:
        print("Creating new vectorizer...")
        vectorizer = CountVectorizer()
        vectorizer.fit(data['password'])
        try:
            joblib.dump(vectorizer, vectorizer_path)
            print("Vectorizer saved.")
        except Exception as e:
            warning_msg = f"Warning: Failed to save the new vectorizer. This may cause issues for later usage. Error: {e}"
            print(warning_msg)
            warnings.append(warning_msg)
    
    return vectorizer, warnings


def transform_data(vectorizer, data):
    """
    Transform the data using the vectorizer.
    
    Parameters:
    vectorizer (CountVectorizer): The vectorizer to use.
    data (pd.DataFrame): The data to transform.
    
    Returns:
    X (sparse matrix): The transformed data.
    y (pd.Series): The target variable.
    """
    X = vectorizer.transform(data['password'])
    y = data['target']
    
    try:
        joblib.dump(X, 'models/X.joblib')
    except Exception as e:
        print(f"Warning: Failed to save transformed X data. Error: {e}")
    
    try:
        joblib.dump(y, 'models/y.joblib')
    except Exception as e:
        print(f"Warning: Failed to save transformed y data. Error: {e}")
    
    print("Transformed data saved.")
    return X, y


def split_data(X, y):
    """
    Split the data into training and testing sets.
    
    Parameters:
    X (sparse matrix): The feature data.
    y (pd.Series): The target variable.
    
    Returns:
    X_train, X_test, y_train, y_test: The split data.
    """
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_single_model_with_preprocessing(model_name, preprocessing_name):
    """
    Train a single model with a single preprocessing option.
    
    Parameters:
    model_name (str): The name of the model to train.
    preprocessing_name (str): The name of the preprocessing option to use.
    
    Returns:
    list: A list of warnings encountered during the process.
    """
    warnings = []
    create_models_directory()
    data = get_data()
    vectorizer, vectorizer_warnings = get_vectorizer(data)
    warnings.extend(vectorizer_warnings)
    X, y = transform_data(vectorizer, data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    model = models[model_name]
    preprocessing = preprocessing_options.get(preprocessing_name, None)
    
    train_model(X_train, X_test, y_train, y_test, model, preprocessing)
    
    return warnings


def train_list_of_models_with_preprocessing(model_preprocessing_list):
    """
    Train a list of models with their respective preprocessing options.
    
    Parameters:
    model_preprocessing_list (list): A list of tuples containing model names and preprocessing names.
    
    Returns:
    list: A list of warnings encountered during the process.
    """
    warnings = []
    create_models_directory()
    data = get_data()
    vectorizer, vectorizer_warnings = get_vectorizer(data)
    warnings.extend(vectorizer_warnings)
    X, y = transform_data(vectorizer, data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    for model_name, preprocessing_name in model_preprocessing_list:
        model = models[model_name]
        preprocessing = preprocessing_options.get(preprocessing_name, None)
        train_model(X_train, X_test, y_train, y_test, model, preprocessing)
    
    return warnings


def train_all_models_with_preprocessing():
    """
    Train all models with all preprocessing options and without preprocessing.
    
    Returns:
    list: A list of warnings encountered during the process.
    """
    warnings = []
    create_models_directory()
    data = get_data()
    vectorizer, vectorizer_warnings = get_vectorizer(data)
    warnings.extend(vectorizer_warnings)
    X, y = transform_data(vectorizer, data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    for model in models.values():
        for preprocessing in preprocessing_options.values():
            train_model(X_train, X_test, y_train, y_test, model, preprocessing)
        train_model(X_train, X_test, y_train, y_test, model, None)
    
    return warnings


def select_combinations():
    """
    Let the user choose models and preprocessing options.
    
    Returns:
    list: A list of selected model and preprocessing combinations.
    """
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
    return selected_combinations


def train_selected_combinations(X_train, X_test, y_train, y_test, selected_combinations):
    """
    Train the selected models with the chosen preprocessing options.
    
    Parameters:
    X_train, X_test, y_train, y_test: The training and testing data.
    selected_combinations (list): The selected model and preprocessing combinations.
    
    Returns:
    None
    """
    for model, preprocessing in selected_combinations:
        train_model(X_train, X_test, y_train, y_test, model, preprocessing)


def main():
    """
    Main function for manual use.
    """
    try:
        create_models_directory()
        data = get_data()
        vectorizer = get_vectorizer(data)
        X, y = transform_data(vectorizer, data)
        X_train, X_test, y_train, y_test = split_data(X, y)
        selected_combinations = select_combinations()
        train_selected_combinations(X_train, X_test, y_train, y_test, selected_combinations)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()