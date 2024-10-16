import os
import json
import joblib
from typing import Dict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

def load_model_storage(model_storage_path, logger=None) -> Dict[str, str]:
    """
    Load the model storage from a JSON file.
    
    Parameters:
    model_storage_path (str): The path to the model storage JSON file.
    logger (LoggerWrapper): The logger instance to use for logging.
    
    Returns:
    dict: A dictionary containing model names and their paths.
    """
    if os.path.exists(model_storage_path):
        if logger:
            logger.info(f"Loading model storage from {model_storage_path}...")
        with open(model_storage_path, 'r') as file:
            return json.load(file)
    return {}

def save_model_storage(model_storage: Dict[str, str], model_storage_path, logger=None) -> None:
    """
    Save the model storage to a JSON file.
    
    Parameters:
    model_storage (dict): A dictionary containing model names and their paths.
    model_storage_path (str): The path to the model storage JSON file.
    logger (LoggerWrapper): The logger instance to use for logging.
    """
    if logger:
        logger.info(f"Saving model storage to {model_storage_path}...")
    with open(model_storage_path, 'w') as file:
        json.dump(model_storage, file, indent=4)

def get_vectorizer(data, vectorizer_path='models/vectorizer.joblib', logger=None):
    """
    Get the vectorizer, creating a new one if it does not exist.
    
    Parameters:
    data (pd.DataFrame): The data to fit the vectorizer on if it needs to be created.
    vectorizer_path (str): The path to the vectorizer joblib file.
    logger (LoggerWrapper): The logger instance to use for logging.
    
    Returns:
    tuple: A tuple containing the vectorizer and a list of warnings.
    
    Raises:
    Warning: If the existing vectorizer is invalid or if saving the new vectorizer fails.
    """
    warnings = []
    if os.path.exists(vectorizer_path):
        try:
            if logger:
                logger.info(f"Loading existing vectorizer from {vectorizer_path}...")
            vectorizer = joblib.load(vectorizer_path)
        except Exception as e:
            warning_msg = f"Warning: Failed to load existing vectorizer. Creating a new one. Error: {e}"
            warnings.append(warning_msg)
            if logger:
                logger.warning(warning_msg)
            vectorizer = CountVectorizer()
            vectorizer.fit(data['password'])
            try:
                joblib.dump(vectorizer, vectorizer_path)
            except Exception as e:
                warning_msg = f"Warning: Failed to save the new vectorizer. This may cause issues for later usage. Error: {e}"
                warnings.append(warning_msg)
                if logger:
                    logger.warning(warning_msg)
    else:
        vectorizer = CountVectorizer()
        vectorizer.fit(data['password'])
        try:
            joblib.dump(vectorizer, vectorizer_path)
        except Exception as e:
            warning_msg = f"Warning: Failed to save the new vectorizer. This may cause issues for later usage. Error: {e}"
            warnings.append(warning_msg)
            if logger:
                logger.warning(warning_msg)
    
    return vectorizer, warnings

def create_models_directory(logger=None):
    """
    Create the models directory if it does not exist.
    
    Parameters:
    logger (LoggerWrapper): The logger instance to use for logging.
    """
    if not os.path.exists('models'):
        os.makedirs('models')
        if logger:
            logger.info("Created 'models' directory.")

def transform_data(vectorizer, data, logger=None):
    """
    Transform the data using the vectorizer.
    
    Parameters:
    vectorizer (CountVectorizer): The vectorizer to use.
    data (pd.DataFrame): The data to transform.
    logger (LoggerWrapper): The logger instance to use for logging.
    
    Returns:
    X (sparse matrix): The transformed data.
    y (pd.Series): The target variable.
    """
    if logger:
        logger.info("Transforming data using the vectorizer...")
    X = vectorizer.transform(data['password'])
    y = data['target']
    
    try:
        joblib.dump(X, 'models/X.joblib')
        if logger:
            logger.info("Transformed X data saved successfully.")
    except Exception as e:
        warning_msg = f"Warning: Failed to save transformed X data. Error: {e}"
        if logger:
            logger.warning(warning_msg)
    
    try:
        joblib.dump(y, 'models/y.joblib')
        if logger:
            logger.info("Transformed y data saved successfully.")
    except Exception as e:
        warning_msg = f"Warning: Failed to save transformed y data. Error: {e}"
        if logger:
            logger.warning(warning_msg)
    
    return X, y

def train_model(save_folder,X_train, X_test, y_train, y_test, model, preprocessing=None, logger=None):
    """
    Train the given model on the provided data, optionally applying a single preprocessing step.
    
    Parameters:
    X_train, X_test, y_train, y_test: The training and testing data.
    model: The model to train.
    preprocessing: Optional single preprocessing step to apply.
    logger (LoggerWrapper): The logger instance to use for logging.
    
    Returns:
    dict: A dictionary containing the model, preprocessing, and AUC score.
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
    
    if logger:
        logger.info(f"Training model {model.__class__.__name__} with preprocessing {preprocessing_name}...")
    
    try:
        pipeline.fit(X_train, y_train)
    except Exception as e:
        if logger:
            logger.error(f"Error training model {model.__class__.__name__} with preprocessing {preprocessing_name}: {e}")
        raise
    
    # Evaluate the model
    try:
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        if logger:
            logger.info(f"AUC score for {model.__class__.__name__} with preprocessing {preprocessing_name}: {auc_score:.4f}")
    except Exception as e:
        if logger:
            logger.error(f"Error evaluating model {model.__class__.__name__} with preprocessing {preprocessing_name}: {e}")
        raise
    
    return {
        'model': model,
        'preprocessing': preprocessing,
        'auc_score': auc_score
    }