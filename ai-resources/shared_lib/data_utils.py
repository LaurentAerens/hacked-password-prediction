import os
import pandas as pd
from sklearn.model_selection import train_test_split

def get_data(file_path='data/combined_data.csv', logger=None):
    """
    Load data from a CSV file into a pandas DataFrame and remove all null values.
    
    Parameters:
    file_path (str): The path to the CSV file.
    logger (LoggerWrapper): The logger instance to use for logging.
    
    Returns:
    pd.DataFrame: The cleaned DataFrame.
    
    Raises:
    FileNotFoundError: If the file does not exist.
    ValueError: If the file is not in CSV format or if the data is empty after removing null values.
    """
    if not os.path.exists(file_path):
        if logger:
            logger.error(f"The file {file_path} does not exist.")
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    if not file_path.endswith('.csv'):
        if logger:
            logger.error(f"The file {file_path} is not in CSV format.")
        raise ValueError(f"The file {file_path} is not in CSV format.")
    
    if logger:
        logger.info(f"Loading data from {file_path}...")
    
    data = pd.read_csv(file_path)
    data.dropna(inplace=True)
    
    if data.empty:
        if logger:
            logger.error("The data is empty after removing null values.")
        raise ValueError("The data is empty after removing null values.")
    
    if logger:
        logger.info("Data loaded and cleaned successfully.")
    
    return data

def split_data(X, y, logger=None):
    """
    Split the data into training and testing sets.
    
    Parameters:
    X (sparse matrix): The feature data.
    y (pd.Series): The target variable.
    logger (LoggerWrapper): The logger instance to use for logging.
    
    Returns:
    X_train, X_test, y_train, y_test: The split data.
    """
    if logger:
        logger.info("Splitting data into training and testing sets...")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if logger:
        logger.info("Data split successfully.")
    
    return X_train, X_test, y_train, y_test