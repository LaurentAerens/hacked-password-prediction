import os
import re
import pandas as pd
import json

# Load configuration from config.json
config_path = "../ai-resources/config.json"
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

azure_url = config.get('azure_url')
api_key = config.get('api_key')
azure_model_auc = config.get('azure_model_auc')

def ListALLModels():
    """
    List all .joblib files in the ../ai-resources/models directory,
    excluding vectorizer.joblib, X.joblib, and y.joblib, and sort them by AUC score.
    
    Returns:
    list: A list of .joblib files sorted by AUC score.
    """
    path = "../ai-resources/models"
    exclude_files = {'vectorizer.joblib', 'X.joblib', 'y.joblib'}
    joblib_files = [f for f in os.listdir(path) if f.endswith('.joblib') and f not in exclude_files]
    
    # Regular expression to match the filenames and extract the AUC score
    regex = re.compile(r'-(\d+\.\d+)\.joblib$')
    
    # Parse filenames to extract AUC scores
    parsed_files = []
    for file in joblib_files:
        match = regex.search(file)
        if match:
            auc_score = float(match.group(1))
            parsed_files.append((file, auc_score))
    
    # Include Azure model if it exists and has valid URL and API key
    if azure_model_auc is not None and azure_url and api_key:
        parsed_files.append(('azure', azure_model_auc))
    
    # Sort files by AUC score
    sorted_files = sorted(parsed_files, key=lambda x: x[1], reverse=True)
    
    # Return sorted filenames
    return [file for file, _ in sorted_files]


def ListFastModels():
    """
    List all .joblib files in the ../ai-resources/models directory that start with 'fast-',
    and sort them by AUC score.
    
    Returns:
    list: A list of .joblib files sorted by AUC score.
    """
    path = "../ai-resources/models"
    joblib_files = [f for f in os.listdir(path) if f.startswith('fast-') and f.endswith('.joblib')]

    # Regular expression to match the filenames and extract the AUC score
    regex = re.compile(r'-(\d+\.\d+)\.joblib$')
    
    # Parse filenames to extract AUC scores
    parsed_files = []
    for file in joblib_files:
        match = regex.search(file)
        if match:
            auc_score = float(match.group(1))
            parsed_files.append((file, auc_score))
    
    # Sort files by AUC score
    sorted_files = sorted(parsed_files, key=lambda x: x[1], reverse=True)
    
    # Return sorted filenames
    return [file for file, _ in sorted_files]

def GetBestModelName():
    """
    Get the filename of the best model in the ../ai-resources/models directory.
    
    Returns:
    str: The filename of the best model.
    """
    path = "../ai-resources/models"
    exclude_files = {'vectorizer.joblib', 'X.joblib', 'y.joblib'}
    joblib_files = [f for f in os.listdir(path) if f.endswith('.joblib') and f not in exclude_files]
    
    # Regular expression to match the filenames and extract the AUC score
    regex = re.compile(r'-(\d+\.\d+)\.joblib$')
    
    # Parse filenames to extract AUC scores
    parsed_files = []
    for file in joblib_files:
        match = regex.search(file)
        if match:
            auc_score = float(match.group(1))
            parsed_files.append((file, auc_score))
    
    # Include Azure model if it exists and has valid URL and API key
    if azure_model_auc is not None and azure_url and api_key:
        parsed_files.append(('azure', azure_model_auc))
    
    # Sort files by AUC score
    sorted_files = sorted(parsed_files, key=lambda x: x[1], reverse=True)
    
    # Return the filename of the best model
    return sorted_files[0][0]


def ListXbestModels(amount):
    """
    List the X best models in the ../ai-resources/models directory.

    Args:
        amount (int): The number of best models to list.

    Returns:
    list: A list of the X best models.
    """
    # Error handling for the amount parameter
    if not isinstance(amount, int) or amount <= 0:
        raise ValueError("The amount parameter must be a whole positive number.")

    path = "../ai-resources/models"
    joblib_files = [f for f in os.listdir(path) if f.endswith('.joblib')]

    # Regular expression to match the filenames and extract the AUC score
    regex = re.compile(r'-(\d+\.\d+)\.joblib$')

    # Parse filenames to extract AUC scores
    parsed_files = []
    for file in joblib_files:
        match = regex.search(file)
        if match:
            auc_score = float(match.group(1))
            parsed_files.append((file, auc_score))

    # Include Azure model if it exists and has valid URL and API key
    if azure_model_auc is not None and azure_url and api_key:
        parsed_files.append(('azure', azure_model_auc))

    # Sort files by AUC score
    sorted_files = sorted(parsed_files, key=lambda x: x[1], reverse=True)

    # Return the filenames of the best models
    return [file for file, _ in sorted_files[:amount]] if amount <= len(sorted_files) else [file for file, _ in sorted_files]

def getTrainingDataFileNames():
    """
    Get the training data from the ../ai-resources/data directory.
    
    Returns:
    list: A list of training data filenames.
    """
    path = "../ai-resources/data"
    exclude_files = {'events.csv'}
    data_files = [f for f in os.listdir(path) if f.endswith('.csv') and f not in exclude_files]
    return data_files

def getTrainingData(Filename):
    """
    Get the training data from the given filename in the ../ai-resources/data directory.

    Args:
        Filename (str): The name of the file to load.

    Returns:
    pandas.DataFrame: The training data.

    Raises:
    FileNotFoundError: If the file does not exist, raises an error containing all available filenames.
    """
    path = f"../ai-resources/data/{Filename}"
    if not os.path.exists(path):
        available_files = getTrainingDataFileNames()
        raise FileNotFoundError(f"The file '{Filename}' does not exist. Available files: {available_files}")
    return pd.read_csv(path)


def AddCrackedPassword(password, Filename):
    """
    Add a cracked password to the training data file.

    Args:
        password (str): The cracked password to add.
        Filename (str): The name of the file to load.
        
    Raises:
    ValueError: If the password is not a string or the filename is not a .csv file.
    FileNotFoundError: If the file does not exist, raises an error containing all available filenames.
    """
    # validate that password is a string
    if not isinstance(password, str):
        raise ValueError("Password must be a string.")
    # validate that filename is a csv file
    if not Filename.endswith('.csv'):
        raise ValueError("Filename must be a .csv file.")
    path = f"../ai-resources/data/{Filename}"
    if not os.path.exists(path):
        available_files = getTrainingDataFileNames()
        raise FileNotFoundError(f"The file '{Filename}' does not exist. Available files: {available_files}")
    trainingdata = pd.read_csv(path)
    # Validate that it has the correct columns
    if 'password' not in trainingdata.columns or 'target' not in trainingdata.columns:
        raise ValueError("The training data file must contain 'password' and 'target' columns.")
    trainingdata = trainingdata.append({'password': password, 'target': 1}, ignore_index=True)
    trainingdata.to_csv(path, index=False)

def AddCrackedPasswords(passwords, Filename):
    """
    Add multiple cracked passwords to the training data file.

    Args:
        passwords (list): A list of cracked passwords to add.
        Filename (str): The name of the file to load.
        
    Raises:
    ValueError: If the passwords are not strings or the filename is not a .csv file.
    FileNotFoundError: If the file does not exist, raises an error containing all available filenames.
    """
    # validate that passwords is a list of strings
    if not all(isinstance(password, str) for password in passwords):
        raise ValueError("Passwords must be a list of strings.")
    # validate that filename is a csv file
    if not Filename.endswith('.csv'):
        raise ValueError("Filename must be a .csv file.")
    path = f"../ai-resources/data/{Filename}"
    if not os.path.exists(path):
        available_files = getTrainingDataFileNames()
        raise FileNotFoundError(f"The file '{Filename}' does not exist. Available files: {available_files}")
    trainingdata = pd.read_csv(path)
    # Validate that it has the correct columns
    if 'password' not in trainingdata.columns or 'target' not in trainingdata.columns:
        raise ValueError("The training data file must contain 'password' and 'target' columns.")
    new_data = [{'password': password, 'target': 1} for password in passwords]
    trainingdata = trainingdata.append(new_data, ignore_index=True)
    trainingdata.to_csv(path, index=False)

def CleanALLButBestModel():
    """
    Clean all models in the ../ai-resources/models directory except for the best model.
    """
    path = "../ai-resources/models"
    exclude_files = {GetBestModelName(), 'vectorizer.joblib', 'X.joblib', 'y.joblib'}
    joblib_files = [f for f in os.listdir(path) if f.endswith('.joblib') and f not in exclude_files]
    
    for file in joblib_files:
        os.remove(os.path.join(path, file))
    
# Example usage
if __name__ == "__main__":
    getTrainingData("passwords.csv")