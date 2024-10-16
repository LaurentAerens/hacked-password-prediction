import os
import re
import json
import ssl
import joblib
import warnings
import urllib.request
from typing import List, Dict, Any, Union

# Define the models directory
models_dir = 'models'

# Load the vectorizer
vectorizer_file = os.path.join(models_dir, 'vectorizer.joblib')
if os.path.exists(vectorizer_file):
    print(f"Loading vectorizer from {vectorizer_file}...")
    vectorizer = joblib.load(vectorizer_file)
else:
    print(f"Vectorizer file {vectorizer_file} not found. Exiting.")
    exit(1)

# Load configuration from config.json
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

azure_url = config.get('azure_url')
api_key = config.get('api_key')
azure_model_auc = config.get('azure_model_auc')

def allowSelfSignedHttps(allowed: bool) -> None:
    """
    Bypass the server certificate verification on client side.
    """
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True)  # This line is needed if you use self-signed certificate in your scoring service.

def check_password_azure(password: str) -> Union[List[bool], None]:
    """
    Send a string to the Azure model and get the response.
    """
    data = {
        "input_data": {
            "data": [
                {"password": password}
            ]
        }
    }

    body = str.encode(json.dumps(data))

    if not azure_url or not api_key:
        raise Exception("Azure URL and API key must be provided in the config file")
        
    headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + api_key}

    req = urllib.request.Request(azure_url, body, headers)
    
    try:
        response = urllib.request.urlopen(req)
        result = response.read()
        return json.loads(result)
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))
        print(error.info())
        print(json.loads(error.read().decode("utf8", 'ignore')))
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def extract_model_details(filename: str, path: str, model_type: str) -> Union[Dict[str, Any], None]:
    """
    Extract model details from filename.
    """
    fast_model_pattern = r"fast-(\w+)-(\w+)-\d{14}-(\d+\.\d+)\.joblib"
    fast_model_no_preprocessing_pattern = r"fast-(\w+)-\d{14}-(\d+\.\d+)\.joblib"
    gen_model_pattern = r"(\w+)-(\w+)-.*-(\d+\.\d+)\.joblib"
    gen_model_no_preprocessing_pattern = r"(\w+)-\d{14}-(\d+\.\d+)\.joblib"
    
    if model_type == 'fast':
        match = re.match(fast_model_pattern, filename)
        if not match:
            match = re.match(fast_model_no_preprocessing_pattern, filename)
    else:
        match = re.match(gen_model_pattern, filename)
        if not match:
            match = re.match(gen_model_no_preprocessing_pattern, filename)
    
    if match:
        return {
            'model_name': match.group(1),
            'preprocessing_name': match.group(2) if len(match.groups()) > 2 else None,
            'auc_score': float(match.group(len(match.groups()))),
            'path': os.path.join(path, filename),
            'type': model_type
        }
    return None

def select_models() -> (Dict[int, Any], List[Dict[str, Any]]):
    """
    Load and select models.
    """
    # Load all fast models
    fast_model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib') and f.startswith('fast-')]
    models = []

    for model_file in fast_model_files:
        model_details = extract_model_details(model_file, models_dir, 'fast')
        if model_details:
            models.append(model_details)

    # Load all generational models from subdirectories
    for subdir in os.listdir(models_dir):
        subdir_path = os.path.join(models_dir, subdir)
        if os.path.isdir(subdir_path):
            gen_model_files = [f for f in os.listdir(subdir_path) if f.endswith('.joblib')]
            for model_file in gen_model_files:
                model_details = extract_model_details(model_file, subdir_path, 'gen')
                if model_details:
                    models.append(model_details)

    # Add Azure model to the list
    if azure_url and api_key and azure_model_auc:
        models.append({
            'model_name': 'Azure Model',
            'preprocessing_name': None,
            'auc_score': azure_model_auc,
            'path': None,
            'type': 'azure'
        })

    # Sort models by AUC score, keeping fast models on top
    models.sort(key=lambda x: (-x['auc_score'], x['type']))

    # Display models and let the user choose which ones to load
    print("Available models:")
    for idx, model in enumerate(models, start=1):
        preprocessing_name = model['preprocessing_name'] if model['preprocessing_name'] else 'None'
        print(f"{idx}. {model['model_name']} (Preprocessing: {preprocessing_name}, AUC: {model['auc_score']:.4f})")

    selected_indices = input("Enter the numbers of the models you want to load, separated by commas: ")
    selected_indices = [int(i.strip()) for i in selected_indices.split(',')]

    # Load the selected models
    selected_models = {}
    for idx in selected_indices:
        if models[idx - 1]['type'] == 'azure':
            selected_models[idx] = 'azure'
        else:
            selected_models[idx] = joblib.load(models[idx - 1]['path'])

    print("Selected models loaded successfully.")
    return selected_models, models

def use_selected_models(selected_models: Dict[int, Any], model_details: List[Dict[str, Any]]) -> None:
    """
    Use selected models on user input.
    """
    while True:
        user_input = input("Enter your password to predict (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break

        # Vectorize the input
        vectorized_input = vectorizer.transform([user_input])

        # Run the input through each selected model and print the results
        for idx, model in selected_models.items():
            if model == 'azure':
                result = check_password_azure(user_input)
                if result is not None and isinstance(result, list) and len(result) > 0:
                    is_hacked = result[0]
                    if isinstance(is_hacked, bool):
                        if is_hacked:
                            print(f"Azure Model (AUC: {azure_model_auc}, Preprocessing: Unknown) predicted that '{user_input}' is a HACKED password.")
                        else:
                            print(f"Azure Model (AUC: {azure_model_auc}, Preprocessing: Unknown) predicted that '{user_input}' is NOT a hacked password.")
                    else:
                        print("Unexpected response format from Azure model.")
                else:
                    print("Failed to get a valid response from the Azure model.")
            else:
                prediction = model.predict(vectorized_input)[0]
                details = model_details[idx - 1]
                preprocessing_name = details['preprocessing_name'] if details['preprocessing_name'] else 'None'
                if prediction == 0:
                    print(f"Model {details['model_name']} (AUC: {details['auc_score']:.4f}, Preprocessing: {preprocessing_name}) predicted that '{user_input}' is NOT a hacked password.")
                else:
                    print(f"Model {details['model_name']} (AUC: {details['auc_score']:.4f}, Preprocessing: {preprocessing_name}) predicted that '{user_input}' is a HACKED password.")


def get_all_model_paths(models_dir: str) -> Dict[str, str]:
    """
    Traverse the models directory and its subdirectories to create a dictionary
    mapping model filenames to their paths.
    """
    model_paths = {}
    for root, _, files in os.walk(models_dir):
        for file in files:
            if file.endswith('.joblib'):
                model_paths[file] = os.path.join(root, file)
    return model_paths

def load_models(models_filename: List[str]) -> List[Dict[str, Any]]:
    """
    Find the models corresponding to the names in models_filename and load them into the models list.
    Not finding a model is a warning, finding no models is an error.
    Next to these models, azure is also an option in the list if azure_url, api_key, and azure_model_auc are set in the config file.
    """
    model_paths = get_all_model_paths(models_dir)
    models = []

    for filename in models_filename:
        if filename == 'azure':
            if azure_url and api_key and azure_model_auc:
                models.append({'model_name': 'Azure Model', 'type': 'azure'})
            else:
                warnings.warn("Azure model specified but azure_url, api_key, or azure_model_auc is missing in the config file.")
        else:
            if filename in model_paths:
                model_path = model_paths[filename]
                models.append({'model_name': filename, 'model': joblib.load(model_path), 'type': 'local'})
            else:
                warnings.warn(f"Model file {filename} not found.")

    if not models:
        raise ValueError("No valid models found. If you only wanted to use the Azure model, just call that API directly on url: {azure_url}")

    return models

def use_Models_on_Passwords(models_filename: List[str], passwords: List[str]) -> Dict[str, Any]:
    """
    Use selected models on the list of passwords.

    Args:
        models_filename (list): List of model names.
        passwords (list): The list of passwords to predict.

    Returns:
    json: A JSON object containing every password and the predictions of each model.
    """
    if not isinstance(passwords, list) or not all(isinstance(p, str) for p in passwords):
        return {"error": "Passwords must be a list of strings."}

    

    vectorizer_file = os.path.join(models_dir, 'vectorizer.joblib')
    if not os.path.exists(vectorizer_file):
        return {"error": "Vectorizer file not found."}
    
    vectorizer = joblib.load(vectorizer_file)
    json_output = {}

    for password in passwords:
        vectorized_input = vectorizer.transform([password])
        predictions = []

        if use_azure:
            result = check_password_azure(password)
            if result is not None and isinstance(result, list) and len(result) > 0:
                is_hacked = result[0]
                if isinstance(is_hacked, bool):
                    predictions.append({"Azure": is_hacked})
                else:
                    raise ValueError("Unexpected response format from Azure model.")
            else:
                raise ValueError("Failed to get a valid response from the Azure model.")

        for model in models:
            prediction = model['model'].predict(vectorized_input)[0]
            predictions.append({model['model_name']: prediction})

        json_output[password] = predictions

    return json_output

# Main function
if __name__ == "__main__":
    selected_models, model_details = select_models()
    use_selected_models(selected_models, model_details)