import joblib
import os
import re
import getpass

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

# Function to extract model details from filename
def extract_model_details(filename, path, model_type):
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

# Function to load and select models
def select_models():
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
    selected_models = {idx: joblib.load(models[idx - 1]['path']) for idx in selected_indices}

    print("Selected models loaded successfully.")
    return selected_models, models

# Function to use selected models on user input
def use_models(selected_models, model_details):
    while True:
        user_input = input("Enter your password to predict (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break

        # Vectorize the input
        vectorized_input = vectorizer.transform([user_input])

        # Run the input through each selected model and print the results
        for idx, model in selected_models.items():
            prediction = model.predict(vectorized_input)[0]
            details = model_details[idx - 1]
            preprocessing_name = details['preprocessing_name'] if details['preprocessing_name'] else 'None'
            if prediction == 0:
                print(f"Model {details['model_name']} (AUC: {details['auc_score']:.4f}, Preprocessing: {preprocessing_name}) predicted that '{user_input}' is NOT a hacked password.")
            else:
                print(f"Model {details['model_name']} (AUC: {details['auc_score']:.4f}, Preprocessing: {preprocessing_name}) predicted that '{user_input}' is a HACKED password.")

# Main function
if __name__ == "__main__":
    selected_models, model_details = select_models()
    use_models(selected_models, model_details)