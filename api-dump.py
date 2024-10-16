import random
import string
from flask import Flask, request, jsonify
import subprocess
import fast_ai_trainer
import generational_ai_trainer

app = Flask(__name__)

@app.route('/generate-data', methods=['POST'])
def generate_data():
    """
    Endpoint to generate data.
    
    Example JSON data:
    {
        "file_path": "data/events.csv", # Optional
        "output_file_path": "data/combined_data.csv", # Optional
        "generated_data_factor": 1 # Optional
    }
    """
    try:
        data = request.json
        file_path = data.get('file_path', 'data/events.csv')
        output_file_path = data.get('output_file_path', 'data/combined_data.csv')
        generated_data_factor = data.get('generated_data_factor', 1)
        
        subprocess.run(["python", "ai-resources/data_generation.py", file_path, output_file_path, str(generated_data_factor)], check=True)
        
        return jsonify({"message": "Data generation completed."})
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Subprocess error: {e}"}), 500
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500

@app.route('/train-single-model', methods=['POST'])
def train_single_model():
    """
    Endpoint to train a single model with a single preprocessing option.
    
    Example JSON data:
    {
        "model_name": "LogisticRegression",
        "preprocessing_name": "StandardScaler"
        "data_file_path": "data/combined_data.csv" # Optional
    }
    """
    try:
        data = request.json
        model_name = data['model_name']
        preprocessing_name = data.get('preprocessing_name', None)
        data_file_path = data.get('data_file_path', 'data/combined_data.csv')
        
        warnings = fast_ai_trainer.train_single_model_with_preprocessing(model_name, preprocessing_name,data_file_path)
        
        return jsonify({"message": f"Training completed for model {model_name} with preprocessing {preprocessing_name}.", "warnings": warnings})
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500

@app.route('/train-list-of-models', methods=['POST'])
def train_list_of_models():
    """
    Endpoint to train a list of models with their respective preprocessing options.
    
    Example JSON data:
    {
        "model_preprocessing_list": [
            ["LogisticRegression", "StandardScaler"],
            ["RandomForest", "MaxAbsScaler"]
        ]
        "data_file_path": "data/combined_data.csv" # Optional
    }
    """
    try:
        data = request.json
        model_preprocessing_list = data['model_preprocessing_list']
        data_file_path = data.get('data_file_path', 'data/combined_data.csv')
        
        warnings = fast_ai_trainer.train_list_of_models_with_preprocessing(model_preprocessing_list, data_file_path)
        
        return jsonify({"message": "Training completed for the provided list of models and preprocessing options.", "warnings": warnings})
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500

@app.route('/train-all-models', methods=['POST'])
def train_all_models():
    """
    Endpoint to train all models with all preprocessing options and without preprocessing.
    
    Example JSON data:
    {
        "data_file_path": "data/combined_data.csv" # Optional
    }
    """
    try:
        data = request.json
        data_file_path = data.get('data_file_path', 'data/combined_data.csv')
        warnings = fast_ai_trainer.train_all_models_with_preprocessing(data_file_path)
        
        return jsonify({"message": "Training completed for all models with all preprocessing options.", "warnings": warnings})
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500
    
@app.route('/evolutionary-training', methods=['POST'])
def evolutionary_training():
    """
    Endpoint to perform evolutionary training.
    
    Example JSON data:
    {
        "experiment_name": "experiment1" # Optional
        "data_file_path": "data/combined_data.csv" # Optional
    }
    """
    try:
        data = request.json
        experiment_name = data.get('experiment_name', ''.join(random.choices(string.ascii_letters + string.digits, k=20)))
        data_file_path = data.get('data_file_path', 'data/combined_data.csv')
        
        generational_ai_trainer.main_loop(experiment_name, data_file_path)
        
        return jsonify({"message": "Evolutionary training completed."})
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)