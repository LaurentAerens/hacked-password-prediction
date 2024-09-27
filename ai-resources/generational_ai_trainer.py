import os
import random
import shutil
import json
from collections import defaultdict
from datetime import datetime

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

# Function to save hyperparameters and AUC scores to a file
def save_hyperparam_scores():
    os.makedirs('tuning', exist_ok=True)
    for model_preprocessing, scores in hyperparam_scores.items():
        file_path = f'tuning/{model_preprocessing}.json'
        with open(file_path, 'w') as f:
            json.dump(scores, f)

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


def train_model(X_train, X_test, y_train, y_test, model, preprocessing=None):
    """
    Train the given model on the provided data, optionally applying a single preprocessing step.
    
    Parameters:
    X_train, X_test, y_train, y_test: The training and testing data.
    model: The model to train.
    preprocessing: Optional single preprocessing step to apply.
    
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
    
    print(f"Training model {model.__class__.__name__} with preprocessing {preprocessing_name}...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC score for {model.__class__.__name__} with preprocessing {preprocessing_name}: {auc_score:.4f}")
    
    return {
        'model': model,
        'preprocessing': preprocessing,
        'auc_score': auc_score
    }

def select_models(models, survival_rate=0.3):
    """
    Select models based on their AUC score with a higher chance of survival for higher-ranked models.
    
    Parameters:
    models (list): A list of dictionaries containing models and their AUC scores.
    survival_rate (float): The rate of models to survive.
    
    Returns:
    list: A list of selected models.
    """
    models = sorted(models, key=lambda x: x['auc_score'], reverse=True)
    num_survivors = int(len(models) * survival_rate)
    selected_models = random.choices(models, k=num_survivors, weights=[model['auc_score'] for model in models])
    return selected_models

def update_hyperparam_scores(model_name, preprocessing_name, params, auc_score):
    """
    Update the hyperparameters and AUC scores for a given model/preprocessing combo.
    
    Parameters:
    model_name (str): The name of the model.
    preprocessing_name (str): The name of the preprocessing step.
    params (dict): The hyperparameters.
    auc_score (float): The AUC score.
    """
    key = json.dumps(params, sort_keys=True)
    combo_key = f"{model_name}_{preprocessing_name}"
    hyperparam_scores[combo_key][key] = auc_score


def get_best_params(model_name, preprocessing_name):
    """
    Get the best hyperparameters for a given model/preprocessing combo.
    
    Parameters:
    model_name (str): The name of the model.
    preprocessing_name (str): The name of the preprocessing step.
    
    Returns:
    dict: The best hyperparameters.
    """
    combo_key = f"{model_name}_{preprocessing_name}"
    if combo_key in hyperparam_scores:
        best_params = max(hyperparam_scores[combo_key], key=hyperparam_scores[combo_key].get)
        return json.loads(best_params)
    return None


def mutate_models(selected_models, param_distributions, total_models, n_iter=10, generation=1, max_generations=10):
    """
    Tune the hyperparameters of the selected models to create new versions.
    
    Parameters:
    selected_models (list): A list of selected models.
    param_distributions (dict): The hyperparameter distributions for each model.
    total_models (int): The total number of models to generate.
    n_iter (int): The number of iterations for hyperparameter tuning.
    generation (int): The current generation number.
    max_generations (int): The maximum number of generations.
    
    Returns:
    list: A list of new models with tuned hyperparameters.
    """
    new_models = []
    weights = [model['auc_score'] for model in selected_models]
    total_weights = sum(weights)
    normalized_weights = [weight / total_weights for weight in weights]
    
    # Adaptive mutation rate
    mutation_rate = max(0.1, 1 - (generation / max_generations))
    
    for model_info in selected_models:
        model_class = model_info['model'].__class__
        model_name = model_class.__name__
        preprocessing_name = model_info['preprocessing'] or 'None'
        param_dist = param_distributions[model_name]
        num_mutations = max(1, int(normalized_weights[selected_models.index(model_info)] * total_models))
        
        # Get the best hyperparameters found so far for this model/preprocessing combo
        best_params = get_best_params(model_name, preprocessing_name)
        
        param_sampler = ParameterSampler(param_dist, n_iter=num_mutations)
        for params in param_sampler:
            if best_params:
                # Determine which parameters to mutate
                params_to_mutate = select_params_to_mutate(best_params, generation, max_generations)
                
                # Mutate from the best hyperparameters
                for param in params_to_mutate:
                    if param in best_params:
                        params[param] = best_params[param] + random.uniform(-mutation_rate, mutation_rate) * best_params[param]
            
            new_model = model_class(**params)
            new_models.append({
                'model': new_model,
                'preprocessing': model_info['preprocessing']
            })
    
    # Ensure the total number of models remains the same
    while len(new_models) < total_models:
        model_info = random.choice(selected_models)
        model_class = model_info['model'].__class__
        model_name = model_class.__name__
        preprocessing_name = model_info['preprocessing'] or 'None'
        param_dist = param_distributions[model_name]
        params = next(ParameterSampler(param_dist, n_iter=1))
        
        best_params = get_best_params(model_name, preprocessing_name)
        if best_params:
            params_to_mutate = select_params_to_mutate(best_params, generation, max_generations, combo_key=f"{model_name}_{preprocessing_name}")
            for param in params_to_mutate:
                if param in best_params:
                    params[param] = best_params[param] + random.uniform(-mutation_rate, mutation_rate) * best_params[param]
        
        new_model = model_class(**params)
        new_models.append({
            'model': new_model,
            'preprocessing': model_info['preprocessing']
        })
    
    return new_models

def select_params_to_mutate(best_params, generation, max_generations, combo_key):
    """
    Select which parameters to mutate based on the generation and parameter stability.
    
    Parameters:
    best_params (dict): The best hyperparameters found so far.
    generation (int): The current generation number.
    max_generations (int): The maximum number of generations.
    
    Returns:
    list: A list of parameters to mutate.
    """
    params = list(best_params.keys())
    num_params = len(params)
    
    # Determine the number of parameters to mutate
    mutation_fraction = max(0.1, 1 - (generation / max_generations))
    num_params_to_mutate = max(1, int(mutation_fraction * num_params))
    
    # Select parameters to mutate
    params_to_mutate = random.sample(params, num_params_to_mutate)
    
    
    # Adjust the probability of mutation based on parameter stability
    for param in params:
        if param not in params_to_mutate:
            top_auc_scores = [hyperparam_scores[combo_key][key] for key in hyperparam_scores[combo_key] if param in json.loads(key)]
            if len(set(top_auc_scores)) == 1:
                if random.random() < 0.1:  # Lower the chance of mutation for stable parameters
                    params_to_mutate.append(param)
    
    return params_to_mutate


def log_best_model(model_info, experiment_dir, generation, previous_generation):
    """
    Log the best model information in a fun and informative way.
    
    Parameters:
    model_info (dict): A dictionary containing the model, preprocessing, and AUC score.
    experiment_dir (str): The directory to save the experiment results.
    generation (int): The current generation number.
    previous_generation (int): The generation number when the last best model was found.
    
    Returns:
    None
    """
    model = model_info['model']
    preprocessing = model_info['preprocessing']
    preprocessing_name = preprocessing.__class__.__name__ if preprocessing else 'None'
    model_name = model.__class__.__name__
    auc_score = model_info['auc_score']
    y_test = model_info['y_test']
    y_pred = model_info['y_pred']
    y_pred_proba = model_info['y_pred_proba']
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    log_file = os.path.join(experiment_dir, 'best-model', 'best_models_log.txt')
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*40}\n")
        f.write(f"Generation {generation}:\n")
        gen_diff = generation - previous_generation
        
        if gen_diff == 1:
            f.write(f"Immediately a new champion arises: {model_name}\n")
            options = [
                f"Before anyone could blink, {model_name} emerged victorious. The old champion barely had time to defend itself before being overthrown!\n",
                f"The battle was swift and decisive. {model_name} entered the arena, and with a single blow, the old champion crumbled. The new era begins now.\n",
                f"No sooner had the last champion settled into the throne than {model_name} swept in, claiming it effortlessly. A short reign, but a reign no more.\n"
            ]
            f.write(random.choice(options))
        
        elif 2 <= gen_diff <= 4:
            f.write(f"In a timely fashion a new champion has been found: {model_name}\n")
            options = [
                f"After a brief struggle, {model_name} demonstrated unparalleled skill, dethroning the former champion with elegance and precision. The throne has a new ruler!\n",
                f"Though it held power for a respectable time, the old champion could not withstand the rising tide. {model_name} has taken the title, leaving no doubt as to who reigns now.\n",
                f"With grace and power, {model_name} has unseated the previous ruler. It was a fair fight, but in the end, only one champion could prevail—and prevail it did.\n"
            ]
            f.write(random.choice(options))
        
        elif 5 <= gen_diff <= 8:
            f.write(f"Finally, a new champion is here: {model_name}\n")
            options = [
                f"The old champion seemed invincible, but fate had other plans. {model_name} has risen from the shadows, toppling the reign and claiming the title.\n",
                f"After a long-fought battle, the winds of change have arrived. {model_name} stands victorious, ending the era of the old champion. The arena buzzes with excitement!\n",
                f"It took time, but every reign must end. {model_name} has finally vanquished the reigning champion, bringing a well-deserved victory to the forefront.\n"
            ]
            f.write(random.choice(options))
        
        elif 9 <= gen_diff <= 12:
            f.write(f"After a long struggle we finally have a new champion: {model_name}\n")
            options = [
                f"After what seemed like an eternity of rule, the old champion has fallen. {model_name} had to claw through the ranks, but the long road has led to triumph!\n",
                f"The reign was ironclad, but nothing lasts forever. {model_name} has fought through countless challenges, seizing victory and shattering the old guard.\n",
                f"Many thought the old champion unbeatable, but {model_name} has proven them wrong. After a grueling journey, a new champion rises, stronger than ever.\n"
            ]
            f.write(random.choice(options))
        
        elif 13 <= gen_diff <= 20:
            f.write(f"Against all odds, a new champion emerges: {model_name}\n")
            options = [
                f"The old champion's strength was legendary, but legends are meant to be surpassed. Against impossible odds, {model_name} has done the unthinkable.\n",
                f"In a tale of perseverance and grit, {model_name} has defied every expectation. The old champion has been dethroned, and a new era begins.\n",
                f"With the odds stacked against it, {model_name} clawed its way to victory, proving that even the mightiest champions can fall. A new legend begins.\n"
            ]
            f.write(random.choice(options))
        
        else:
            f.write(f"After an epic journey, a new champion is crowned: {model_name}\n")
            options = [
                f"The journey was long and treacherous, but finally {model_name} stands tall. The old champion’s reign felt endless, but all things come to an end, and so too has this one.\n",
                f"Through countless battles, {model_name} has weathered every storm. After an epic odyssey, the old champion has been toppled, and a new ruler takes the throne.\n",
                f"Legends will speak of this day. {model_name}, after enduring what seemed like a never-ending saga, has emerged as the new champion. Hope is rekindled.\n"
            ]
            f.write(random.choice(options))
        
        f.write(f"AUC Score: {auc_score:.4f}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"Confusion Matrix:\n{conf_matrix}\n")
        f.write(f"{'='*40}\n")

def update_best_model(model_info, experiment_dir, highest_auc, generation, previous_generation):
    """
    Update the best model if the current model has a higher AUC score.
    
    Parameters:
    model_info (dict): A dictionary containing the model, preprocessing, and AUC score.
    experiment_dir (str): The directory to save the experiment results.
    highest_auc (float): The highest AUC score so far.
    generation (int): The current generation number.
    previous_generation (int): The generation number when the last best model was found.
    
    Returns:
    float: The updated highest AUC score.
    int: The updated generation number when the best model was found.
    """
    model = model_info['model']
    preprocessing = model_info['preprocessing']
    preprocessing_name = preprocessing.__class__.__name__ if preprocessing else 'None'
    model_name = model.__class__.__name__
    params = model.get_params()
    params_str = '-'.join([f'{key}={value}' for key, value in params.items()])
    auc_score = model_info['auc_score']
    date_str = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{model_name}-{preprocessing_name}-{params_str}-{date_str}-{auc_score:.4f}.joblib"
    best_model_dir = os.path.join(experiment_dir, 'best-model')
    generation_dir = os.path.join(best_model_dir, f'gen_{generation}')
    os.makedirs(generation_dir, exist_ok=True)
    
    if auc_score > highest_auc:
        highest_auc = auc_score
        # Remove the existing .joblib model in the root best-model directory
        for file in os.listdir(best_model_dir):
            if file.endswith(".joblib"):
                os.remove(os.path.join(best_model_dir, file))
        # Save the new best model in the root best-model directory
        shutil.copy(filepath, best_model_dir)
        print(f"New best model saved as {os.path.join(best_model_dir, filename)}")
        # Save the new best model in the generation folder
        shutil.copy(filepath, generation_dir)
        print(f"New best model also saved in generation folder as {os.path.join(generation_dir, filename)}")
        
        # Log the best model information
        log_best_model(model_info, experiment_dir, generation, previous_generation)
        previous_generation = generation
    
    return highest_auc, previous_generation

def evolutionary_training(X_train, X_test, y_train, y_test, experiment_dir, survival_rate=0.3, n_iter=10):
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
    
    # Initialize the first generation
    initial_models = []
    for model_class, param_dist in models.values():
        for preprocessing in list(preprocessing_options.values()) + [None]:
            param_sampler = ParameterSampler(param_dist, n_iter=1)
            for params in param_sampler:
                model = model_class(**params)
                initial_models.append({
                    'model': model,
                    'preprocessing': preprocessing
                })
    
    current_generation = initial_models
    best_models = []
    highest_auc = 0
    highest_avg_auc = 0
    no_improvement_generations = 0
    previous_generation = 0
    
    while no_improvement_generations < 3:
        generation = len(best_models) // total_models + 1
        print(f"Generation {generation}")
        generation_dir = os.path.join(experiment_dir, f'generation_{generation}')
        os.makedirs(generation_dir, exist_ok=True)
        
        evaluated_models = []
        for model_info in current_generation:
            model = model_info['model']
            preprocessing = model_info['preprocessing']
            pipeline = Pipeline([
                ('preprocessing', preprocessing),
                ('model', model)
            ]) if preprocessing else Pipeline([
                ('model', model)
            ])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_pred_proba)
            model_info.update({
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'auc_score': auc_score
            })
            evaluated_models.append(model_info)
        
        best_models = select_models(evaluated_models, survival_rate)
        
        for model_info in evaluated_models:
            model = model_info['model']
            preprocessing = model_info['preprocessing']
            preprocessing_name = preprocessing.__class__.__name__ if preprocessing else 'None'
            model_name = model.__class__.__name__
            params = model.get_params()
            params_str = '-'.join([f'{key}={value}' for key, value in params.items()])
            auc_score = model_info['auc_score']
            date_str = datetime.now().strftime("%Y%m%d%H%M%S")
            if preprocessing_name == 'None':
                filename = f"{model_name}-{params_str}-{date_str}-{auc_score:.4f}.joblib"
            else:
                filename = f"{model_name}-{preprocessing_name}-{params_str}-{date_str}-{auc_score:.4f}.joblib"
            filepath = os.path.join(generation_dir, filename)
            joblib.dump(model, filepath)
            print(f"Model saved as {filepath}")
            
            # Update the best model if this model has a higher AUC score
            highest_auc, previous_generation = update_best_model(model_info, experiment_dir, highest_auc, generation, previous_generation)
        
        current_generation = mutate_models(best_models, {model_class.__name__: param_dist for model_class, param_dist in models.values()}, n_iter, total_models)
        
        # Check for improvement
        avg_auc = sum(model['auc_score'] for model in best_models) / len(best_models)
        print(f"Average AUC for generation {generation}: {avg_auc:.4f}")
        if avg_auc > highest_avg_auc:
            highest_avg_auc = avg_auc
            no_improvement_generations = 0
        else:
            no_improvement_generations += 1

def main_loop(experiment_name, data_file_path='data/combined_data.csv'):
    # step 0: create models directory
    if not os.path.exists('models'):
        os.makedirs('models')

    # Step 1: Get the data
    data = get_data(data_file_path)

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

    # Step 4: Create an experiment directory
    date_str = datetime.now().strftime("%Y%m%d%H%M%S")
    experiment_dir = f"models/experiment-{experiment_name}-{date_str}"
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'best-model'), exist_ok=True)

    # Step 5: Perform evolutionary training
    evolutionary_training(X_train, X_test, y_train, y_test, experiment_dir)
def main():
    experiment_name = input("Enter the experiment name: ")
    main_loop(experiment_name)
if __name__ == "__main__":
    main()