import logging
from tqdm import tqdm
from data_loader import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from preprocessing import preprocess_data
from experiments import run_experiment, plot_results, plot_regression_metrics_barplots
from easy_models import sklearn_random_forest, sklearn_decision_tree, sklearn_knn
from custom_random_forest import custom_random_forest
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load datasets
logging.info("Loading Adult dataset...")
X_adult, y_adult = load_dataset(2)
logging.info("Loading Spambase dataset...")
X_spambase, y_spambase = load_dataset(94)

# Preprocess datasets
logging.info("Preprocessing Adult dataset...")
X_adult_scaled = preprocess_data(X_adult)
logging.info("Preprocessing Spambase dataset...")
X_spambase_scaled = preprocess_data(X_spambase)

# Split datasets into training and testing sets
logging.info("Splitting Adult dataset into training and testing sets...")
X_adult_train, X_adult_test, y_adult_train, y_adult_test = train_test_split(X_adult_scaled, y_adult, test_size=0.2)
logging.info("Splitting Spambase dataset into training and testing sets...")
X_spambase_train, X_spambase_test, y_spambase_train, y_spambase_test = train_test_split(X_spambase_scaled, y_spambase, test_size=0.2)

# Create output directory for plots
import os
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
logging.info("Running experiments with custom random forest...")
# Run custom random forest separately
import numpy as np

predictions1 = np.array(custom_random_forest(X_adult_train, y_adult_train, X_adult_test))
mse1 = mean_squared_error(y_adult_test, predictions1)
r2_1 = r2_score(y_adult_test, predictions1)
mae1 = mean_absolute_error(y_adult_test, predictions1)

predictions2 = np.array(custom_random_forest(X_spambase_train, y_spambase_train, X_spambase_test))
mse2 = mean_squared_error(y_spambase_test, predictions2)
r2_2 = r2_score(y_spambase_test, predictions2)
mae2 = mean_absolute_error(y_spambase_test, predictions2)

# Run easy models for comparison
logging.info("Running easy models for comparison...")
easy_models = [
    ("Random Forest", sklearn_random_forest),
    ("Decision Tree", sklearn_decision_tree),
    ("KNN", sklearn_knn)
]

results_adult = {}
results_spambase = {}
reports_adult = {}
reports_spambase = {}
for name, model_func in tqdm(easy_models, desc="Running easy models on Adult dataset"):
    predictions, mse, r2 = model_func(X_adult_train, y_adult_train, X_adult_test, y_adult_test)
    mae = mean_absolute_error(y_adult_test, predictions)
    results_adult[name] = {'mse': mse, 'r2': r2, 'mae': mae, 'predictions': predictions}
for name, model_func in tqdm(easy_models, desc="Running easy models on Spambase dataset"):
    predictions, mse, r2 = model_func(X_spambase_train, y_spambase_train, X_spambase_test, y_spambase_test)
    mae = mean_absolute_error(y_spambase_test, predictions)
    results_spambase[name] = {'mse': mse, 'r2': r2, 'mae': mae, 'predictions': predictions}

# Create output directory for plots
import os
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Plot classification report heatmaps
plot_regression_metrics_barplots({
    'Adult Dataset': {
        'Custom RF': {'mse': mse1, 'r2': r2_1, 'mae': mae1},
        'Sklearn RF': results_adult["Random Forest"],
        'Decision Tree': results_adult["Decision Tree"],
        'KNN': results_adult["KNN"]
    },
    'Spambase Dataset': {
        'Custom RF': {'mse': mse2, 'r2': r2_2, 'mae': mae2},
        'Sklearn RF': results_spambase["Random Forest"],
        'Decision Tree': results_spambase["Decision Tree"],
        'KNN': results_spambase["KNN"]
    }
}, output_dir)
logging.info("Plotting results...")
results = {
    'Adult Dataset': {
        'Custom RF': {'mse': mse1, 'r2': r2_1},
        'Sklearn RF': {'mse': results_adult["Random Forest"]['mse'], 'r2': results_adult["Random Forest"]['r2']},
        'Decision Tree': {'mse': results_adult["Decision Tree"]['mse'], 'r2': results_adult["Decision Tree"]['r2']},
        'KNN': {'mse': results_adult["KNN"]['mse'], 'r2': results_adult["KNN"]['r2']}
    },
    'Spambase Dataset': {
        'Custom RF': {'mse': mse2, 'r2': r2_2},
        'Sklearn RF': {'mse': results_spambase["Random Forest"]['mse'], 'r2': results_spambase["Random Forest"]['r2']},
        'Decision Tree': {'mse': results_spambase["Decision Tree"]['mse'], 'r2': results_spambase["Decision Tree"]['r2']},
        'KNN': {'mse': results_spambase["KNN"]['mse'], 'r2': results_spambase["KNN"]['r2']}
    }
}

plot_results(results, output_dir)
logging.info("Experiments completed!")
