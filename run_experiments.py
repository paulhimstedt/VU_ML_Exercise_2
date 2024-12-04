import logging
from tqdm import tqdm
from data_loader import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from preprocessing import preprocess_data
from experiments import run_experiment, plot_results, plot_classification_report_heatmaps
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
# Convert continuous predictions to binary using a threshold
binary_predictions1 = (predictions1 > 0.5).astype(int)
plot_classification_report_heatmap(y_adult_test, binary_predictions1, "Custom RF Adult Dataset", output_dir, "PuRd")

predictions2 = np.array(custom_random_forest(X_spambase_train, y_spambase_train, X_spambase_test))
mse2 = mean_squared_error(y_spambase_test, predictions2)
r2_2 = r2_score(y_spambase_test, predictions2)
mae2 = mean_absolute_error(y_spambase_test, predictions2)
# Convert continuous predictions to binary using a threshold
binary_predictions2 = (predictions2 > 0.5).astype(int)
plot_classification_report_heatmap(y_spambase_test, binary_predictions2, "Custom RF Spambase Dataset", output_dir, "PuRd")

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
    predictions, _, _ = model_func(X_adult_train, y_adult_train, X_adult_test, y_adult_test)
    mse = mean_squared_error(y_adult_test, predictions)
    r2 = r2_score(y_adult_test, predictions)
    mae = mean_absolute_error(y_adult_test, predictions)
    results_adult[name] = {'mse': mse, 'r2': r2, 'mae': mae}
    plot_classification_report_heatmap(y_adult_test, predictions, f"{name} Adult Dataset", output_dir, "Greens")
for name, model_func in tqdm(easy_models, desc="Running easy models on Spambase dataset"):
    predictions, _, _ = model_func(X_spambase_train, y_spambase_train, X_spambase_test, y_spambase_test)
    mse = mean_squared_error(y_spambase_test, predictions)
    r2 = r2_score(y_spambase_test, predictions)
    mae = mean_absolute_error(y_spambase_test, predictions)
    results_spambase[name] = {'mse': mse, 'r2': r2, 'mae': mae}
    plot_classification_report_heatmap(y_spambase_test, predictions, f"{name} Spambase Dataset", output_dir, "Greens")

# Create output directory for plots
import os
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Plot classification report heatmaps
plot_classification_report_heatmaps({
    'Adult Dataset': {
        'Custom RF': {'y_true': y_adult_test, 'y_pred': binary_predictions1},
        'Sklearn RF': {'y_true': y_adult_test, 'y_pred': results_adult["Random Forest"]['predictions']},
        'Decision Tree': {'y_true': y_adult_test, 'y_pred': results_adult["Decision Tree"]['predictions']},
        'KNN': {'y_true': y_adult_test, 'y_pred': results_adult["KNN"]['predictions']}
    },
    'Spambase Dataset': {
        'Custom RF': {'y_true': y_spambase_test, 'y_pred': binary_predictions2},
        'Sklearn RF': {'y_true': y_spambase_test, 'y_pred': results_spambase["Random Forest"]['predictions']},
        'Decision Tree': {'y_true': y_spambase_test, 'y_pred': results_spambase["Decision Tree"]['predictions']},
        'KNN': {'y_true': y_spambase_test, 'y_pred': results_spambase["KNN"]['predictions']}
    }
}, output_dir)
logging.info("Plotting results...")
results = {
    'Adult Dataset': {
        'Custom RF': {'mse': mse1, 'r2': r2_1},
        'Sklearn RF': {'mse': results_adult["Random Forest"][0], 'r2': results_adult["Random Forest"][1]},
        'Decision Tree': {'mse': results_adult["Decision Tree"][0], 'r2': results_adult["Decision Tree"][1]},
        'KNN': {'mse': results_adult["KNN"][0], 'r2': results_adult["KNN"][1]}
    },
    'Spambase Dataset': {
        'Custom RF': {'mse': mse2, 'r2': r2_2},
        'Sklearn RF': {'mse': results_spambase["Random Forest"][0], 'r2': results_spambase["Random Forest"][1]},
        'Decision Tree': {'mse': results_spambase["Decision Tree"][0], 'r2': results_spambase["Decision Tree"][1]},
        'KNN': {'mse': results_spambase["KNN"][0], 'r2': results_spambase["KNN"][1]}
    }
}

plot_results(results, output_dir)
logging.info("Experiments completed!")
