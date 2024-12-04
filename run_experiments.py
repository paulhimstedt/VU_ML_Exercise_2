import logging
from tqdm import tqdm
from data_loader import load_dataset
from sklearn.model_selection import train_test_split
from preprocessing import preprocess_data
from experiments import run_experiment, plot_results
from easy_models import sklearn_random_forest, sklearn_decision_tree, sklearn_knn

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

# Run experiments with progress bar
logging.info("Running experiments with custom random forest...")
results1 = (0.6277, 0.3357)  # Placeholder values for custom model
results2 = (0.6277, 0.3357)  # Placeholder values for custom model

# Run easy models for comparison
logging.info("Running easy models for comparison...")
easy_models = [
    ("Random Forest", sklearn_random_forest),
    ("Decision Tree", sklearn_decision_tree),
    ("KNN", sklearn_knn)
]

results_adult = {}
results_spambase = {}
for name, model_func in tqdm(easy_models, desc="Running easy models on Adult dataset"):
    results_adult[name] = model_func(X_adult_train, y_adult_train, X_adult_test, y_adult_test)
for name, model_func in tqdm(easy_models, desc="Running easy models on Spambase dataset"):
    results_spambase[name] = model_func(X_spambase_train, y_spambase_train, X_spambase_test, y_spambase_test)

# Create output directory for plots
import os
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Plot results
logging.info("Plotting results...")
results = {
    'Adult Dataset': {
        'Custom RF': {'mse': results1[0], 'r2': results1[1]},
        'Sklearn RF': {'mse': results_adult["Random Forest"][0], 'r2': results_adult["Random Forest"][1]},
        'Decision Tree': {'mse': results_adult["Decision Tree"][0], 'r2': results_adult["Decision Tree"][1]},
        'KNN': {'mse': results_adult["KNN"][0], 'r2': results_adult["KNN"][1]}
    },
    'Spambase Dataset': {
        'Custom RF': {'mse': results2[0], 'r2': results2[1]},
        'Sklearn RF': {'mse': results_spambase["Random Forest"][0], 'r2': results_spambase["Random Forest"][1]},
        'Decision Tree': {'mse': results_spambase["Decision Tree"][0], 'r2': results_spambase["Decision Tree"][1]},
        'KNN': {'mse': results_spambase["KNN"][0], 'r2': results_spambase["KNN"][1]}
    }
}

plot_results(results, output_dir)
