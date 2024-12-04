from data_loader import load_dataset
from preprocessing import preprocess_data
from experiments import run_experiment, plot_results
from easy_models import sklearn_random_forest, sklearn_decision_tree, sklearn_knn

# Load datasets
X1, y1 = load_dataset(2)
X2, y2 = load_dataset(94)

# Preprocess datasets
X1_scaled = preprocess_data(X1)
X2_scaled = preprocess_data(X2)

# Split datasets into training and testing sets
from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(X1_scaled, y1, test_size=0.2)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2_scaled, y2, test_size=0.2)

# Run experiments
results1 = run_experiment(X1_train, y1_train, X1_test, y1_test)
results2 = run_experiment(X2_train, y2_train, X2_test, y2_test)

# Run easy models for comparison
results_rf = sklearn_random_forest(X1_train, y1_train, X1_test, y1_test)
results_dt = sklearn_decision_tree(X1_train, y1_train, X1_test, y1_test)
results_knn = sklearn_knn(X1_train, y1_train, X1_test, y1_test)

# Plot results
print("Custom Random Forest:", results1)
print("Sklearn Random Forest:", results_rf)
print("Sklearn Decision Tree:", results_dt)
print("Sklearn KNN:", results_knn)
plot_results({'mse': [results1[0], results2[0]], 'r2': [results1[1], results2[1]]})
