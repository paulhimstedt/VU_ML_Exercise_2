import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

def run_experiment(model, X_train, y_train, X_test, y_test, is_custom=False):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, r2

def plot_regression_metrics_heatmap(results, output_dir):
    fig, axs = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle("Regression Metrics Heatmaps")

    models = ['Custom RF', 'Sklearn RF', 'Decision Tree', 'KNN']
    datasets = ['Adult Dataset', 'Spambase Dataset']
    metrics = ['MSE', 'R2', 'MAE']

    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            scores = [results[dataset][model][metric.lower()] for metric in metrics]
            sns.heatmap([scores], annot=True, cmap='coolwarm', ax=axs[i, j], cbar=False, xticklabels=metrics)
            axs[i, j].set_title(f"{model} on {dataset}")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{output_dir}/regression_metrics_heatmaps.png")
    plt.close()

def plot_results(results, output_dir):
    sns.set_palette("husl")
    fig, axs = plt.subplots(2, 4, figsize=(20, 15))
    fig.suptitle("Model Performance Comparison")

    models = ['Custom RF', 'Sklearn RF', 'Decision Tree', 'KNN']
    datasets = ['Adult Dataset', 'Spambase Dataset']

    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            metrics = ['MSE', 'R2']
            scores = [results[dataset][model]['mse'], results[dataset][model]['r2']]
            axs[i, j].bar(metrics, scores, color=['blue', 'orange', 'green'])
            axs[i, j].set_ylim(0, max(scores) + 0.1)
            axs[i, j].set_title(f"{model} on {dataset}")
            axs[i, j].set_xlabel('Metric')
            axs[i, j].set_ylabel('Score')


    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    plt.savefig(f"{output_dir}/model_performance_comparison.png")
    plt.close()
