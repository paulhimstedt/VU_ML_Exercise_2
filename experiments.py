import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def run_experiment(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, r2

def plot_results(results, output_dir):
    sns.set_palette("husl")
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Model Performance Comparison")

    models = ['Custom RF', 'Sklearn RF', 'Decision Tree', 'KNN']
    datasets = ['Adult Dataset', 'Spambase Dataset']

    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            metrics = ['MSE', 'R2']
            scores = [results[dataset][model]['mse'], results[dataset][model]['r2']]
            axs[i, j].plot(metrics, scores, marker='o')
            axs[i, j].set_title(f"{model} on {dataset}")
            axs[i, j].set_xlabel('Metric')
            axs[i, j].set_ylabel('Score')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{output_dir}/model_performance_comparison.png")
    plt.close()
