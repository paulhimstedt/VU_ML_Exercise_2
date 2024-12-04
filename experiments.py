import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import seaborn as sns

def run_experiment(model, X_train, y_train, X_test, y_test, is_custom=False):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    # Generate classification report
    report = classification_report(y_test, predictions, output_dict=True)
    return mse, r2, report

def plot_results(results, output_dir, reports):
    sns.set_palette("husl")
    fig, axs = plt.subplots(2, 4, figsize=(20, 15))
    fig.suptitle("Model Performance Comparison")

    models = ['Custom RF', 'Sklearn RF', 'Decision Tree', 'KNN']
    datasets = ['Adult Dataset', 'Spambase Dataset']

    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            metrics = ['MSE', 'R2']
            scores = [results[dataset][model]['mse'], results[dataset][model]['r2']]
            # Plotting with additional details
            axs[i, j].plot(metrics, scores, marker='o', label=f"{model} Scores")
            axs[i, j].fill_between(metrics, [0, 0], scores, alpha=0.2)
            axs[i, j].scatter(metrics, scores, color='red')
            axs[i, j].set_ylim(0, 1)
            axs[i, j].legend()
            axs[i, j].set_title(f"{model} on {dataset}")
            axs[i, j].set_xlabel('Metric')
            axs[i, j].set_ylabel('Score')

    # Plot classification report heatmaps
    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            report = reports[dataset][model]
            sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap="Greens" if model != 'Custom RF' else "PuRd", ax=axs[i, j+2])
            axs[i, j+2].set_title(f"{model} Report on {dataset}")

    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    plt.savefig(f"{output_dir}/model_performance_comparison.png")
    plt.close()
