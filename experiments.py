import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def run_experiment(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, r2

def plot_results(results, title, output_dir):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(title)

    axs[0].plot(results['mse'], label='MSE')
    axs[0].set_title('Mean Squared Error')
    axs[0].set_xlabel('Model')
    axs[0].set_ylabel('MSE')
    axs[0].legend()

    axs[1].plot(results['r2'], label='R2 Score')
    axs[1].set_title('R2 Score')
    axs[1].set_xlabel('Model')
    axs[1].set_ylabel('R2')
    axs[1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{output_dir}/{title.replace(' ', '_').lower()}.png")
    plt.close()
