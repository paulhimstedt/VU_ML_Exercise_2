import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def run_experiment(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, r2

def plot_results(results, title, output_dir):
    plt.figure()
    plt.title(title)
    plt.plot(results['mse'], label='MSE')
    plt.plot(results['r2'], label='R2 Score')
    plt.legend()
    plt.savefig(f"{output_dir}/{title.replace(' ', '_').lower()}.png")
    plt.close()
