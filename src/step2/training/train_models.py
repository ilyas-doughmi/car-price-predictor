import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from models import get_models


def load_data():
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
    y_test = pd.read_csv('data/processed/y_test.csv').squeeze()
    return X_train, X_test, y_train, y_test


def compute_metrics(y_test, y_pred):
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return rmse, mae, r2


def main():
    X_train, X_test, y_train, y_test = load_data()

    log_y_train = np.log1p(y_train)

    results = []
    for name, model in get_models().items():
        pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
        pipeline.fit(X_train, log_y_train)
        y_pred = np.expm1(pipeline.predict(X_test))
        rmse, mae, r2 = compute_metrics(y_test, y_pred)
        print(name, "-> RMSE:", rmse, "MAE:", mae, "R2:", r2)
        results.append([name, rmse, mae, r2])

    pd.DataFrame(results, columns=['model', 'rmse', 'mae', 'r2']).to_csv(
        'data/models/results_step2.csv', index=False)
    print("done")


if __name__ == '__main__':
    main()