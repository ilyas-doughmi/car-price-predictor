import json
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


def load_data():
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
    y_test = pd.read_csv('data/processed/y_test.csv').squeeze()
    return X_train, X_test, y_train, y_test


def main():
    X_train, X_test, y_train, y_test = load_data()

    log_y_train = np.log1p(y_train)

    with open('data/models/best_params.json') as f:
        best_params = json.load(f)
    print("Parametres optimises:", best_params)

    model = XGBRegressor(random_state=42, **best_params)
    model.fit(X_train, log_y_train)

    y_pred = np.expm1(model.predict(X_test))
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("R2:", r2, "RMSE:", rmse, "MAE:", mae)

    joblib.dump(model, 'models/model.pkl')
    print("model saved: models/model.pkl")


if __name__ == '__main__':
    main()