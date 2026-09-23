import json
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV


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

    baseline = XGBRegressor(random_state=42)
    baseline.fit(X_train, log_y_train)
    y_pred_before = np.expm1(baseline.predict(X_test))
    rmse_before, mae_before, r2_before = compute_metrics(y_test, y_pred_before)
    print("AVANT (defaut) -> RMSE:", rmse_before, "MAE:", mae_before, "R2:", r2_before)

    param_grid = {
        'learning_rate': [0.05, 0.1],
        'max_depth': [4, 6, 8],
        'n_estimators': [300, 600],
        'subsample': [0.7, 1.0],
    }

    grid = GridSearchCV(
        XGBRegressor(random_state=42),
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
    )
    grid.fit(X_train, log_y_train)

    y_pred_after = np.expm1(grid.best_estimator_.predict(X_test))
    rmse_after, mae_after, r2_after = compute_metrics(y_test, y_pred_after)

    print("Meilleurs parametres:", grid.best_params_)
    print("APRES (optimise)    -> RMSE:", rmse_after, "MAE:", mae_after, "R2:", r2_after)

    results = pd.DataFrame([{
        'model': 'XGBoost',
        'rmse_before': rmse_before, 'mae_before': mae_before, 'r2_before': r2_before,
        'rmse_after': rmse_after, 'mae_after': mae_after, 'r2_after': r2_after,
    }])
    results.to_csv('data/models/results_step3.csv', index=False)

    with open('data/models/best_params.json', 'w') as f:
        json.dump(grid.best_params_, f)
    print("saved: results_step3.csv + best_params.json")


if __name__ == '__main__':
    main()