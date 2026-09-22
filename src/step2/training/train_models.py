import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from step2.training.models import get_models


def load_data():
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
    y_test = pd.read_csv('data/processed/y_test.csv').squeeze()
    return X_train, X_test, y_train, y_test


def main():
    X_train, X_test, y_train, y_test = load_data()

    rows = []
    for name, model in get_models().items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rows.append({
            'model': name,
            'R2': r2_score(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': mean_squared_error(y_test, y_pred, squared=False),
        })

    results = pd.DataFrame(rows).sort_values('R2', ascending=False)
    results.to_csv('data/models/results_step2.csv', index=False)
    print(results.to_string(index=False))
    joblib.dump(get_models(), 'data/models/baselines.pkl')


if __name__ == '__main__':
    main()
