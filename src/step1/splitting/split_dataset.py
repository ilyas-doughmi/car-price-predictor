import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    df = pd.read_csv('data/processed/car_price_encoded.csv')

    X = df.drop(columns=['selling_price'])
    y = df['selling_price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print("X_train:", X_train.shape, "X_test:", X_test.shape)

    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    print("done")


if __name__ == '__main__':
    main()