import pandas as pd


def encode_categoricals(df):
    bm_codes = df['bm'].astype('category').cat.codes

    df = pd.get_dummies(
        df,
        columns=['fuel', 'seller_type', 'transmission', 'bm'],
        dtype=int,
    )
    df['bm_code'] = bm_codes

    owner_map = {
        'First Owner': 0,
        'Second Owner': 1,
        'Third Owner': 2,
        'Fourth & Above Owner': 3,
        'Test Drive Car': 0,
    }
    df['owner'] = df['owner'].map(owner_map)
    return df


def main():
    df = pd.read_csv('data/processed/car_price_clean.csv')

    df = encode_categoricals(df)

    print("Shape after encoding:", df.shape)
    print("Columns:", df.columns.tolist())

    df.to_csv('data/processed/car_price_encoded.csv', index=False)
    print("done")


if __name__ == '__main__':
    main()