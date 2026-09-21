import pandas as pd


def encode_categoricals(df):
    df = pd.get_dummies(df, columns=['fuel', 'seller_type', 'transmission'], dtype=int)

    owner_map = {
        'First Owner': 0,
        'Second Owner': 1,
        'Third Owner': 2,
        'Fourth & Above Owner': 3,
        'Test Drive Car': 0,
    }
    df['owner'] = df['owner'].map(owner_map)
    return df
