import pandas as pd


def encode_categoricals(df):
    df = pd.get_dummies(df, columns=['fuel', 'seller_type', 'transmission'], dtype=int)
    return df
