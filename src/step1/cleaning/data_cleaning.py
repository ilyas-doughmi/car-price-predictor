import pandas as pd
import numpy as np


def add_brand_model(df):
    df['bm'] = df['name'].str.split().str[:2].str.join(' ')
    df = df.drop(columns=['name'])
    return df



def replace_empty(df):
    return df.replace(r'^\s*$', np.nan, regex=True)


def impute_missing(df):
    df["km_driven"] = df["km_driven"].fillna(df["km_driven"].median())
    df["year"] = df["year"].fillna(df["year"].median())

    df["fuel"] = df["fuel"].fillna(df["fuel"].mode()[0])
    df["seller_type"] = df["seller_type"].fillna(df["seller_type"].mode()[0])
    df["owner"] = df["owner"].fillna(df["owner"].mode()[0])
    return df


def drop_duplicates(df):
    df = df.drop_duplicates()
    return df


def main():
    df = pd.read_csv('data/raw/car_price.csv')
    print("Before cleaning:", df.shape)

    df = replace_empty(df)
    df = impute_missing(df)

    print("missing value after cleaning:", df.isnull().sum().sum())

    print("Duplicates before:", df.duplicated().sum())
    df = drop_duplicates(df)
    print("Duplicates after:", df.duplicated().sum())

    df = add_brand_model(df)
    print("Shape after cleaning + brand/model:", df.shape)

    df.to_csv('data/processed/car_price_clean.csv', index=False)
    print("done")


if __name__ == '__main__':
    main()