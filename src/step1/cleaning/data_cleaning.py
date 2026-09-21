import pandas as pd


def replace_empty(df):
    return df.replace(r'^\s*$', None, regex=True)


def impute_missing(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())
    return df


def drop_duplicates(df):
    return df.drop_duplicates()
