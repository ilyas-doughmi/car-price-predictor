import pandas as pd
import matplotlib.pyplot as plt


def show_structure(df):
    print("Shape:", df.shape)
    print("Data types:", df.dtypes)
    print("First 5 rows:", df.head())


def show_missing(df):
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
    print("Missing values (%):", missing_pct)


def show_duplicates(df):
    print("Duplicate rows:", df.duplicated().sum())


def show_counts(df):
    for col in ['fuel', 'seller_type', 'transmission', 'owner']:
        print(col, ":", df[col].value_counts(dropna=False))


def plot_year_price(df):
    plt.scatter(df['year'], df['selling_price'])
    plt.title('Year vs Selling Price')
    plt.show()


def main():
    df = pd.read_csv('data/raw/car_price.csv')

    show_structure(df)
    show_missing(df)
    show_duplicates(df)
    show_counts(df)
    plot_year_price(df)


if __name__ == '__main__':
    main()