import pandas as pd
import matplotlib.pyplot as plt


def get_bounds(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return lower, upper


def detect_outliers(df):
    for col in ['selling_price', 'km_driven', 'year']:
        lower, upper = get_bounds(df, col)
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        print(col, "outliers:", outliers)


def plot_boxplots(df):
    df.boxplot(column=['selling_price', 'km_driven', 'year'])
    plt.title('Boxplots - outliers detection')
    plt.savefig('data/models/outliers_boxplots.png', dpi=120)


def treat_outliers(df):
    for col in ['selling_price', 'km_driven', 'year']:
        lower, upper = get_bounds(df, col)
        df[col] = df[col].clip(lower, upper)
    return df


def main():
    df = pd.read_csv('data/processed/car_price_clean.csv')

    detect_outliers(df)
    plot_boxplots(df)
    df = treat_outliers(df)

    df.to_csv('data/processed/car_price_clean.csv', index=False)
    print("done")


if __name__ == '__main__':
    main()