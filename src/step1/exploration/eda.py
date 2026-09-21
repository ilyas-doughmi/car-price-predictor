import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/processed/car_price_clean.csv')

print(df.describe(include='all'))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df['selling_price'], bins=30, kde=True, ax=axes[0])
axes[0].set_title('Distribution - Selling Price')
sns.histplot(df['km_driven'], bins=30, kde=True, ax=axes[1])
axes[1].set_title('Distribution - Km Driven')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df.select_dtypes('number').corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x='year', y='selling_price', hue='transmission', data=df)
plt.title('Year vs Selling Price')
plt.show()

sns.pairplot(df.select_dtypes('number'))
plt.show()