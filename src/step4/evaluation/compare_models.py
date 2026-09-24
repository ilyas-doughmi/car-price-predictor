import pandas as pd
import matplotlib.pyplot as plt

RATE = 9.98  # 1 MAD = 9.98 INR

base = pd.read_csv('data/models/results_step2.csv')
opt = pd.read_csv('data/models/results_step3.csv')

rows = []
for _, row in base.iterrows():
    rows.append({'model': row['model'], 'stage': 'baseline',
                 'rmse': row['rmse'], 'mae': row['mae'], 'r2': row['r2']})
rows.append({'model': 'XGBoost optimise', 'stage': 'optimise',
             'rmse': opt['rmse_after'].iloc[0],
             'mae': opt['mae_after'].iloc[0],
             'r2': opt['r2_after'].iloc[0]})
df = pd.DataFrame(rows)

df['mae_mad'] = (df['mae'] / RATE).round(0)
df['rmse_mad'] = (df['rmse'] / RATE).round(0)

print("=== COMPARAISON FINALE DES MODELES ===")
print(df[['model', 'stage', 'r2', 'mae_mad', 'rmse_mad']].to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

colors = ['#bbbbbb', '#bbbbbb', '#bbbbbb', '#bbbbbb', '#1f77b4']

axes[0].barh(df['model'], df['r2'], color=colors)
axes[0].axvline(0, color='black', linewidth=0.8)
axes[0].set_title('R2 (1 = parfait)')

axes[1].barh(df['model'], df['mae_mad'], color=colors)
axes[1].set_title('MAE (en MAD)')

axes[2].barh(df['model'], df['rmse_mad'], color=colors)
axes[2].set_title('RMSE (en MAD)')

plt.tight_layout()
plt.savefig('data/models/metrics_comparison.png', dpi=120)
print("graph saved: data/models/metrics_comparison.png")