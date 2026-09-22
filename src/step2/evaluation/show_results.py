import pandas as pd

RATE = 9.98

results = pd.read_csv('data/models/results_step2.csv')

results['rmse_mad'] = (results['rmse'] / RATE).round(0)
results['mae_mad'] = (results['mae'] / RATE).round(0)

print("\n" + "=" * 68)
print(results.to_string(
    index=False,
    formatters={'rmse': '{:,.0f}'.format, 'mae': '{:,.0f}'.format},
))
print("=" * 68)

print("\nResultats convertis en MAD (1 MAD = 9.98 INR) :")
for _, row in results.iterrows():
    print(f"{row['model']:<20} RMSE {row['rmse_mad']:>10,.0f} MAD"
          f"   MAE {row['mae_mad']:>10,.0f} MAD   R2 {row['r2']:.3f}")

print("\nInterpretation :")
print("- MAE   = erreur moyenne (en MAD)")
print("- RMSE  = erreur moyenne, grosses erreurs penalisees (en MAD)")
print("- R2    = part de variance expliquee (1 = parfait)")