import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

RATE = 9.98  # 1 MAD = 9.98 INR

model = joblib.load('data/models/model.pkl')
X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv').squeeze()

y_pred = np.expm1(model.predict(X_test))

mape = (np.abs(y_test - y_pred) / y_test).mean() * 100
print(f"MAPE (erreur moyenne en %) : {mape:.2f} %")
print(f"Precision approx. = 100 - MAPE : {100 - mape:.2f} %")
print(f"R2 (variance expliquee) : {r2_score(y_test, y_pred):.3f}")
print(f"MAE : {np.abs(y_test - y_pred).mean():,.0f} INR (~{(np.abs(y_test - y_pred).mean() / RATE):,.0f} MAD)")

errors = pd.DataFrame({'real': y_test, 'pred': y_pred})
errors['err_pct'] = np.abs(errors['pred'] - errors['real']) / errors['real'] * 100
errors['range'] = pd.cut(
    errors['real'],
    bins=[0, 200000, 400000, 600000, 800000, 1000000, float('inf')],
)

print("\nErreur moyenne par tranche de prix reelle (INR) :")
print(errors.groupby('range', observed=True)['err_pct'].agg(['mean', 'count']).round(2))

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.4)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, color='red', linewidth=1)
plt.xlabel('Prix reel (INR)')
plt.ylabel('Prix predit (INR)')
plt.title('Prediction vs Real')
plt.tight_layout()
plt.savefig('data/models/pred_vs_real.png', dpi=120)
print("graph saved: data/models/pred_vs_real.png")