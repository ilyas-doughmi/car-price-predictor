# Car Price Predictor

Predicting the resale price (`selling_price`) of used cars, in INR, from a
CarDekho dataset (4340 cars). Machine Learning pipeline, end to end.

## Pipeline

| Step | Scripts | Purpose |
|---|---|---|
| 1 | `src/step1/...` | cleaning (impute, dedup, outliers IQR+clip), feature engineering `bm` (brand+model), encoding, split 80/20, scaling |
| 2 | `src/step2/...` | 4 baseline models trained on `log(price)` |
| 3 | `src/step3/...` | GridSearchCV to optimize XGBoost, retrain best model |
| 4 | `src/step4/...` | final comparison + error analysis |
| 5 | `src/step5/predict.py` | interactive price prediction |

## Run everything

```bash
python src/step1/cleaning/data_cleaning.py
python src/step1/cleaning/outliers.py
python src/step1/encoding/encode_categorical.py
python src/step1/splitting/split_dataset.py
python src/step1/scaling/scale_features.py
python src/step2/training/train_models.py
python src/step3/optimization/optimize_xgboost.py
python src/step3/retrain/retrain_best.py
python src/step4/evaluation/compare_models.py
python src/step4/evaluation/analyze_errors.py
python src/step5/predict.py
```

## Key choices

- Extract brand+model (`bm`) from the `name` column instead of dropping it.
- Target = `log(price)` so the model optimizes relative error.
- 1 MAD = 9.98 INR (results shown in both currencies).