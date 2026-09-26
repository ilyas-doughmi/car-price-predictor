import os
import subprocess
import sys

STEPS = [
    'src/step1/cleaning/data_cleaning.py',
    'src/step1/cleaning/outliers.py',
    'src/step1/encoding/encode_categorical.py',
    'src/step1/splitting/split_dataset.py',
    'src/step1/scaling/scale_features.py',
    'src/step2/training/train_models.py',
    'src/step3/optimization/optimize_xgboost.py',
    'src/step3/retrain/retrain_best.py',
    'src/step4/evaluation/compare_models.py',
    'src/step4/evaluation/analyze_errors.py',
    'src/step5/predict.py',
]


def main():
    for script in STEPS:
        print("\n=== RUN %s ===" % script)
        result = subprocess.run([sys.executable, script], cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if result.returncode != 0:
            sys.exit("Failed at step: %s" % script)


if __name__ == '__main__':
    main()