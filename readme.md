# 🚗 Prédicteur de Prix de Voitures d'Occasion

Un projet de **Machine Learning de bout en bout** : prédire le prix de revente
(`selling_price`) d'une voiture d'occasion à partir de ses caractéristiques.

🏆 **Résultat final :** R² = **0.862** · erreur moyenne ≈ **7 586 MAD**
(~75 706 INR) · précision ≈ **80 %**.

---

## 🎯 Objectif du projet

Ce projet montre toute la chaîne d'un projet Data Science réel, du **fichier
brut** au **modèle entraîné et prêt à prédire** :

1. Explorer et comprendre les données
2. Nettoyer et préparer les données (valeurs manquantes, doublons, outliers)
3. **Feature engineering** : extraire la marque + le modèle (`bm`)
4. Encoder, scinder (train/test), normaliser
5. Comparer 4 modèles de régression
6. Optimiser le meilleur avec GridSearchCV
7. Sauvegarder et utiliser le modèle (`predict.py`)

---

## 📊 Le dataset

Données classiques **CarDekho** (voitures d'occasion vendues en Inde) :

| Colonne | Description |
|---|---|
| `name` | Nom complet de la voiture (marque + modèle + version) |
| `year` | Année de mise en circulation |
| `selling_price` | **Prix de vente (en INR = roupies indiennes)** ← à prédire |
| `km_driven` | Kilométrage parcouru |
| `fuel` | Carburant (Petrol, Diesel, CNG, LPG, Electric) |
| `seller_type` | Vendeur (Dealer, Individual, Trustmark Dealer) |
| `transmission` | Boîte (Manual, Automatic) |
| `owner` | Nombre/type de propriétaires (First, Second…) |

- **4340 lignes** → **3672** après nettoyage
- **204 colonnes** après encodage
- Split : **2937 train / 735 test** (80/20, `random_state=42`)

> 💱 **Conversion utilisée dans le projet :** 1 MAD = 9.98 INR.

---

## 🛠️ Les 6 étapes de la chaîne

| Étape | Scripts | Ce que ça fait |
|---|---|---|
| **1. Nettoyage** | `src/step1/cleaning/` | impute les valeurs manquantes (médiane/mode), supprime 668 doublons, borne les outliers (IQR + `clip`) |
| **1. Feature eng.** | `src/step1/cleaning/data_cleaning.py` | crée `bm` = marque + modèle (2 premiers mots de `name`) |
| **1. Encodage** | `src/step1/encoding/encode_categorical.py` | one-hot (`fuel`, `seller_type`, `transmission`, `bm`) + label (`owner`) + `bm_code` |
| **1. Split + scaling** | `src/step1/splitting/`, `src/step1/scaling/` | train/test 80/20 + StandardScaler (fit sur train uniquement) |
| **2. Modèles de base** | `src/step2/training/train_models.py` | entraîne 4 baselines sur **log(prix)** |
| **3. Optimisation** | `src/step3/optimization/optimize_xgboost.py` | GridSearchCV (5 folds) sur XGBoost |
| **3bis. Réentraînement** | `src/step3/retrain/retrain_best.py` | meilleur modèle → `models/model.pkl` |
| **4. Évaluation** | `src/step4/evaluation/` | tableau des scores + analyse des erreurs par tranche de prix |
| **5. Prédiction** | `src/step5/predict.py` | prédire le prix d'une nouvelle voiture (interactif) |

### 🧠 Les 2 choix qui ont tout changé

1. **Marque + modèle (`bm`)** au lieu de jeter le nom : le prix d'une voiture
   dépend **surtout du modèle** (+30 points de R²).
2. **Prédire log(prix)** : les prix sont très asymétriques ; apprendre le
   logarithme minimise l'**erreur relative** (le « pour-cent »), pas l'erreur
   absolue → les petits prix sont traités équitablement.

---

## 📈 Résultats

Tous les modèles sont évalués sur le **test** (données jamais vues), cible log :
métriques reconverties en prix réel.

| Modèle | R² | MAE (INR) | MAE (MAD) | RMSE (INR) |
|---|---|---|---|---|
| Linear Regression | 0.828 | 78 335 | 7 849 | 130 047 |
| Random Forest | 0.796 | 86 212 | 8 638 | 141 605 |
| XGBoost (base) | 0.843 | 80 090 | 8 025 | 123 920 |
| SVR | 0.827 | 80 043 | 8 020 | 130 376 |
| **XGBoost optimisé** 🏆 | **0.862** | **75 706** | **7 586** | **116 317** |

**Meilleurs hyperparamètres :**
`learning_rate=0.1`, `max_depth=4`, `n_estimators=600`, `subsample=0.7`.

**Erreur par tranche de prix (homogène ~16-18 % sur le marché principal) :**

| Tranche (INR) | Erreur moyenne | Nb |
|---|---|---|
| 0 – 200k | 28.6 % | 183 |
| 200k – 400k | 16.4 % | 238 |
| 400k – 600k | 17.9 % | 138 |
| 600k – 800k | 15.9 % | 81 |
| 800k – 1M | 16.1 % | 33 |
| > 1M | 15.4 % | 62 |

---

## ▶️ Installation et lancement

### Prérequis

- Python 3.10+ (testé sur 3.12+)
- Les bibliothèques : `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `joblib`

```bash
pip install pandas numpy scikit-learn xgboost matplotlib joblib
```

### Relancer toute la chaîne (depuis la racine du projet)

```bash
# Étape 1 — préparation des données
python src/step1/cleaning/data_cleaning.py
python src/step1/cleaning/outliers.py
python src/step1/encoding/encode_categorical.py
python src/step1/splitting/split_dataset.py
python src/step1/scaling/scale_features.py

# Étape 2 — 4 modèles de base
python src/step2/training/train_models.py
python src/step2/evaluation/show_results.py

# Étape 3 — optimisation
python src/step3/optimization/optimize_xgboost.py
python src/step3/retrain/retrain_best.py

# Étape 4 — évaluation finale
python src/step4/evaluation/compare_models.py
python src/step4/evaluation/analyze_errors.py

# Étape 5 — prédiction (interactif)
python src/step5/predict.py
```

### Utiliser la prédiction

```bash
python src/step5/predict.py
```

Réponds aux questions (marque, modèle, année, kilométrage, carburant…)
et obtiens un prix estimé en INR **et en MAD**.

---

## 📁 Structure du projet

```
car-price-predictor
├── data/
│   ├── raw/car_price.csv          # dataset brut (gardé dans git)
│   └── processed/                 # données nettoyées/encodées (généré, ignoré)
├── models/
│   └── model.pkl                  # modèle final entraîné (XGBoost optimisé)
├── src/
│   ├── step1/   exploration + nettoyage + encodage + split + scaling
│   ├── step2/   modèles de base + affichage des résultats
│   ├── step3/   optimisation (GridSearchCV) + réentraînement
│   ├── step4/   comparaison finale + analyse des erreurs
│   └── step5/   prédiction interactive
├── docs/        documentation pédagogique (débrief, concepts)
└── readme.md
```

Les fichiers **générés** (`data/processed/`, `data/models/` : metrics, graphiques,
`best_params.json`) et `docs/` sont ignorés par `.gitignore`. Le **modèle final**
`models/model.pkl` est, lui, gardé pour être livré avec le projet.

---

## 📚 Documentation

| Fichier | Contenu |
|---|---|
| `docs/debrief.md` | guide de soutenance : toutes les notions (R², MAE, RMSE, MAPE…), Q&R probable |
| `docs/concepts.md` | tous les concepts : utilisés / non utilisés et pourquoi |
| `docs/step1.md` | documentation pédagogique détaillée de l'étape 1 |
| `docs/step6.md` | rapport final complet |

---

## ⚠️ Limites connues (assumées)

- L'**état réel du véhicule** (accidents, entretien…) n'est pas dans le dataset :
  ~16-20 % d'erreur incompressible avec ces seules variables.
- Les voitures **très bon marché** (< 200k INR) sont les plus difficiles à prédire
  (~28 % d'erreur).
- Un modèle à plus de 90 % de précision **n'est pas honnêtement atteignable** sur
  ce dataset : au-delà de ~80 %, le modèle **mémorise** les prix déjà vus
  (sur-apprentissage) au lieu d'apprendre.

### 🔮 Améliorations possibles

- Ajouter l'état, la version exacte (trim) et le nombre de propriétaires détaillé
- Recherche randomisée (RandomSearchCV) ou Bayesian (Optuna) sur une grille plus large
- Stacking XGBoost + LightGBM
- Transformer le tout en API ou interface Streamlit

---

## 🎓 Points clés à retenir

1. **Protocole honnête** : évaluation sur test jamais vu, même règle pour chaque modèle.
2. **Feature engineering** : `bm` (marque+modèle) = le haussement majeur du score.
3. **Cible log** : optimisation de l'erreur relative, idéale pour des prix très étalés.
4. **Optimisation** : GridSearchCV 5-folds → R² 0.843 → **0.862**, MAE −5 500 INR/voiture.
5. **Interprétation finale** : R² = 86 % de variance expliquée, précision ≈ 80 %,
   erreur moyenne ≈ 7 586 MAD.