import pandas as pd
import numpy as np
import joblib

RATE = 9.98  # 1 MAD = 9.98 INR
COLUMNS = list(pd.read_csv('data/processed/X_train.csv').columns)

_CLEAN = pd.read_csv('data/processed/car_price_clean.csv')
BM_MAP = dict(zip(_CLEAN['bm'], _CLEAN['bm'].astype('category').cat.codes))


def get_bm_code(bm):
    if bm in BM_MAP:
        return BM_MAP[bm]
    return max(BM_MAP.values()) + 1

OWNER_MAP = {
    'First Owner': 0,
    'Second Owner': 1,
    'Third Owner': 2,
    'Fourth & Above Owner': 3,
    'Test Drive Car': 0,
}


def encode_input(year, km_driven, owner, brand, model, fuel, seller_type, transmission):
    df = pd.DataFrame([{
        'year': year,
        'km_driven': km_driven,
        'owner': OWNER_MAP[owner],
        'bm': (brand + ' ' + model).strip(),
        'fuel': fuel,
        'seller_type': seller_type,
        'transmission': transmission,
    }])
    df = pd.get_dummies(df, columns=['fuel', 'seller_type', 'transmission', 'bm'], dtype=int)
    df['bm_code'] = get_bm_code((brand + ' ' + model).strip())
    return df.reindex(columns=COLUMNS, fill_value=0)


def predict_price(model, row):
    price_inr = float(np.expm1(model.predict(row))[0])
    return price_inr, price_inr / RATE


def demo_on_real_cars(model):
    df = pd.read_csv('data/processed/car_price_encoded.csv')
    sample = df.sample(5, random_state=42)
    X_sample = sample.drop(columns=['selling_price'])
    real = sample['selling_price']

    print("\n=== DEMO : prediction sur 5 voitures reelles du dataset ===")
    for idx, row in X_sample.iterrows():
        price_inr, _ = predict_price(model, row.to_frame().T)
        real_inr = real.loc[idx]
        err = abs(price_inr - real_inr) / real_inr * 100
        print(f"  reel: {real_inr:>10,.0f} INR | predit: {price_inr:>10,.0f} INR"
              f" | erreur: {err:5.1f} %")


def interactive_prediction(model):
    print("\n=== PREDICTION D'UNE NOUVELLE VOITURE ===")
    print("(appuie sur Entree pour la valeur par defaut)")

    brand = input("Marque (defaut Maruti) : ") or "Maruti"
    model_name = input("Modele (defaut Swift) : ") or "Swift"
    year = input("Annee (defaut 2017) : ") or "2017"
    km = input("Kilometrage (defaut 50000) : ") or "50000"
    owner = input("Owner (defaut First Owner) : ") or "First Owner"
    fuel = input("Fuel - Petrol/Diesel/CNG/LPG/Electric (defaut Diesel) : ") or "Diesel"
    seller = input("Seller type - Dealer/Individual/Trustmark Dealer (defaut Individual) : ") or "Individual"
    trans = input("Transmission - Manual/Automatic (defaut Manual) : ") or "Manual"

    row = encode_input(int(year), int(km), owner, brand, model_name, fuel, seller, trans)
    price_inr, price_mad = predict_price(model, row)

    print(f"\nPrix estime : {price_inr:,.0f} INR (~{price_mad:,.0f} MAD)")


def main():
    model = joblib.load('models/model.pkl')

    demo_on_real_cars(model)
    interactive_prediction(model)


if __name__ == '__main__':
    main()