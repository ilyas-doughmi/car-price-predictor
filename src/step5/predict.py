import pandas as pd
import numpy as np
import joblib

RATE = 9.98  # 1 MAD = 9.98 INR
COLUMNS = list(pd.read_csv('data/processed/X_train.csv').columns)

_CLEAN = pd.read_csv('data/processed/car_price_clean.csv')
BM_MAP = dict(zip(_CLEAN['bm'], _CLEAN['bm'].astype('category').cat.codes))

OWNER_MAP = {
    'First Owner': 0,
    'Second Owner': 1,
    'Third Owner': 2,
    'Fourth & Above Owner': 3,
    'Test Drive Car': 0,
}


def get_bm_code(bm):
    if bm in BM_MAP:
        return BM_MAP[bm]
    return max(BM_MAP.values()) + 1


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


def main():
    model = joblib.load('models/model.pkl')

    print("=== PREDICTION D'UNE NOUVELLE VOITURE ===")
    brand = input("Marque (defaut Maruti) : ") or "Maruti"
    model_name = input("Modele (defaut Swift) : ") or "Swift"
    year = input("Annee (defaut 2017) : ") or "2017"
    km = input("Kilometrage (defaut 50000) : ") or "50000"
    owner = input("Owner (defaut First Owner) : ") or "First Owner"
    fuel = input("Fuel (defaut Diesel) : ") or "Diesel"
    seller = input("Seller (defaut Individual) : ") or "Individual"
    trans = input("Transmission (defaut Manual) : ") or "Manual"

    row = encode_input(int(year), int(km), owner, brand, model_name, fuel, seller, trans)
    price_inr = float(np.expm1(model.predict(row))[0])
    print(f"\nPrix estime : {price_inr:,.0f} INR (~{price_inr / RATE:,.0f} MAD)")


if __name__ == '__main__':
    main()
