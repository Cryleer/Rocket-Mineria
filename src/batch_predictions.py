import pandas as pd
import joblib
import os

RAW_DATA_PATH = "data/processed/processed_matches.csv"
ENCODED_DATA_PATH = "data/processed/processed_encoded.csv"

MODEL_PATH = "data/models/random_forest_model.pkl"
WINNER_ENCODER_PATH = "data/models/winner_encoder.pkl"

OUTPUT_PATH = "data/processed/model_predictions.csv"

def main():
    # Cargar datos originales (con game_mode, team_color, etc.)
    raw = pd.read_csv(RAW_DATA_PATH)

    # Cargar datos codificados (para las features numericas)
    encoded = pd.read_csv(ENCODED_DATA_PATH)

    # Cargar modelo
    model = joblib.load(MODEL_PATH)
    winner_encoder = joblib.load(WINNER_ENCODER_PATH)

    features = [
        'team_color_encoded',
        'goal_difference',
        'match_duration',
        'mode_Duel',
        'mode_Doubles',
        'mode_Standard',
        'is_competitive',
        'overtime'
    ]

    X = encoded[features]

    # Predicción numérica
    preds = model.predict(X)
    pred_labels = winner_encoder.inverse_transform(preds)

    # Probabilidades
    proba = model.predict_proba(X)

    # Unir todo en un solo dataframe completo
    out = raw.copy()
    out["predicted_winner"] = pred_labels
    out["proba_blue"] = proba[:, 0]
    out["proba_orange"] = proba[:, 1]
    out["proba_tie"] = proba[:, 2] if proba.shape[1] > 2 else 0

    out.to_csv(OUTPUT_PATH, index=False)

    print("Archivo generado:", OUTPUT_PATH)
    print("Shape:", out.shape)

if __name__ == "__main__":
    main()
