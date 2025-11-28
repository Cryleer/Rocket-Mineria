from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Cargar modelo y encoders
model = joblib.load('../data/models/random_forest_model.pkl')
team_enc = joblib.load('../data/models/team_encoder.pkl')
winner_enc = joblib.load('../data/models/winner_encoder.pkl')

class MatchInput(BaseModel):
    team_color: str
    game_mode: str
    goal_difference: int
    match_duration: int
    overtime: bool

@app.get("/")
def root():
    return {"message": "API funcionando correctamente!"}

@app.post("/predict")
def predict_match(match: MatchInput):

    # === Encoding de team_color ===
    team_color_encoded = team_enc.transform([match.team_color])[0]

    # === One-hot para game_mode ===
    mode_duel = 1 if match.game_mode == "Duel" else 0
    mode_doubles = 1 if match.game_mode == "Doubles" else 0
    mode_standard = 1 if match.game_mode == "Standard" else 0

    # === Feature adicional ===
    is_competitive = 1 if abs(match.goal_difference) <= 2 else 0

    # === Crear vector de entrada ===
    X = np.array([[
        team_color_encoded,
        match.goal_difference,
        match.match_duration,
        mode_duel,
        mode_doubles,
        mode_standard,
        is_competitive,
        int(match.overtime)
    ]])

    # === Predicción ===
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    winner = winner_enc.inverse_transform([pred])[0]

    return {
        "winner_prediction": winner,
        "confidence": float(proba[pred]),
        "probabilities": {
            "Blue": float(proba[0]),
            "Draw": float(proba[1]),
            "Orange": float(proba[2])
        }
    }
