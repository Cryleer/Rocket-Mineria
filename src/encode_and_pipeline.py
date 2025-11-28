import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Cargar dataset con features
df = pd.read_csv('data/processed/processed_features.csv')

# === 1. LabelEncoder para team_color ===
le_team = LabelEncoder()
df['team_color_encoded'] = le_team.fit_transform(df['team_color'])
joblib.dump(le_team, 'data/models/team_encoder.pkl')

# === 2. One-Hot Encoding para game_mode ===
df = pd.get_dummies(df, columns=['game_mode'], prefix='mode')

# === 3. LabelEncoder para winner (TARGET) ===
le_winner = LabelEncoder()
df['winner_encoded'] = le_winner.fit_transform(df['winner'])
joblib.dump(le_winner, 'data/models/winner_encoder.pkl')

# Guardar dataset final procesado
df.to_csv('data/processed/processed_encoded.csv', index=False)

print("Dataset final guardado en data/processed/processed_encoded.csv")
print("Encoders guardados en data/models/")
print("Shape final:", df.shape)
