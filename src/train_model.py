import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

# === Cargar dataset final ===
df = pd.read_csv('data/processed/processed_encoded.csv')

# === Selección de features ===
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

# Filtrar por si alguna columna no existe
features = [f for f in features if f in df.columns]

X = df[features]
y = df['winner_encoded']

# === Train / Test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# === Crear modelo ===
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# Entrenar
model.fit(X_train, y_train)

# === Evaluar ===
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')

print("=== MÉTRICAS DEL MODELO ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Macro: {f1:.4f}")
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# === Guardar modelo ===
joblib.dump(model, 'data/models/random_forest_model.pkl')

# === Guardar metadata ===
metadata = {
    'features_used': features,
    'accuracy': float(accuracy),
    'f1_macro': float(f1)
}
joblib.dump(metadata, 'data/models/model_metadata.pkl')

print("\nModelo guardado en data/models/random_forest_model.pkl")
print("Metadata guardada en data/models/model_metadata.pkl")
