import pandas as pd

# Cargar tu dataset
df = pd.read_excel('data/raw/rocket_league_matches.xlsx')

print("=== SHAPE (Filas, Columnas) ===")
print(df.shape)

print("\n=== TIPOS DE DATOS ===")
print(df.dtypes)

print("\n=== PRIMERAS 5 FILAS ===")
print(df.head())

print("\n=== NULOS POR COLUMNA ===")
print(df.isnull().sum())

print("\n=== ESTADÍSTICAS ===")
print(df.describe(include='all'))
