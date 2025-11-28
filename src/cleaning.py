import pandas as pd

def clean_data(df):
    # Convertir fechas a formato datetime
    df['match_date'] = pd.to_datetime(df['match_date'])

    # Asegurar tipos correctos
    df['team_color'] = df['team_color'].astype(str)
    df['game_mode'] = df['game_mode'].astype(str)
    df['winner'] = df['winner'].astype(str)

    return df

if __name__ == "__main__":
    df = pd.read_excel('data/raw/rocket_league_matches.xlsx')
    df_clean = clean_data(df)
    df_clean.to_csv('data/processed/processed_matches.csv', index=False)
    print("Archivo generado: data/processed/processed_matches.csv")
    print("Shape:", df_clean.shape)
