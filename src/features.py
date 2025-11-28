import pandas as pd
import numpy as np

def create_features(df):

    # 1) Categoría por diferencia de goles
    df['goal_diff_category'] = pd.cut(
        df['goal_difference'],
        bins=[-np.inf, -3, -1, 1, 3, np.inf],
        labels=['large_loss', 'small_loss', 'close', 'small_win', 'large_win']
    )

    # 2) bucket de duración del partido
    df['duration_bucket'] = pd.cut(
        df['match_duration'],
        bins=[0, 300, 360, 420, np.inf],
        labels=['short', 'normal', 'long', 'very_long']
    )

    # 3) indicador si el match fue competitivo (goles cercanos)
    df['is_competitive'] = (df['goal_difference'].abs() <= 2).astype(int)

    # 4) feature combinada: equipo + modo
    df['team_mode'] = df['team_color'] + '_' + df['game_mode']

    return df

if __name__ == "__main__":
    df = pd.read_csv('data/processed/processed_matches.csv')
    df = create_features(df)
    df.to_csv('data/processed/processed_features.csv', index=False)
    print("Features creadas correctamente.")
    print("Archivo: data/processed/processed_features.csv")
    print("Shape:", df.shape)
