import pandas as pd

def create_features(df):
    df = df.copy()

    # Remover valores inválidos
    df = df[df['LapTimeSeconds'] > 0]

    # Normalizar idade do pneu
    df['TyreLife'] = df['TyreLife'].fillna(0)

    # Progresso da corrida
    max_lap = df['LapNumber'].max()
    df['RaceProgress'] = df['LapNumber'] / max_lap

    # Codificar pneus (One-hot simples)
    compounds = pd.get_dummies(df['Compound'], prefix='Compound')
    df = pd.concat([df, compounds], axis=1)

    # Features finais
    feature_cols = [
        'LapNumber',
        'TyreLife',
        'RaceProgress'
    ] + list(compounds.columns)

    target_col = 'LapTimeSeconds'

    return df, feature_cols, target_col