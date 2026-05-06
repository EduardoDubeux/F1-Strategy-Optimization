import pandas as pd


# =============================
# 🛞 DEGRADAÇÃO NÃO LINEAR
# =============================

def compute_tire_degradation(tyre_life, compound):
    if compound == "SOFT":
        return 0.06 * tyre_life + 0.003 * (tyre_life ** 2)
    elif compound == "MEDIUM":
        return 0.04 * tyre_life + 0.002 * (tyre_life ** 2)
    elif compound == "HARD":
        return 0.025 * tyre_life + 0.001 * (tyre_life ** 2)
    return 0


# =============================
# 🏁 CONTEXTO DE CORRIDA
# =============================

def add_race_context(df):
    df = df.copy()

    # posição de largada (grid)
    first_lap = df[df['LapNumber'] == 1]
    grid_map = dict(zip(first_lap['Driver'], first_lap['Position']))

    df['GridPosition'] = df['Driver'].map(grid_map)

    return df


# =============================
# FEATURE ENGINEERING
# =============================

def build_features(df):
    df = df.copy()

    # contexto
    df = add_race_context(df)

    # TyreLife
    if 'TyreLife' not in df.columns:
        df['TyreLife'] = df.groupby(['Driver', 'Stint']).cumcount()

    # StintLap
    df['StintLap'] = df.groupby(['Driver', 'Stint']).cumcount() + 1

    # Degradação
    df['TireDeg'] = df.apply(
        lambda row: compute_tire_degradation(
            row['TyreLife'],
            row['Compound']
        ),
        axis=1
    )

    # Performance do piloto
    df['DriverPace'] = df.groupby('Driver')['LapTime'].transform('mean')

    # One-hot encoding
    df = pd.get_dummies(df, columns=['Compound'])

    return df


# =============================
# PREPARAÇÃO PARA MODELOS
# =============================

def prepare_model_data(df):
    features = [
        'LapNumber',
        'TyreLife',
        'Stint',
        'StintLap',
        'TireDeg',
        'GridPosition',
        'DriverPace'
    ]

    compound_cols = [col for col in df.columns if "Compound_" in col]
    features.extend(compound_cols)

    X = df[features]
    y = df['LapTime']

    return X, y