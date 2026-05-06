"""
features.py - Engenharia de Features para Predição de Lap Times na F1
Autor: Eduardo Sampaio Dubeux
"""

import pandas as pd
import numpy as np
from typing import Tuple


def compute_tire_degradation(tyre_life: int, compound: str) -> float:
    """Degradação calibrada para dados reais da F1"""
    if compound == "SOFT":
        base = 0.035
        quadratic = 0.0018
    elif compound == "MEDIUM":
        base = 0.022
        quadratic = 0.0010
    elif compound == "HARD":
        base = 0.015
        quadratic = 0.0006
    elif compound in ["INTERMEDIATE", "WET"]:
        base = 0.060
        quadratic = 0.004
    else:
        base = 0.025
        quadratic = 0.001
    
    return base * tyre_life + quadratic * (tyre_life ** 2)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ====================== FEATURES BÁSICAS ======================
    if 'TyreLife' not in df.columns or df['TyreLife'].isnull().all():
        df['TyreLife'] = df.groupby(['Driver', 'Stint']).cumcount() + 1

    df['StintLap'] = df.groupby(['Driver', 'Stint']).cumcount() + 1

    # ====================== FEATURES DE CONTEXTO ======================
    total_laps = df['LapNumber'].max()
    df['LapProgress'] = df['LapNumber'] / total_laps
    df['LapProgress_Squared'] = df['LapProgress'] ** 2

    df['FreshTyre'] = (df['TyreLife'] == 1).astype(int)

    # NOVA FEATURE - Pneu relativo à média da corrida
    df['TyreLifeNorm'] = df['TyreLife'] / df['TyreLife'].max()

    # ====================== DEGRADAÇÃO DOS PNEUS ======================
    df['TireDeg'] = df.apply(
        lambda row: compute_tire_degradation(int(row['TyreLife']), str(row['Compound'])), 
        axis=1
    )
    df['TireDeg_Squared'] = df['TireDeg'] ** 2

    compound_code = pd.Categorical(df['Compound']).codes
    df['TyreLife_Compound'] = df['TyreLife'] * compound_code

    # ====================== OUTRAS FEATURES ======================
    df['LapNumber_Squared'] = df['LapNumber'] ** 2
    df['PositionChange'] = df.groupby('Driver')['Position'].diff().fillna(0)

    # ====================== ENCODING ======================
    df = pd.get_dummies(df, columns=['Compound'], prefix='Compound', dtype=int)

    return df


def prepare_model_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepara X e y SEM data leakage.
    """
    feature_cols = [
        'LapNumber',
        'LapProgress',
        'LapProgress_Squared',
        'LapNumber_Squared',
        'TyreLife',
        'TyreLifeNorm',
        'StintLap',
        'FreshTyre',
        'TireDeg',
        'TireDeg_Squared',
        'TyreLife_Compound',
        'PositionChange',
    ]

    compound_cols = [col for col in df.columns if col.startswith('Compound_')]
    feature_cols.extend(compound_cols)

    available_features = [col for col in feature_cols if col in df.columns]
    
    X = df[available_features].copy()
    y = df['LapTime'].copy()

    print(f"Features utilizadas ({len(available_features)}):")
    print(available_features)
    
    return X, y


# Teste rápido
if __name__ == "__main__":
    print("✅ features.py carregado com sucesso!")
    print("Funções: build_features() e prepare_model_data()")