import fastf1
import pandas as pd
import os

# =============================
# CONFIGURAÇÃO DE CACHE
# =============================

def enable_cache(cache_dir="data/cache"):
    """
    Ativa o cache do FastF1 para evitar downloads repetidos.
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    fastf1.Cache.enable_cache(cache_dir)


# =============================
# CARREGAR UMA SESSÃO
# =============================

def load_session(year, race, session_type='R'):
    """
    Carrega uma sessão de F1.
    """
    session = fastf1.get_session(year, race, session_type)
    session.load()
    return session


# =============================
# EXTRAIR DADOS DE VOLTAS
# =============================

def extract_lap_data(session, driver=None):
    """
    Extrai dados relevantes das voltas.
    """
    laps = session.laps

    df = laps[[
        'Driver',
        'LapNumber',
        'LapTime',
        'Compound',
        'TyreLife',
        'Stint',
        'Position'
    ]].copy()

    # =============================
    # (OPCIONAL) FILTRO POR PILOTO
    # =============================
    if driver is not None:
        df = df[df['Driver'] == driver].copy()

    # =============================
    # MAPA DE PILOTOS (ROBUSTO)
    # =============================

    results = session.results
    driver_map = {}

    for _, row in results.iterrows():
        code = row['Abbreviation']
        full_name = f"{row['FirstName']} {row['LastName']}"
        team = row['TeamName']

        driver_map[code] = {
            "FullName": full_name,
            "Team": team
        }

    df.loc[:, 'FullName'] = df['Driver'].map(
        lambda d: driver_map.get(d, {}).get("FullName", d)
    )

    df.loc[:, 'Team'] = df['Driver'].map(
        lambda d: driver_map.get(d, {}).get("Team", "Unknown")
    )

    # =============================
    # LIMPEZA
    # =============================

    df = df.dropna().copy()

    # Converter tempo (SEM WARNING)
    df.loc[:, 'LapTime'] = df['LapTime'].dt.total_seconds()

    # =============================
    # FILTRO DE VOLTAS REALISTAS
    # =============================

    df = df[df['LapTime'] < df['LapTime'].quantile(0.95)].copy()
    df = df[df['LapTime'] > df['LapTime'].quantile(0.05)].copy()
    df = df[(df['LapTime'] > 60) & (df['LapTime'] < 200)].copy()

    return df


# =============================
# PROCESSAMENTO BÁSICO
# =============================

def preprocess_data(df):
    """
    Faz tratamento básico
    """
    df = df.copy()
    return df


# =============================
# PIPELINE COMPLETO
# =============================

def load_and_process(year, race, session_type='R', driver=None, save_csv=True):
    """
    Pipeline completo
    """
    enable_cache()

    session = load_session(year, race, session_type)
    df = extract_lap_data(session, driver=driver)
    df = preprocess_data(df)

    # Salvar dataset
    if save_csv:
        path = f"data/processed/{year}_{race}_{session_type}.csv"
        os.makedirs("data/processed", exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Dataset salvo em: {path}")

    return df


# =============================
# CARREGAR MÚLTIPLAS CORRIDAS
# =============================

def load_multiple_sessions(sessions_list):
    """
    Carrega múltiplas corridas
    """
    all_data = []

    for year, race in sessions_list:
        print(f"Carregando: {year} - {race}")
        df = load_and_process(year, race, save_csv=False)
        all_data.append(df)

    final_df = pd.concat(all_data, ignore_index=True)

    os.makedirs("data/processed", exist_ok=True)
    final_df.to_csv("data/processed/full_dataset.csv", index=False)

    print("Dataset completo salvo em data/processed/full_dataset.csv")

    return final_df