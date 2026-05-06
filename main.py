from src.data_loader import load_and_process
from src.features import build_features, prepare_model_data
from src.models import run_models
from src.simulator import run_simulation
from src.visualization import plot_all, print_f1_style_output


def main():
    # =============================
    # CONFIGURAÇÕES
    # =============================
    YEAR = 2023
    RACE = 'Bahrain'
    SESSION = 'R'
    DRIVER = "VER"  # ex: VER, HAM, ALO

    print("\nIniciando pipeline...\n")

    # =============================
    # 1. CARREGAR DADOS
    # =============================
    df_raw = load_and_process(YEAR, RACE, SESSION)

    # filtrar piloto
    df_raw = df_raw[df_raw['Driver'] == DRIVER].copy()

    if df_raw.empty:
        print(f"Erro: piloto {DRIVER} não encontrado nos dados.")
        return

    # nome completo e equipe
    driver_name = df_raw['FullName'].iloc[0]
    team = df_raw['Team'].iloc[0]

    # posição de largada
    grid_position = int(df_raw['Position'].iloc[0])

    print(f"Analisando piloto: {driver_name} ({team})\n")

    # =============================
    # 2. FEATURE ENGINEERING
    # =============================
    df = build_features(df_raw.copy())

    # =============================
    # 3. MODELAGEM
    # =============================
    X, y = prepare_model_data(df)

    best_name, best_model, results = run_models(X, y)

    print(f"Melhor modelo selecionado: {best_name}\n")

    # =============================
    # 4. SIMULAÇÃO
    # =============================
    results_df, best_strategy = run_simulation(
        best_model,
        X.columns,
        grid_position,
        total_laps=57
    )

    # =============================
    # 5. OUTPUT ESTILO F1
    # =============================
    print_f1_style_output(
        driver_name,
        team,
        RACE,
        YEAR,
        grid_position,
        best_strategy,
        results_df
    )

    # =============================
    # 6. GRÁFICOS
    # =============================
    plot_all(
        results_df,
        best_strategy,
        results,
        df_raw
    )


if __name__ == "__main__":
    main()