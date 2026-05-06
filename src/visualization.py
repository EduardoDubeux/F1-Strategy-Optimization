import matplotlib.pyplot as plt

# =============================
# CORES OFICIAIS DOS PNEUS
# =============================

TIRE_COLORS = {
    "SOFT": "red",
    "MEDIUM": "yellow",
    "HARD": "black"
}

# =============================
# FORMATADOR DE TEMPO
# =============================

def format_time(seconds):
    seconds = int(seconds)

    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if h > 0:
        return f"{h}h {m}min {s}s"
    elif m > 0:
        return f"{m}min {s}s"
    else:
        return f"{s}s"

# =============================
# 🔥 1. GRÁFICO ESTILO F1 (STINTS)
# =============================

def plot_stint_strategies(results_df, top_n=3):
    """
    Gráfico estilo transmissão de F1 (linha de stints)
    """
    top = results_df.sort_values(by="total_time").head(top_n)

    plt.figure(figsize=(10, 4))

    for idx, row in enumerate(top.itertuples()):
        y = idx
        last_lap = 0

        for compound, lap in row.detail:
            plt.plot(
                [last_lap, lap],
                [y, y],
                linewidth=10,
                color=TIRE_COLORS.get(compound, "gray"),
                solid_capstyle='butt'
            )
            last_lap = lap

    plt.yticks(range(len(top)), [f"{i+1}º lugar" for i in range(len(top))])
    plt.xlabel("Voltas")
    plt.title("Top 3 Estratégias")

    plt.tight_layout()
    plt.show()


# =============================
# 🛞 2. DEGRADAÇÃO DE PNEUS (AJUSTADO)
# =============================

def plot_tire_degradation(df):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure()

    for compound in df['Compound'].unique():
        subset = df[df['Compound'] == compound]

        # agrupar por idade do pneu
        grouped = subset.groupby('TyreLife')['LapTime'].agg(['mean', 'std', 'count']).reset_index()

        # intervalo de confiança (95%)
        grouped['ci'] = 1.96 * (grouped['std'] / np.sqrt(grouped['count']))

        # suavização
        grouped['mean_smooth'] = grouped['mean'].rolling(window=3, min_periods=1).mean()

        # linha média
        plt.plot(
            grouped['TyreLife'],
            grouped['mean_smooth'],
            label=compound,
            color=TIRE_COLORS.get(compound, None),
            linewidth=2
        )

        # intervalo de confiança
        plt.fill_between(
            grouped['TyreLife'],
            grouped['mean'] - grouped['ci'],
            grouped['mean'] + grouped['ci'],
            color=TIRE_COLORS.get(compound, None),
            alpha=0.2
        )

    # =============================
    # FORMATAÇÃO DE TEMPO
    # =============================
    def format_time(seconds):
        if seconds >= 3600:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = seconds % 60
            return f"{h}h {m}m {s:.1f}s"
        elif seconds >= 60:
            m = int(seconds // 60)
            s = seconds % 60
            return f"{m}m {s:.1f}s"
        else:
            return f"{seconds:.2f}s"

    yticks = plt.yticks()[0]
    plt.yticks(yticks, [format_time(y) for y in yticks])

    plt.xlabel("Idade do Pneu (voltas)")
    plt.ylabel("Tempo de Volta")
    plt.title("Degradação de Pneus com Intervalo de Confiança")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

# =============================
# 🤖 3. COMPARAÇÃO DE MODELOS
# =============================

def plot_model_performance(results):
    names = list(results.keys())
    rmses = [results[name]['rmse'] for name in names]

    plt.figure()
    plt.bar(names, rmses)

    plt.ylabel("RMSE")
    plt.title("Comparação de Modelos")

    plt.tight_layout()
    plt.show()


# =============================
# 🏆 PRINT TOP 3
# =============================

def print_top_strategies(results_df, top_n=3):
    top = results_df.sort_values(by="total_time").head(top_n)

    print("\nTOP 3 ESTRATÉGIAS:\n")

    for i, row in enumerate(top.itertuples(), 1):
        print(f"{i}º lugar:")
        print(f"  Pit stops: {row.strategy}")
        print(f"  Tempo: {format_time(row.total_time)}")

        print("  Stints:")
        for comp, lap in row.detail:
            print(f"    → {comp} até a volta {lap}")

        print()

def print_f1_style_output(driver_name, team, race, year, grid_position, best_strategy, results_df):
    
    def format_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60

        if h > 0:
            return f"{h}h {m}m {s:.1f}s"
        elif m > 0:
            return f"{m}m {s:.1f}s"
        else:
            return f"{s:.2f}s"

    print("\n==========================================")
    print("🏁 FORMULA 1 STRATEGY ANALYSIS")
    print("==========================================\n")

    print(f"Piloto: {driver_name} ({team})")
    print(f"Corrida: {race} {year}")
    print(f"Grid: P{grid_position}\n")

    # =============================
    # MELHOR ESTRATÉGIA
    # =============================
    print("------------------------------------------")
    print("🏆 MELHOR ESTRATÉGIA")
    print("------------------------------------------")

    print("Stints:")

    prev_lap = 1
    for compound, lap in best_strategy['detail']:
        print(f"{compound:<6} | L{prev_lap} → L{lap}")
        prev_lap = lap + 1

    print(f"\nTempo Total: {format_time(best_strategy['total_time'])}\n")

    # =============================
    # TOP 3
    # =============================
    print("------------------------------------------")
    print("📊 TOP 3 ESTRATÉGIAS")
    print("------------------------------------------\n")

    top3 = results_df.sort_values(by="total_time").head(3)

    for i, row in enumerate(top3.itertuples(), 1):
        pits = ", ".join([f"L{p}" for p in row.strategy])
        compounds = " → ".join([c for c, _ in row.detail])

        print(f"#{i} → {format_time(row.total_time)}")
        print(compounds)
        print(f"Pits: {pits}\n")

    print("==========================================\n")

# =============================
# 🚀 FUNÇÃO PRINCIPAL DE PLOT
# =============================

def plot_all(results_df, best_strategy, model_results, df):
    print("\nGerando gráficos...")

    plot_stint_strategies(results_df)
    plot_model_performance(model_results)
    plot_tire_degradation(df)