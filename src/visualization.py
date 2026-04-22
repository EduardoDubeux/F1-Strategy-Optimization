import matplotlib.pyplot as plt

def plot_strategy_results(results):
    labels = []
    times = []

    for strat, time in results:
        label = " | ".join([f"{c}-{l}" for c, l in strat])
        labels.append(label)
        times.append(time)

    plt.figure()
    plt.bar(labels, times)

    plt.xticks(rotation=45)
    plt.xlabel("Estratégias")
    plt.ylabel("Tempo total (s)")
    plt.title("Comparação de Estratégias de Pit Stop")

    plt.tight_layout()
    plt.show()


def plot_tire_degradation(df):
    plt.figure()

    for compound in df['Compound'].unique():
        subset = df[df['Compound'] == compound]

        plt.scatter(
            subset['TyreLife'],
            subset['LapTimeSeconds'],
            label=compound,
            alpha=0.5
        )

    plt.xlabel("Idade do Pneu (voltas)")
    plt.ylabel("Tempo de volta (s)")
    plt.title("Degradação de Pneus")
    plt.legend()

    plt.tight_layout()
    plt.show()