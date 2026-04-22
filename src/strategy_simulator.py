import pandas as pd

PIT_STOP_TIME = 22  # segundos médios

def simulate_strategy(model, feature_cols, total_laps, strategy):
    """
    strategy = lista de stints
    Ex:
    [
        ("SOFT", 20),
        ("MEDIUM", 30)
    ]
    """

    total_time = 0
    current_lap = 1

    for compound, stint_length in strategy:
        tyre_age = 0

        for _ in range(stint_length):
            row = {
                'LapNumber': current_lap,
                'TyreLife': tyre_age,
                'RaceProgress': current_lap / total_laps,
                'Compound_SOFT': 1 if compound == 'SOFT' else 0,
                'Compound_MEDIUM': 1 if compound == 'MEDIUM' else 0,
                'Compound_HARD': 1 if compound == 'HARD' else 0
            }

            X = pd.DataFrame([row])[feature_cols]
            lap_time = model.predict(X)[0]

            total_time += lap_time

            tyre_age += 1
            current_lap += 1

        # adicionar pit stop (exceto último stint)
        if current_lap <= total_laps:
            total_time += PIT_STOP_TIME

    return total_time


def find_best_strategy(model, feature_cols, total_laps):
    strategies = [
        [("SOFT", 20), ("MEDIUM", total_laps - 20)],
        [("MEDIUM", 25), ("HARD", total_laps - 25)],
        [("SOFT", 15), ("MEDIUM", 20), ("HARD", total_laps - 35)],
    ]

    results = []

    for strat in strategies:
        time = simulate_strategy(model, feature_cols, total_laps, strat)
        results.append((strat, time))

    best = min(results, key=lambda x: x[1])

    return best, results