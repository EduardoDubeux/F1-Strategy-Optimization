import pandas as pd
import random


# =============================
# 🚗 TRÁFEGO
# =============================

def traffic_penalty(lap, grid_position):
    if grid_position > 10:
        base = 0.6
    else:
        base = 0.3

    if lap < 10:
        chance = 0.5
    elif lap < 30:
        chance = 0.3
    else:
        chance = 0.2

    if random.random() < chance:
        return random.uniform(base, base + 1.5)

    return 0


# =============================
# 🛞 STINT PENALTY
# =============================

def stint_penalty(compound, tyre_life):
    penalty = 0

    if compound == "SOFT" and tyre_life > 15:
        penalty += 2.5 * (tyre_life - 15)

    if compound == "MEDIUM" and tyre_life > 20:
        penalty += 1.5 * (tyre_life - 20)

    if compound == "HARD" and tyre_life > 30:
        penalty += 1.0 * (tyre_life - 30)

    return penalty


# =============================
# SIMULAÇÃO
# =============================

def simulate_strategy(model, feature_columns, total_laps, pit_laps, grid_position, compound_plan=None):
    total_time = 0
    tyre_life = 0
    stint = 1

    PIT_LOSS = 20

    if compound_plan is None:
        compound_plan = ["SOFT", "MEDIUM", "HARD"]

    compound_idx = 0
    strategy_detail = []

    for lap in range(1, total_laps + 1):

        if lap in pit_laps:
            total_time += PIT_LOSS
            strategy_detail.append((compound_plan[compound_idx], lap))
            tyre_life = 0
            stint += 1
            compound_idx = min(compound_idx + 1, len(compound_plan) - 1)

        tyre_life += 1
        compound = compound_plan[compound_idx]

        row = {
            'LapNumber': lap,
            'TyreLife': tyre_life,
            'Stint': stint,
            'StintLap': tyre_life,
            'TireDeg': 0,
            'GridPosition': grid_position,
            'DriverPace': 0
        }

        for col in feature_columns:
            if col not in row:
                row[col] = 0

        compound_col = f"Compound_{compound}"
        if compound_col in row:
            row[compound_col] = 1

        row_df = pd.DataFrame([row])[feature_columns]

        lap_time = model.predict(row_df)[0]

        lap_time += stint_penalty(compound, tyre_life)
        lap_time += traffic_penalty(lap, grid_position)

        total_time += lap_time

    strategy_detail.append((compound_plan[compound_idx], total_laps))

    return total_time, strategy_detail


# =============================
# ESTRATÉGIAS
# =============================

def generate_strategies(total_laps, max_pits=3, n_samples=120):
    strategies = []

    for _ in range(n_samples):
        n_pits = random.randint(1, max_pits)
        pits = sorted(random.sample(range(10, total_laps - 10), n_pits))
        strategies.append(tuple(pits))

    return strategies


# =============================
# RUN
# =============================

def run_simulation(model, feature_columns, grid_position, total_laps=57):
    strategies = generate_strategies(total_laps)

    print(f"Testando {len(strategies)} estratégias...")

    results = []

    for pits in strategies:
        total_time, detail = simulate_strategy(
            model,
            feature_columns,
            total_laps,
            pits,
            grid_position
        )

        results.append({
            "strategy": pits,
            "total_time": total_time,
            "detail": detail
        })

    df = pd.DataFrame(results)
    best = df.sort_values(by="total_time").iloc[0]

    return df, best