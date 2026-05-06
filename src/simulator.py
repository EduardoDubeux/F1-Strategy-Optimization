import pandas as pd
import random
from src.features import compute_tire_degradation


def get_compound_features(compound: str):
    compounds = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
    return {f"Compound_{c}": 1 if c == compound else 0 for c in compounds}


def calibrate_lap_time(base_time: float, lap: int, total_laps: int, compound: str) -> float:
    """Calibração específica para reduzir viés do modelo"""
    correction = -2.45  # viés médio atual
    
    # Pista melhora um pouco no final
    if lap > total_laps * 0.75:
        correction -= 0.4
    
    # Bônus leve para Verstappen (pole position)
    if lap < 15:
        correction -= 0.3
        
    return base_time + correction


def simulate_strategy(model, feature_columns, total_laps, pit_laps, grid_position):
    total_time = 0.0
    tyre_life = 0
    current_compound = "SOFT"
    strategy_detail = []

    PIT_LOSS = 24.0

    compound_plan = ["SOFT", "MEDIUM", "HARD"]
    compound_idx = 0

    for lap in range(1, total_laps + 1):
        if lap in pit_laps:
            total_time += PIT_LOSS
            strategy_detail.append((current_compound, lap))
            tyre_life = 0
            compound_idx = min(compound_idx + 1, len(compound_plan) - 1)
            current_compound = compound_plan[compound_idx]

        tyre_life += 1

        # Features
        lap_progress = lap / total_laps
        row = {
            'LapNumber': lap,
            'LapProgress': lap_progress,
            'LapProgress_Squared': lap_progress ** 2,
            'LapNumber_Squared': lap ** 2,
            'TyreLife': tyre_life,
            'TyreLifeNorm': tyre_life / 40,
            'StintLap': tyre_life,
            'FreshTyre': 1 if tyre_life == 1 else 0,
            'PositionChange': 0,
        }

        # Degradação
        tire_deg = compute_tire_degradation(tyre_life, current_compound)

        row['TireDeg'] = tire_deg
        row['TireDeg_Squared'] = tire_deg ** 2
        row['TyreLife_Compound'] = tyre_life * compound_plan.index(current_compound)
        row.update(get_compound_features(current_compound))

        row_df = pd.DataFrame([row])
        for col in feature_columns:
            if col not in row_df.columns:
                row_df[col] = 0
        row_df = row_df[feature_columns]

        lap_time = model.predict(row_df)[0]
        
        # Calibração
        lap_time = calibrate_lap_time(lap_time, lap, total_laps, current_compound)

        # Penalidades fortes
        if current_compound == "SOFT" and tyre_life > 16:
            lap_time += 2.5 * (tyre_life - 16)
        elif current_compound == "MEDIUM" and tyre_life > 24:
            lap_time += 1.8 * (tyre_life - 24)
        elif current_compound == "HARD" and tyre_life > 33:
            lap_time += 1.1 * (tyre_life - 33)

        total_time += lap_time

    strategy_detail.append((current_compound, total_laps))
    return total_time, strategy_detail


# Geração de estratégias (prioriza 2 paradas)
def generate_strategies(total_laps=57, n_samples=120):
    strategies = [
        (12, 29), (13, 30), (14, 31), (15, 32), (16, 33), (10, 28),
        (18, 35), (9, 27), (11, 32)
    ]
    # Adiciona aleatórias
    for _ in range(n_samples - len(strategies)):
        pits = sorted(random.sample(range(8, total_laps-8), random.randint(1, 2)))
        strategies.append(tuple(pits))
    
    return strategies


def run_simulation(model, feature_columns, grid_position, total_laps=57):
    strategies = generate_strategies(total_laps)
    print(f"Testando {len(strategies)} estratégias...")

    results = []
    for pits in strategies:
        total_time, detail = simulate_strategy(model, feature_columns, total_laps, pits, grid_position)
        results.append({"strategy": pits, "total_time": total_time, "detail": detail})

    df = pd.DataFrame(results)
    best = df.sort_values(by="total_time").iloc[0]

    return df, best