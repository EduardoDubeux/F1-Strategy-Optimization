import os
import pandas as pd

from src.feature_engineering import create_features
from src.strategy_model import train_model
from src.strategy_simulator import find_best_strategy
from src.visualization import plot_strategy_results, plot_tire_degradation

# IMPORTANTE: importar função de dataset
from data_loader import build_dataset  # ou ajuste conforme seu arquivo

DATA_PATH = 'data/processed/lap_data.csv'

# =========================
# GARANTIR QUE DATASET EXISTE
# =========================

if not os.path.exists(DATA_PATH):
    print("Dataset não encontrado. Gerando agora...\n")
    dataset = build_dataset()
    os.makedirs('data/processed', exist_ok=True)
    dataset.to_csv(DATA_PATH, index=False)
    print("Dataset criado com sucesso!\n")

# =========================
# CARREGAR DATASET
# =========================

df = pd.read_csv(DATA_PATH)

# =========================
# FEATURE ENGINEERING
# =========================

df, feature_cols, target_col = create_features(df)

# =========================
# TREINAR MODELO
# =========================

model = train_model(df, feature_cols, target_col)

# =========================
# SIMULAR ESTRATÉGIAS
# =========================

best, results = find_best_strategy(model, feature_cols, total_laps=70)

print("\n===== RESULTADOS =====")

for strat, time in results:
    print(f"Estratégia: {strat} -> Tempo: {time:.2f}s")

print("\n🏆 Melhor estratégia:")
print(best)

# =========================
# GRÁFICOS
# =========================

plot_strategy_results(results)
plot_tire_degradation(df)