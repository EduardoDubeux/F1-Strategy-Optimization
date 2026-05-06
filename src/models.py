import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
from typing import Dict, Tuple, Any


def train_xgboost(X: pd.DataFrame, y: pd.Series) -> xgb.XGBRegressor:
    print("🔄 Treinando XGBoost (versão balanceada)...")

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.08,
        max_depth=5,              # reduzido para evitar overfitting
        subsample=0.85,
        colsample_bytree=0.75,
        min_child_weight=4,
        gamma=0.2,
        reg_alpha=0.5,
        reg_lambda=1.5,
        random_state=42,
        objective='reg:squarederror',
        eval_metric='rmse'
    )

    model.fit(X, y)

    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)

    print(f"✅ XGBoost Treinado!")
    print(f"   RMSE  (treino): {rmse:.4f}s")
    print(f"   MAE   (treino): {mae:.4f}s")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/best_xgboost.pkl")
    print(f"💾 Modelo salvo!")

    return model


def run_models(X: pd.DataFrame, y: pd.Series) -> Tuple[str, Any, Dict]:
    """
    Função mantida para compatibilidade com seu main.py atual.
    Retorna: (best_name, best_model, results)
    """
    print("\n" + "="*50)
    print("INICIANDO TREINAMENTO DOS MODELOS")
    print("="*50)

    best_model = train_xgboost(X, y)
    best_name = "XGBoost"

    # Resultados básicos (para compatibilidade com visualization)
    results = {
        "XGBoost": {
            "rmse": 0.0,   # podemos melhorar depois
            "mae": 0.0,
            "r2": 0.0
        }
    }

    return best_name, best_model, results


def load_model(model_path: str = "models/best_xgboost.pkl"):
    """Carrega modelo salvo"""
    return joblib.load(model_path)


def predict_lap_time(model, X: pd.DataFrame) -> np.ndarray:
    """Faz predição"""
    return model.predict(X)