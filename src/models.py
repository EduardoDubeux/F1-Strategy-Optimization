import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# =============================
# DIVISÃO DOS DADOS
# =============================

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Divide os dados em treino e teste
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# =============================
# CRIAÇÃO DOS MODELOS
# =============================

def get_models():
    """
    Retorna os modelos definidos no TCC
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
    }

    return models


# =============================
# TREINAMENTO E AVALIAÇÃO
# =============================

def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    """
    Treina todos os modelos e calcula RMSE
    """
    results = {}

    for name, model in models.items():
        print(f"\nTreinando: {name}")

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        results[name] = {
            "model": model,
            "rmse": rmse
        }

        print(f"{name} RMSE: {rmse:.4f}")

    return results


# =============================
# SELEÇÃO DO MELHOR MODELO
# =============================

def select_best_model(results):
    """
    Seleciona o modelo com menor RMSE
    """
    best_name = min(results, key=lambda x: results[x]["rmse"])
    best_model = results[best_name]["model"]

    print(f"\nMelhor modelo: {best_name}")
    print(f"RMSE: {results[best_name]['rmse']:.4f}")

    return best_name, best_model


# =============================
# FUNÇÃO PRINCIPAL
# =============================

def run_models(X, y):
    """
    Pipeline completo:
    - split
    - treino
    - avaliação
    - seleção
    """
    X_train, X_test, y_train, y_test = split_data(X, y)

    models = get_models()

    results = train_and_evaluate(models, X_train, X_test, y_train, y_test)

    best_name, best_model = select_best_model(results)

    return best_name, best_model, results