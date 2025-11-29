"""
AF3 – Predicción de calificaciones de examen con regresión lineal

Este script:
1. Carga el dataset de Kaggle.
2. Preprocesa los datos (nulos, categóricas).
3. Divide en train/test (70/30).
4. Normaliza con StandardScaler.
5. Entrena un modelo de regresión lineal.
6. Calcula MSE, MAE y R².
7. Genera una gráfica de valores reales vs. predichos.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


# ---------------------------------------------------------------------
# 1. Carga del dataset
# ---------------------------------------------------------------------
def cargar_datos(ruta_csv: Path) -> pd.DataFrame:
    """
    Carga el dataset desde un archivo CSV.

    Parameters
    ----------
    ruta_csv : Path
        Ruta al archivo CSV.

    Returns
    -------
    df : pd.DataFrame
        DataFrame con los datos cargados.
    """
    if not ruta_csv.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_csv}")

    df = pd.read_csv(ruta_csv)

    print("Primeras filas del dataset:")
    print(df.head())
    print("\nInformación del dataset:")
    print(df.info())

    return df


# ---------------------------------------------------------------------
# 2. Preprocesamiento
# ---------------------------------------------------------------------
def preprocesar_datos(df: pd.DataFrame):
    """
    Realiza preprocesamiento básico:
    - Elimina filas completamente vacías.
    - Imputa valores nulos numéricos con la media.
    - Codifica variables categóricas con one-hot encoding.
    - Separa X (predictoras) e y (objetivo).

    Returns
    -------
    X : pd.DataFrame
        Variables predictoras.
    y : pd.Series
        Variable objetivo (última columna del DataFrame).
    """
    # Eliminar filas totalmente vacías
    df = df.dropna(how="all")

    # Imputar nulos numéricos con la media
    columnas_numericas = df.select_dtypes(include=["int64", "float64"]).columns
    for col in columnas_numericas:
        if df[col].isnull().any():
            df[col].fillna(df[col].mean(), inplace=True)

    # Variable objetivo: última columna
    y = df.iloc[:, -1]

    # Variables predictoras: todas menos la última
    X = df.iloc[:, :-1]

    # Codificar categóricas
    X = pd.get_dummies(X, drop_first=True)

    print("\nShape de X después de get_dummies:", X.shape)
    print("Shape de y:", y.shape)

    return X, y


# ---------------------------------------------------------------------
# 3. Entrenamiento y evaluación
# ---------------------------------------------------------------------
def entrenar_y_evaluar_modelo(X, y):
    """
    Divide los datos, normaliza, entrena y evalúa el modelo.

    Returns
    -------
    modelo : LinearRegression
    X_test_scaled : ndarray
    y_test : pd.Series
    y_pred : ndarray
    """
    # División 70% / 30%
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    # Normalización
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modelo
    modelo = LinearRegression()
    modelo.fit(X_train_scaled, y_train)

    # Predicciones
    y_pred = modelo.predict(X_test_scaled)

    # Métricas
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n=== Métricas del modelo de regresión lineal ===")
    print(f"MSE (Error cuadrático medio): {mse:.3f}")
    print(f"MAE (Error absoluto medio):  {mae:.3f}")
    print(f"R^2 (Coeficiente de determinación): {r2:.3f}")

    return modelo, X_test_scaled, y_test, y_pred


# ---------------------------------------------------------------------
# 4. Gráfica valores reales vs predichos
# ---------------------------------------------------------------------
def graficar_resultados(y_test, y_pred, ruta_salida: Path):
    """
    Genera y guarda la gráfica de valores reales vs predichos.

    Parameters
    ----------
    y_test : pd.Series
        Valores reales.
    y_pred : ndarray
        Valores predichos.
    ruta_salida : Path
        Ruta del archivo .png a guardar.
    """
    # Crear carpeta si no existe
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel("Calificación real")
    plt.ylabel("Calificación predicha")
    plt.title("Regresión lineal – Valores reales vs predichos")

    # Línea y = x
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r")

    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=300)
    plt.show()

    print(f"\nGráfica guardada en: {ruta_salida}")


# ---------------------------------------------------------------------
# 5. Función principal
# ---------------------------------------------------------------------
def main():
    # Ruta al CSV (ajusta el nombre si tu archivo se llama distinto)
    ruta_csv = Path("data") / "exam_score_prediction.csv"

    # 1. Cargar datos
    df = cargar_datos(ruta_csv)

    # 2. Preprocesar
    X, y = preprocesar_datos(df)

    # 3. Entrenar y evaluar
    modelo, X_test_scaled, y_test, y_pred = entrenar_y_evaluar_modelo(X, y)

    # 4. Graficar
    ruta_imagen = Path("imagenes") / "exam_scatter.png"
    graficar_resultados(y_test, y_pred, ruta_imagen)


if __name__ == "__main__":
    main()
