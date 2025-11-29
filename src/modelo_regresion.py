"""
modelo_regresion.py
AF3 – Predicción de calificaciones de examen con regresión lineal
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
    print("Primeras filas del dataset:\n", df.head())
    print("\nInformación del dataset:\n")
    print(df.info())
    return df


def preprocesar_datos(df: pd.DataFrame):
    """
    Realiza preprocesamiento básico:
    - Elimina filas completamente vacías.
    - Imputa valores nulos numéricos con la media.
    - Codifica variables categóricas con one-hot encoding.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    X : pd.DataFrame
        Variables predictoras.
    y : pd.Series
        Variable objetivo (última columna del DataFrame).
    """
    # Eliminar filas que estén completamente vacías
    df = df.dropna(how="all")

    # Imputar nulos numéricos con la media
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].mean(), inplace=True)

    # Suponemos que la última columna es la calificación de examen
    y = df.iloc[:, -1]

    # Variables predictoras: todas menos la última
    X = df.iloc[:, :-1]

    # Codificar columnas categóricas (object) con one-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    return X, y


def entrenar_y_evaluar_modelo(X, y):
    """
    Divide los datos en train/test, normaliza, entrena una regresión lineal
    y evalúa el modelo.

    Returns
    -------
    modelo : LinearRegression
    X_test_scaled : ndarray
    y_test : pd.Series
    y_pred : ndarray
    """
    # División 70 % entrenamiento, 30 % prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
    )

    # Normalización de características numéricas
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Definición y entrenamiento del modelo
    modelo = LinearRegression()
    modelo.fit(X_train_scaled, y_train)

    # Predicciones
    y_pred = modelo.predict(X_test_scaled)

    # Métricas de evaluación
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n=== Métricas del modelo de regresión lineal ===")
    print(f"Error cuadrático medio (MSE): {mse:.3f}")
    print(f"Error absoluto medio (MAE):  {mae:.3f}")
    print(f"Coeficiente de determinación (R^2): {r2:.3f}")

    return modelo, X_test_scaled, y_test, y_pred


def graficar_resultados(y_test, y_pred, ruta_salida: Path):
    """
    Genera una gráfica de valores reales vs. predichos y la guarda en disco.

    Parameters
    ----------
    y_test : pd.Series
        Valores reales.
    y_pred : ndarray
        Valores predichos por el modelo.
    ruta_salida : Path
        Ruta del archivo de imagen a guardar.
    """
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel("Calificación real")
    plt.ylabel("Calificación predicha")
    plt.title("Rendimiento del modelo de regresión lineal")

    # Línea y = x como referencia
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val])

    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=300)
    plt.close()

    print(f"\nGráfica guardada en: {ruta_salida}")


def main():
    # Ruta al CSV (ajústala si tu archivo se llama distinto)
    ruta_csv = Path("data") / "exam_score_prediction.csv"

    # 1. Cargar datos
    df = cargar_datos(ruta_csv)

    # 2. Preprocesar (nulos, categóricas, separar X e y)
    X, y = preprocesar_datos(df)

    # 3. Entrenar y evaluar modelo
    modelo, X_test_scaled, y_test, y_pred = entrenar_y_evaluar_modelo(X, y)

    # 4. Graficar resultados
    ruta_imagen = Path("imagenes") / "exam_scatter.png"
    graficar_resultados(y_test, y_pred, ruta_imagen)


if __name__ == "__main__":
    main()
