# 📘 AF3 – Predicción de Calificaciones de Examen con Regresión Lineal

Este repositorio contiene el desarrollo completo del **Producto Integrador de Aprendizaje (AF3)** de la materia **Programación para Inteligencia Artificial**.  
El proyecto implementa un modelo de **regresión lineal múltiple** para predecir la calificación final de un examen utilizando un dataset real de Kaggle.

---

# 🎯 Objetivo del Proyecto

Aplicar el flujo completo de **aprendizaje supervisado** para construir un modelo predictivo capaz de estimar calificaciones. Este proyecto demuestra habilidades en:

- Análisis del problema.
- Preprocesamiento y normalización.
- Entrenamiento de un modelo supervisado.
- Evaluación con métricas estándar.
- Visualización de resultados.
- Documentación técnica.

---

# 📊 Dataset Utilizado

- **Nombre:** Exam Score Prediction Dataset  
- **Fuente:** Kaggle  
- **URL:** https://www.kaggle.com/datasets/kundanbedmutha/exam-score-prediction-dataset  
- **Variable objetivo:** `Exam_Score`  
- **Tipo de problema:** Regresión (valor continuo)


---

# 🧪 Flujo de Trabajo del Proyecto

## 1️⃣ Selección del Caso de Estudio
- Dataset real y público.
- Más de 200 registros y múltiples variables.
- Adecuado para regresión lineal.

## 2️⃣ Preprocesamiento
Incluye:
- Eliminación de valores nulos.
- Imputación de valores faltantes numéricos.
- Codificación one-hot de variables categóricas.
- Normalización con StandardScaler.
- Revisión del DataFrame antes y después.

## 3️⃣ Implementación del Modelo
- Uso de `LinearRegression()`.
- División 70% train – 30% test.
- Normalización estándar.
- Entrenamiento supervisado.

## 4️⃣ Evaluación del Modelo
Métricas implementadas:

- **MSE** – Error cuadrático medio  
- **MAE** – Error absoluto medio  
- **R²** – Coeficiente de determinación  




