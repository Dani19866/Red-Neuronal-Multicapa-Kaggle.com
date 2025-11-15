# 💧 Clasificador de Potabilidad de Agua con Red Neuronal (MLP)

Proyecto de Computación Emergente (FPTSP25) para la Universidad Metropolitana.

* **Estudiantes:** Eduardo Curiel, Daniel De Oliveira, Vincent Perez
* **Notebook de Kaggle:** 💻 [Notebook en Kaggle](https://www.kaggle.com/code/danieldeoliveira00/vincent-eduardo-daniel-proyecto-c-emergente)
* **Informe:** 📄 `[No listo]`

---

## 🎯 1. Objetivos del Proyecto

### Objetivo General
Diseñar, implementar y evaluar una red neuronal multicapa (MLP) para la clasificación de potabilidad del agua.
* **Meta de Rendimiento:** Superar el **70%** de exactitud (accuracy) en el conjunto de prueba.
* **Resultado Final:** **67% de exactitud.**

### Objetivos Específicos
1.  **Analizar:** Realizar un análisis exploratorio completo del dataset (distribuciones, correlaciones, etc.).
2.  **Preprocesar:** Implementar un pipeline robusto que incluya manejo de nulos, normalización y balanceo de clases (SMOTE).
3.  **Identificar:** Usar métodos basados en árboles (Random Forest) para identificar las características fisicoquímicas más relevantes.
4.  **Diseñar:** Diseñar una arquitectura de MLP óptima para el problema.

## 🚱 2. El Problema y Justificación

La contaminación del agua es una crisis de salud pública, especialmente en Venezuela (ej. Lago de Maracaibo y Lago de Valencia). Los análisis tradicionales son lentos y costosos. Este proyecto explora el **Machine Learning** como una alternativa rápida y de bajo costo para la monitorización automatizada.

---

## 🛠️ 3. Pipeline de Datos y Metodología

### 📊 3.1. Análisis Exploratorio (EDA)

El análisis inicial reveló 3 desafíos clave:
1.  **Valores Nulos:** Datos faltantes en `ph`, `Sulfate` y `Trihalomethanes`.
2.  **Desbalance de Clases:** El dataset estaba desbalanceado (61% No Potable vs 39% Potable).
3.  **Escalas de Datos:** Características con escalas, medias y varianzas muy diferentes.

### ⚙️ 3.2. Pipeline de Preprocesamiento

Se implementó un pipeline riguroso para preparar los datos:

1.  **División de Datos:** `train_test_split` (70/30) con `stratify=y` para mantener la proporción de clases en ambos conjuntos.
2.  **Imputación de Nulos:** Se usó `KNNImputer` (con `n_neighbors=30`) para estimar valores faltantes basándose en sus "vecinos" más cercanos. Se ajustó solo en *train* para evitar *data leakage*.
3.  **Balanceo de Clases:** Se aplicó `SMOTE` (con `sampling_strategy=0.75`) **solo al set de entrenamiento** para crear muestras sintéticas de la clase minoritaria ("Potable") y balancear el modelo.
4.  **Normalización:** Se usó `StandardScaler` para reescalar todas las características (media 0, desviación 1), un paso crucial para el rendimiento de las redes neuronales.

### 🔍 3.3. Selección de Características

Se entrenó un `RandomForestClassifier` solo para evaluar la importancia de las características, como se planteó en la metodología. El resultado mostró que **todas las 9 características eran relevantes**, por lo que se usaron todas en la MLP.

### 🧠 3.4. Arquitectura de la Red Neuronal (MLP)

Se diseñó un modelo `Sequential` en Keras con una fuerte estrategia de regularización para combatir el sobreajuste:

* **Entrada:** `Input(shape=(9,))`
* **Capa Oculta 1:** `Dense(128, 'relu')` + `L2(0.001)` + `BatchNormalization` + `Dropout(0.2)`
* **Capa Oculta 2:** `Dense(64, 'relu')` + `L2(0.001)` + `BatchNormalization` + `Dropout(0.2)`
* **Capa Oculta 3:** `Dense(32, 'relu')` + `L2(0.001)` + `BatchNormalization` + `Dropout(0.2)`
* **Salida:** `Dense(1, 'sigmoid')` (para clasificación binaria).

**Compilación y Entrenamiento:**
* **Optimizador:** `Nadam` (Learning Rate = 0.0005)
* **Pérdida:** `binary_crossentropy`
* **Callbacks:**
    * `EarlyStopping` (monitoreando `val_auc`, `patience=25`)
    * `ReduceLROnPlateau` (monitoreando `val_loss`, `patience=8`)
* **Entrenamiento:** 150 épocas con `batch_size=64`.

---

## 📉 4. Resultados y Conclusiones

* **Objetivo de Precisión:** > 70%
* **Exactitud Final (Accuracy):** **67%**

El objetivo de rendimiento principal **no se cumplió** por un margen del **3%**.

### 4.1. Reporte de Clasificación

#### REPORTE DE CLASIFICACION

| Clase       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| No Potable | 0.71      | 0.82   | 0.76     | 605     |
| Potable    | 0.58      | 0.44   | 0.50     | 378     |

- **Accuracy**: 0.67 (983 muestras)
- **Macro Avg**: 0.64 precision, 0.63 recall, 0.63 f1-score
- **Weighted Avg**: 0.66 precision, 0.67 recall, 0.66 f1-score

**Análisis**: El modelo es mucho mejor identificando agua "No Potable" (Recall de 0.82) que agua "Potable" (Recall de 0.44).

### 4.2. Matriz de Confusión

| (n=983) | Predicción: No Potable (0) | Predicción: Potable (1) |
| :--- | :---: | :---: |
| **Real: No Potable (0)** | **TN = 496** | **FP = 109** |
| **Real: Potable (1)** | **FN = 214** | **TP = 164** |

**Análisis de Errores Críticos:**
* **🔴 Falsos Positivos (FP): 109** - ¡El error más peligroso! 109 muestras **no potables** se clasificaron erróneamente como **potables**.
* **🟡 Falsos Negativos (FN): 214** - 214 muestras **potables** se clasificaron como **no potables**.

### 4.3. Conclusión General

El modelo MLP, a pesar del robusto preprocesamiento y la arquitectura compleja, no logró el objetivo del 70%. Esto sugiere que las 9 características del dataset pueden no ser suficientes para separar las clases con alta precisión. El modelo, además, genera una cantidad preocupante de Falsos Positivos, lo que lo haría riesgoso para una implementación real.

---

## 🚀 5. Trabajo Futuro

1.  **Probar otros modelos:** Comparar con `Random Forest` o `XGBoost`, que suelen ser superiores en datos tabulares.
2.  **Optimizar Hiperparámetros:** Usar `GridSearchCV` o `KerasTuner` para encontrar una mejor arquitectura.
3.  **Ingeniería de Características:** Crear *ratios* y nuevas interacciones entre las características existentes.

## 💻 6. Stack Técnico

* Python
* TensorFlow (Keras)
* Scikit-learn
* Imbalanced-learn (SMOTE)
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Missingno
