# 🔧 Proceso Completo en Altair AI Studio + Python

## Librerías necesarias para el repositorio
```bash
pip install pandas openpyxl matplotlib seaborn numpy
```

## Estructura del Proceso en Altair AI Studio

```
PROCESO: Detección de Burnout Laboral
│
├── 1. Read CSV (Absenteeism_at_work.csv)
│   └── Configurar delimitador: ";"
│
├── 2. Execute Python - Feature Engineering
│   └── Script: feature_engineering.py
│
├── 3. Execute Python - Aumento de Datos (Oversampling)
│   └── Script: oversampling.py
│
├── 4. Split Data (80% Train / 20% Test)
│   └── Estratificado por Burnout_Risk
│
├── 5. Normalize (Z-transformation)
│   └── Solo para Regresión Logística
│
├── 6. Logistic Regression
│   └── kernel: Auto
│
├── 7. Random Forest
│   └── number of trees: 100
│   └── criterion: gain ratio
│
├── 8. Apply Model
│   └── Aplicar ambos modelos al test set
│
├── 9. Performance (Classification)
│   └── Métricas: Accuracy, Precision, Recall, AUC
│
└── 10. Write Results
    └── Exportar predicciones y métricas
```

## Operadores Clave a Usar

### 1️⃣ **Read CSV**
- **Función**: Cargar el dataset base
- **Parámetros**:
  - csv file: `Absenteeism_at_work.csv`
  - column separators: `;`
  - use quotes: yes

### 2️⃣ **Execute Python** (Feature Engineering)
- **Función**: Añadir variables temporales y externas
- **Input**: ExampleSet del Read CSV
- **Output**: ExampleSet enriquecido
- **Script**: Ver `feature_engineering.py`

### 3️⃣ **Execute Python** (Oversampling)
- **Función**: Aumentar datos con oversampling
- **Input**: ExampleSet enriquecido
- **Output**: ExampleSet balanceado
- **Script**: Ver `oversampling.py`

### 4️⃣ **Split Data**
- **Parámetros**:
  - split ratio: 0.8
  - sampling type: stratified sampling
  - local random seed: 42

### 5️⃣ **Normalize**
- **Tipo**: Z-transformation (mean=0, std=1)
- **Apply to**: Solo features numéricas
- **Exclude**: Burnout_Risk, ID

### 6️⃣ **Logistic Regression**
- **Parámetros**:
    - Solver: AUTO → selecciona automáticamente el mejor método de optimización según los datos.
    - Add intercept (use bias): TRUE → añade el término independiente al modelo.
    - Use regularization: FALSE → no se aplica penalización L1/L2.
    - Standardize: TRUE → las variables se estandarizan (media 0, desviación 1) antes del entrenamiento.
    - Missing values handling: Mean Imputation → los valores faltantes se sustituyen por la media.
    - Early stopping: activado (3 rondas, tolerancia 0.001) → evita sobreajuste deteniendo el entrenamiento si no hay mejora.
    - Compute p-values: TRUE → calcula los valores p para evaluar la significancia de cada coeficiente.
    - Remove collinear columns: TRUE → elimina atributos altamente correlacionados.
    - Add intercept (bias): TRUE → incluye el término independiente.

### 7️⃣ **Random Forest**
- **Parámetros**:
    - number of trees = 100
    - criterion = gain_ratio
    - maximal depth = 20
    - apply pruning = no
    - apply prepruning = no
    - voting strategy = confidence vote
    - guess subset ratio = yes
    - parallel execution = yes

### 8️⃣ **Apply Model**
- **Input**: Model + Test Set
- **Output**: Labeled ExampleSet

### 9️⃣ **Performance (Classification)**
- **Métricas a calcular**:
  - ✓ accuracy
  - ✓ precision
  - ✓ recall
  - ✓ f1-score
  - ✓ AUC (area under curve)
  - ✓ confusion matrix

### 🔟 **Write Results**
- **Formato**: CSV o Excel
- **Incluir**: Predicciones, probabilidades, métricas

---

## 🎯 Flujo Visual del Proceso

```
[Read CSV] 
    ↓
[Execute Python: Feature Eng] → Variables temporales, estacionales
    ↓
[Execute Python: Oversampling] → Balanceo de clases
    ↓
[Split Data] → Training (80%) / Test (20%)
    ↓                    ↓
[Normalize]          [Normalize]
    ↓                    ↓
[Logistic Reg]       [Random Forest]
    ↓                    ↓
[Apply Model]        [Apply Model]
    ↓                    ↓
[Performance]        [Performance]
    ↓                    ↓
[Compare Results] ← Determinar mejor modelo
    ↓
[Write Results] → Exportar predicciones finales
```

---

## 📊 Outputs Esperados

1. **Métricas de rendimiento** (archivo CSV):
   - Accuracy, Precision, Recall, F1-Score, AUC
   - Por cada modelo

2. **Matriz de confusión** (visualización):
   - True Positives, False Positives
   - True Negatives, False Negatives

3. **Feature Importance** (gráfico):
   - Variables más importantes para predecir burnout

4. **Predicciones finales** (CSV):
   - ID, Real, Predicho, Probabilidad

5. **Curva ROC** (imagen):
   - Comparación visual de modelos
