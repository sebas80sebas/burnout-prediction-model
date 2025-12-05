# Sistema de Predicción de Burnout Laboral

## Descripción del Proyecto

Sistema de Machine Learning para predecir el riesgo de burnout en empleados basándose en datos de absentismo laboral. Utiliza **Altair AI Studio** (anteriormente RapidMiner) con procesamiento Python para ingeniería de características y dos modelos de clasificación: **Regresión Logística** y **Random Forest**.

---

## Tecnologías Utilizadas

- **Altair AI Studio 11.1.001**
- **Python 3.x** con extensión Python Scripting
- **H2O** para Logistic Regression
- **Pandas** para manipulación de datos

---

## Librerías Necesarias

```bash
pip install pandas openpyxl matplotlib seaborn numpy scipy scikit-learn imbalanced-learn
```

---

## Estructura del Proyecto

```
burnout-prediction-model/
│
├── prediccion_burnout.rmp          # Proceso principal de Altair AI Studio
├── prediccion_burnout.xml          # Backup del proceso en XML
├── feature_engineering.py          # Script de ingeniería de características
├── oversampling.py                 # Script de balanceo de datos (SMOTE)
├── README.md                       # Este archivo
├── dataset/
│   └── Absenteeism_at_work.csv    # Dataset original
└── burnout_prediction.csv      # Resultados y métricas 
```

---

## Arquitectura del Proceso Completo

```
PROCESO: Detección de Burnout Laboral
│
├── 1. Read CSV (Absenteeism_at_work.csv)
│   └── Delimitador: ";"
│   └── 21 columnas originales
│
├── 2. Execute Python - Feature Engineering
│   └── Script: feature_engineering.py
│   └── Genera ~40 nuevas características
│
├── 3. Execute Python - Oversampling (SMOTE)
│   └── Script: oversampling.py
│   └── Balancea la clase minoritaria (Burnout_Risk)
│
├── 4. Set Role
│   └── Asigna "Burnout_Risk" como label (variable objetivo)
│
├── 5. Nominal to Binominal
│   └── Convierte "Burnout_Risk" a variable binaria
│   └── Valores: "true" / "false"
│
├── 6. Split Data (80% Train / 20% Test)
│   └── Estratificado por Burnout_Risk
│   └── Random seed: 42
│
├── 7. Multiply (Training Set)
│   └── Duplica datos para enviar a ambos modelos
│   │
│   ├── → Branch 1: Logistic Regression
│   │   ├── Select Attributes (40 features)
│   │   └── Normalize (Z-transformation)
│   │
│   └── → Branch 2: Random Forest
│       └── Todas las features sin normalizar
│
├── 8. Multiply (Test Set)
│   └── Duplica test set para evaluar ambos modelos
│
├── 9. Model Training
│   ├── Logistic Regression (H2O)
│   └── Random Forest (100 árboles)
│
├── 10. Apply Model
│   ├── Apply Model (Random Forest)
│   └── Apply Model (2) (Logistic Regression)
│
├── 11. Performance Evaluation
│   ├── Performance (Random Forest)
│   │   └── Performance to Data
│   │
│   └── Performance (2) (Logistic Regression)
│       └── Performance to Data (2)
│
├── 12. Append
│   └── Combina métricas de ambos modelos
│
└── 13. Write CSV
    └── Exporta resultados: burnout_prediction.csv
```

---

## Operadores de Altair AI Studio

### 1. Read CSV
**Función**: Cargar el dataset base de absentismo laboral

**Parámetros**:
```
csv_file: Absenteeism_at_work.csv
column_separators: ";"
use_quotes: true
parse_numbers: true
decimal_character: "."
use_header_row: true
encoding: UTF-8
time_zone: Europe/Madrid
locale: English (United States)
```

**Columnas originales** (21):
- ID
- Reason for absence
- Month of absence
- Day of the week
- Seasons
- Transportation expense
- Distance from Residence to Work
- Service time
- Age
- Work load Average/day
- Hit target
- Disciplinary failure
- Education
- Son
- Social drinker
- Social smoker
- Pet
- Weight
- Height
- Body mass index
- Absenteeism time in hours

---

### 2. Execute Python - Feature Engineering
**Función**: Crear variables derivadas temporales, estacionales y de comportamiento

**Configuración**:
```
script_file: feature_engineering.py
use_default_python: true
package_manager: conda (anaconda)
use_macros: false
```

**Features generadas** (~20 nuevas):
- **Temporales**: `Es_Lunes`, `Es_Viernes`, `Dia_Semana`, `Mes`, `Trimestre`, `Estacion`
- **Interacciones**: `Edad_X_Experiencia`, `Distancia_X_Lunes`
- **Categorías**: `Grupo_Edad`, `Nivel_Experiencia`
- **Indicadores**: `Cerca_Vacaciones`, `Cierre_Trimestre`, `Inicio_Fin_Semana`
- **Acumulados**: `Ausencias_Acumuladas`, `Freq_Ausencias_Medicas`
- **Binarios**: `Ausencia_Medica_Seria`, `Commute_Largo`, `Sobrecarga`
- **Sintético**: `Es_Sintetico` (si se añaden datos)

---

### 3. Execute Python - Oversampling
**Función**: Balancear clases mediante técnica SMOTE (Synthetic Minority Over-sampling Technique)

**Configuración**:
```
script_file: oversampling.py
use_default_python: true
package_manager: conda (anaconda)
use_macros: false
```

**Técnica**: SMOTE
- Genera ejemplos sintéticos de la clase minoritaria
- Evita overfitting por simple replicación
- Mejora el recall de la clase positiva (Burnout)

---

### 4. Set Role
**Función**: Definir la variable objetivo del modelo

**Parámetros**:
```
set_roles:
  - Burnout_Risk: label
```

---

### 5. Nominal to Binominal
**Función**: Convertir la variable objetivo a formato binario

**Parámetros**:
```
attribute_filter_type: single
attribute: Burnout_Risk
transform_binominal: false
use_underscore_in_name: false
include_special_attributes: true
```

**Resultado**: `Burnout_Risk = true` o `Burnout_Risk = false`

---

### 6. Split Data
**Función**: Dividir datos en conjuntos de entrenamiento y prueba

**Parámetros**:
```
partitions:
  - ratio: 0.8  (80% Training)
  - ratio: 0.2  (20% Test)
sampling_type: stratified sampling
use_local_random_seed: true
local_random_seed: 42
```

**Estratificación**: Mantiene la proporción de `Burnout_Risk = true/false` en ambos conjuntos

---

### 7. Multiply (Training Set)
**Función**: Duplicar el training set para alimentar ambos modelos en paralelo

**Salidas**:
- **Output 1** → Logistic Regression (con Select Attributes + Normalize)
- **Output 2** → Random Forest (todas las features, sin normalizar)

---

### 8. Select Attributes
**Función**: Seleccionar subset de 40 características para Logistic Regression

**Parámetros**:
```
type: include attributes
attribute_filter_type: a subset
```

**Features seleccionadas** (40):
```
Absenteeism time in hours, Age, Ausencia_Medica_Seria, Ausencias_Acumuladas,
Body mass index, Cerca_Vacaciones, Cierre_Trimestre, Commute_Largo,
Day of the week, Dia_Semana, Disciplinary failure, Distance from Residence to Work,
Distancia_X_Lunes, Edad_X_Experiencia, Education, Es_Lunes, Es_Sintetico,
Es_Viernes, Estacion, Freq_Ausencias_Medicas, Grupo_Edad, Height, Hit target,
Inicio_Fin_Semana, Mes, Month of absence, Nivel_Experiencia, Pet,
Reason for absence, Seasons, Service time, Sobrecarga, Social drinker,
Social smoker, Son, Transportation expense, Trimestre, Weight, Work load Average/day
```

**Excluye**: `ID`, `Burnout_Risk` (label)

---

### 9. Normalize
**Función**: Estandarizar features numéricas (solo para Logistic Regression)

**Parámetros**:
```
method: Z-transformation
attribute_filter_type: all
value_type: numeric
include_special_attributes: false
```

**Fórmula**: 
```
z = (x - μ) / σ
```
Donde:
- `μ` = media
- `σ` = desviación estándar

**Resultado**: Variables con media = 0 y desviación estándar = 1

---

### 10. Multiply (Test Set)
**Función**: Duplicar el test set para evaluación de ambos modelos

**Salidas**:
- **Output 1** → Evaluación de Random Forest
- **Output 2** → Evaluación de Logistic Regression

---

### 11. Logistic Regression (H2O)
**Función**: Entrenar modelo de regresión logística

**Parámetros**:
```
solver: AUTO
reproducible: false
maximum_number_of_threads: 4
use_regularization: false
lambda_search: false
early_stopping: true
stopping_rounds: 3
stopping_tolerance: 0.001
standardize: true
non-negative_coefficients: false
add_intercept: true
compute_p-values: true
remove_collinear_columns: true
missing_values_handling: MeanImputation
max_iterations: 0 (sin límite)
max_runtime_seconds: 0 (sin límite)
```

**Características**:
- Modelo interpretable (coeficientes claros)
- Rápido entrenamiento
- Calcula p-values para significancia estadística
- Maneja colinealidad automáticamente
- Requiere normalización previa
- Asume relaciones lineales

---

### 12. Random Forest
**Función**: Entrenar modelo de bosque aleatorio

**Parámetros**:
```
number_of_trees: 100
criterion: gain_ratio
maximal_depth: 20
apply_pruning: false
confidence: 0.1
apply_prepruning: false
minimal_gain: 0.01
minimal_leaf_size: 2
minimal_size_for_split: 4
number_of_prepruning_alternatives: 3
random_splits: false
guess_subset_ratio: true
subset_ratio: 0.2
voting_strategy: confidence vote
use_local_random_seed: false
enable_parallel_execution: true
```

**Características**:
- Alta precisión
- Maneja relaciones no lineales
- No requiere normalización
- Feature importance automático
- Robusto a outliers
- Menos interpretable (caja negra)
- Mayor tiempo de entrenamiento

---

### 13. Apply Model
**Función**: Aplicar modelos entrenados al test set

**Configuración**:
- **Apply Model**: Random Forest → Test Set (output 1)
- **Apply Model (2)**: Logistic Regression → Test Set (output 2)

**Output**: Test set con columnas adicionales:
- `prediction(Burnout_Risk)`: Predicción del modelo
- `confidence(true)`: Probabilidad de burnout
- `confidence(false)`: Probabilidad de no burnout

---

### 14. Performance (Binominal Classification)
**Función**: Calcular métricas de rendimiento de cada modelo

**Parámetros**:
```
main_criterion: accuracy
accuracy: true
classification_error: true
AUC: true
precision: true
recall: true
f_measure: true
skip_undefined_labels: true
use_example_weights: true
```

**Métricas calculadas**:
- **Accuracy**: % de predicciones correctas
- **Precision**: TP / (TP + FP) - Calidad de positivos
- **Recall (Sensitivity)**: TP / (TP + FN) - Capacidad de detectar positivos
- **F1-Score**: Media armónica de Precision y Recall
- **AUC**: Área bajo la curva ROC (0.5 = random, 1.0 = perfecto)
- **Classification Error**: % de predicciones incorrectas

---

### 15. Performance to Data
**Función**: Convertir métricas de performance a formato tabular

**Uso**:
- **Performance to Data**: Para Random Forest
- **Performance to Data (2)**: Para Logistic Regression

**Output**: ExampleSet con columnas:
```
Criterion | Value
----------|-------
accuracy  | 0.XX
precision | 0.XX
recall    | 0.XX
f1        | 0.XX
AUC       | 0.XX
```

---

### 16. Append
**Función**: Combinar métricas de ambos modelos en un único ExampleSet

**Parámetros**:
```
data_management: auto
merge_type: all
```

**Resultado**: Tabla comparativa con métricas de ambos modelos

---

### 17. Write CSV
**Función**: Exportar resultados finales a archivo CSV

**Parámetros**:
```
csv_file: burnout_prediction.csv
column_separator: ";"
write_attribute_names: true
quote_nominal_values: true
format_date_attributes: true
date_format: yyyy-MM-dd HH:mm:ss
append_to_file: false
encoding: SYSTEM
```

**Contenido del CSV**:
- Métricas de ambos modelos (accuracy, precision, recall, F1, AUC)
- Identificación del modelo (`criterion` column)

---

## Diferencias en el Preprocessing por Modelo

### Random Forest:
| Característica | Valor |
|----------------|-------|
| Features utilizadas | **TODAS** (40+ variables) |
| Normalización | NO requerida |
| Feature Selection | NO aplica |
| Manejo de categóricas | Excelente |
| Manejo de outliers | Robusto |
| Relaciones no lineales | Captura automáticamente |

### Logistic Regression:
| Característica | Valor |
|----------------|-------|
| Features utilizadas | **40 seleccionadas** (Select Attributes) |
| Normalización | Z-transformation obligatoria |
| Feature Selection | Manual (40 features) |
| Manejo de categóricas | Requiere encoding |
| Manejo de outliers | Sensible |
| Relaciones no lineales | Solo lineales |

---

## Diagrama de Conexiones (Puertos)

```
Read CSV → Execute Python (Feature Eng) → Execute Python (2) [Oversampling] 
    ↓
Set Role → Nominal to Binominal → Split Data
    ↓                                   ↓
partition 1 (80% Train)          partition 2 (20% Test)
    ↓                                   ↓
Multiply ──┬─────────────────────────> Multiply (2) ──┬──────────────────┐
           │                                           │                  │
    output 1                                    output 1            output 2
           │                                           │                  │
    Select Attributes                                  │                  │
           ↓                                           │                  │
    Normalize                                          │                  │
           ↓                                           │                  │
    Logistic Regression                                │                  │
           │                                           │                  │
           └────────> Apply Model (2) <────────────────┘                  │
                            ↓                                             │
                      Performance (2)                                     │
                            ↓                                             │
                 Performance to Data (2) ───────────────┐                 │
                                                        │                 │
    output 2                                            │                 │
           │                                            │                 │
    Random Forest                                       │                 │
           │                                            │                 │
           └────────> Apply Model <─────────────────────┼─────────────────┘
                            ↓                           │
                      Performance                       │
                            ↓                           │
                 Performance to Data ────────────────> Append
                                                        ↓
                                                   Write CSV
                                                        ↓
                                                    Result 1
```

---

## Outputs del Sistema

### 1. burnout_prediction.csv
Archivo con métricas comparativas de ambos modelos:

```csv
Criterion;Value
accuracy;0.XX
precision;0.XX
recall;0.XX
f_measure;0.XX
AUC;0.XX
```

### 2. Matriz de Confusión (visualización en AI Studio)
```
                Predicted
              False    True
Actual False    TN      FP
       True     FN      TP
```

### 3. Feature Importance (de Random Forest)
Lista de las variables más influyentes en la predicción

### 4. Coeficientes (de Logistic Regression)
Pesos de cada variable con p-values de significancia

---

## Cómo Ejecutar el Proceso

### Paso 1: Preparar el Entorno
```bash
# Instalar librerías Python
pip install pandas numpy scikit-learn imbalanced-learn

# Verificar que Altair AI Studio tiene Python Scripting habilitado
```

### Paso 2: Configurar Rutas en AI Studio
Editar estos parámetros en el proceso:

1. **Read CSV**:
   ```
   csv_file: burnout-prediction-model/dataset/Absenteeism_at_work.csv
   ```

2. **Execute Python**:
   ```
   script_file: burnout-prediction-model/feature_engineering.py
   ```

3. **Execute Python (2)**:
   ```
   script_file: burnout-prediction-model/oversampling.py
   ```

4. **Write CSV**:
   ```
   csv_file: burnout-prediction-model/burnout_prediction.csv
   ```

### Paso 3: Ejecutar el Proceso
1. Abrir `prediccion_burnout.rmp` en Altair AI Studio
2. Click en el botón **Run Process**
3. Esperar finalización (~2-5 minutos)
4. Revisar resultados en `burnout_prediction.csv`

---

## Resultados Esperados

### Métricas Objetivo:
| Métrica | Random Forest | Logistic Regression |
|---------|---------------|---------------------|
| Accuracy | > 0.85 | > 0.80 |
| Precision | > 0.80 | > 0.75 |
| Recall | > 0.75 | > 0.70 |
| F1-Score | > 0.77 | > 0.72 |
| AUC | > 0.88 | > 0.82 |

### Interpretación:
- **Random Forest** suele tener mejor rendimiento global
- **Logistic Regression** es más interpretable para análisis de factores de riesgo
- **AUC > 0.80** indica un buen poder discriminativo

---

## Visualización de Resultados

### Script de Análisis y Gráficos

El proyecto incluye un script Python (`analizar_burnout.py`) que genera visualizaciones completas del modelo:

#### Requisitos Adicionales
```bash
matplotlib seaborn numpy pandas openpyxl scikit-learn
```

#### Ejecución del Script

```bash
# Asegurarse de estar en el directorio del proyecto
cd burnout-prediction-model

# Instalar requisitos 
virtualenv venv
. venv/bin/activate

# Ejecutar el script de análisis
python analizar_burnout.py
```

#### Salidas Generadas

El script produce dos archivos de visualización:

1. **analisis_burnout_completo.png** - Contiene 4 gráficos:
   - **Train vs Test Metrics**: Comparación de métricas entre entrenamiento y prueba
   - **Radar de Métricas (Test)**: Visualización radial del rendimiento en test
   - **Distribución de Resultados Estimada**: Gráfico de barras con TP, FP, FN, TN
   - **Diferencia Train vs Test (Gap)**: Análisis de overfitting/underfitting

2. **Salida en consola**:
   ```
   === Métricas Train ===
   === Métricas Test ===
   === Comparación Train vs Test ===
   === Matriz de Confusión Estimada ===
   ```

#### Interpretación de los Gráficos

##### 1. Train vs Test Metrics
- Barras verdes = Rendimiento en entrenamiento
- Barras coral = Rendimiento en prueba
- Línea punteada en 0.8 = Threshold de calidad
- **Interpretación**: Si las barras test están cerca de train, el modelo generaliza bien

##### 2. Radar de Métricas (Test)
- Pentágono azul muestra balance entre métricas
- Área rellena = Rendimiento global
- **Ideal**: Pentágono uniforme y cercano al borde exterior

##### 3. Distribución de Resultados
- **TN (Verde)**: Verdaderos Negativos - Correctamente identificados sin burnout
- **FP (Naranja)**: Falsos Positivos - Incorrectamente identificados con burnout
- **FN (Rojo)**: Falsos Negativos - No detectados (casos perdidos) ⚠️
- **TP (Azul)**: Verdaderos Positivos - Correctamente identificados con burnout

##### 4. Gap Train-Test
- Barras positivas = Train > Test (posible overfitting)
- Barras negativas = Test > Train (poco común, revisar datos)
- **Objetivo**: Gaps cercanos a 0

#### Ejemplo de Uso

```bash
# Después de ejecutar el proceso en Altair AI Studio
# y generar burnout_prediction.csv

# 1. Ejecutar análisis
python analizar_burnout.py

# 2. Revisar métricas en consola
# 3. Abrir imagen generada
# En Windows:
start analisis_burnout_completo.png
# En Linux/Mac:
xdg-open analisis_burnout_completo.png
# o
open analisis_burnout_completo.png
```

#### Personalización del Script

Para modificar los gráficos, editar las siguientes secciones en `analizar_burnout.py`:

```python
# Cambiar tamaño de figura
fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # Modificar (ancho, alto)

# Cambiar colores
colors = ['green','orange','red','blue']  # TN, FP, FN, TP

# Cambiar resolución de salida
plt.savefig("analisis_burnout_completo.png", dpi=300)  # dpi=150 o 600
```

---
## Troubleshooting

### Error: "Python Scripting extension not found"
**Solución**: Instalar extensión desde AI Studio → Extensions → Python Scripting

### Error: "File not found"
**Solución**: Verificar rutas absolutas en todos los operadores (Read CSV, Execute Python, Write CSV)

### Error: "Import error: No module named 'imblearn'"
**Solución**: 
```bash
pip install imbalanced-learn
```

### Warning: "Missing values detected"
**Solución**: El modelo usa `MeanImputation` automáticamente, pero revisar calidad de datos

### Performance muy bajo (< 0.70)
**Posibles causas**:
1. Desbalanceo de clases no corregido → Verificar `oversampling.py`
2. Features mal normalizadas → Verificar operador `Normalize`
3. Overfitting → Reducir `maximal_depth` en Random Forest

---

## Referencias

- [Altair AI Studio Documentation](https://docs.altair.com/rapidminer/)
- [H2O Logistic Regression](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/glm.html)
- [SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813)
- [Absenteeism at Work Dataset](https://archive.ics.uci.edu/ml/datasets/Absenteeism+at+work)

---

## Autor

**Iván Sebastián Loor Weir**  
Proyecto de Predicción de Burnout Laboral  
[GitHub](https://github.com/sebas80sebas/burnout-prediction-model)

---

## Licencia

Este proyecto está bajo licencia MIT. Ver archivo `LICENSE` para más detalles.

---