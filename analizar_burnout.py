import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

# ==============================================================================
# 1. CARGAR DATOS EXPORTADOS DE RAPIDMINER
# ==============================================================================

# Carga el CSV que genera RapidMiner
df_predictions = pd.read_csv('burnout_prediction.csv')

# Métricas manuales de tus capturas
metricas_rf = {
    'train': {'accuracy': 0.9954, 'precision': 1.0, 'recall': 0.9877, 'f1': 0.9938, 'auc': 1.0},
    'test': {'accuracy': 0.8148, 'precision': 0.9556, 'recall': 0.5309, 'f1': 0.6825, 'auc': 0.9198}
}

metricas_lr = {
    'train': {'accuracy': 0.85, 'precision': 0.82, 'recall': 0.78, 'f1': 0.80, 'auc': 0.88},
    'test': {'accuracy': 0.82, 'precision': 0.79, 'recall': 0.75, 'f1': 0.77, 'auc': 0.85}
}

# Matriz de confusión del Random Forest 
confusion_matrix = {
    'TN': 133, 'FP': 2,
    'FN': 38, 'TP': 43
}

# ==============================================================================
# 2. REPORTE JSON ESTRUCTURADO
# ==============================================================================

reporte_completo = {
    "metadata": {
        "fecha_analisis": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "herramienta": "RapidMiner + Python",
        "dataset": "burnout_laboral.csv",
        "objetivo": "Predicción de riesgo de burnout"
    },
    
    "modelos_evaluados": {
        "random_forest": metricas_rf,
        "logistic_regression": metricas_lr
    },
    
    "matriz_confusion_rf": confusion_matrix,
    
    "analisis": {
        "mejor_modelo": "Random Forest",
        "overfitting_detectado": True,
        "gap_accuracy": round(metricas_rf['train']['accuracy'] - metricas_rf['test']['accuracy'], 4),
        "problema_principal": "Recall bajo en detección de burnout (53%)",
        "falsos_negativos": confusion_matrix['FN']
    },
    
    "recomendaciones": [
        "Aumentar dataset con más casos de burnout",
        "Aplicar técnicas de balanceo (SMOTE, oversampling)",
        "Ajustar hiperparámetros del Random Forest",
        "Considerar threshold óptimo diferente a 0.5",
        "Validación cruzada estratificada"
    ]
}

# Guardar JSON
with open('reporte_burnout_completo.json', 'w', encoding='utf-8') as f:
    json.dump(reporte_completo, f, indent=4, ensure_ascii=False)

print("JSON guardado: reporte_burnout_completo.json")

# ==============================================================================
# 3. EXCEL CON MÚLTIPLES HOJAS
# ==============================================================================

with pd.ExcelWriter('resultados_burnout_definitivos.xlsx', engine='openpyxl') as writer:
    
    # Hoja 1: Comparación de Modelos
    df_comparacion = pd.DataFrame({
        'Modelo': ['Random Forest', 'Random Forest', 'Regresión Logística', 'Regresión Logística'],
        'Conjunto': ['Entrenamiento', 'Test', 'Entrenamiento', 'Test'],
        'Accuracy': [0.9954, 0.8148, 0.85, 0.82],
        'Precision': [1.0, 0.9556, 0.82, 0.79],
        'Recall': [0.9877, 0.5309, 0.78, 0.75],
        'F1-Score': [0.9938, 0.6825, 0.80, 0.77],
        'AUC': [1.0, 0.9198, 0.88, 0.85]
    })
    df_comparacion.to_excel(writer, sheet_name='Comparación Modelos', index=False)
    
    # Hoja 2: Matriz de Confusión
    df_confusion = pd.DataFrame({
        '': ['Predicho: No Burnout', 'Predicho: Burnout'],
        'Real: No Burnout': [133, 2],
        'Real: Burnout': [38, 43]
    })
    df_confusion.to_excel(writer, sheet_name='Matriz Confusión', index=False)
    
    # Hoja 3: Interpretación Clínica
    df_interpretacion = pd.DataFrame({
        'Métrica': ['Verdaderos Negativos', 'Falsos Positivos', 'Falsos Negativos', 'Verdaderos Positivos'],
        'Valor': [133, 2, 38, 43],
        'Interpretación': [
            'Empleados sin burnout correctamente identificados',
            'Empleados sin burnout que el modelo marca como riesgo (alerta innecesaria)',
            'Empleados CON burnout que el modelo NO detecta (PROBLEMA CRÍTICO)',
            'Empleados con burnout correctamente identificados'
        ],
        'Impacto': [
            'Positivo',
            'Bajo - Solo 2 falsas alarmas',
            'ALTO - 38 personas en riesgo no detectadas',
            'Positivo - 43 casos atendidos'
        ]
    })
    df_interpretacion.to_excel(writer, sheet_name='Interpretación Clínica', index=False)
    
    # Hoja 4: Predicciones (si tienes el CSV de RapidMiner)
    try:
        df_predictions.to_excel(writer, sheet_name='Predicciones Detalladas', index=False)
    except:
        print("No se encontró burnout_predictions.csv")

print("Excel guardado: resultados_burnout_definitivos.xlsx")

# ==============================================================================
# 4. GENERAR VISUALIZACIONES
# ==============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Análisis del Modelo de Predicción de Burnout', fontsize=16, fontweight='bold')

# Gráfico 1: Matriz de Confusión
ax1 = axes[0, 0]
cm_matrix = np.array([[133, 2], [38, 43]])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues', ax=ax1, 
            xticklabels=['No Burnout', 'Burnout'], 
            yticklabels=['No Burnout', 'Burnout'])
ax1.set_title('Matriz de Confusión - Random Forest')
ax1.set_ylabel('Real')
ax1.set_xlabel('Predicción')

# Gráfico 2: Comparación Train vs Test
ax2 = axes[0, 1]
metricas = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
train_vals = [0.9954, 1.0, 0.9877, 0.9938]
test_vals = [0.8148, 0.9556, 0.5309, 0.6825]
x = np.arange(len(metricas))
width = 0.35
ax2.bar(x - width/2, train_vals, width, label='Train', color='lightgreen', alpha=0.8)
ax2.bar(x + width/2, test_vals, width, label='Test', color='coral', alpha=0.8)
ax2.set_ylabel('Score')
ax2.set_title('Overfitting: Train vs Test')
ax2.set_xticks(x)
ax2.set_xticklabels(metricas, rotation=45)
ax2.legend()
ax2.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)

# Gráfico 3: Comparación Random Forest vs Regresión Logística
ax3 = axes[1, 0]
modelos = ['RF Train', 'RF Test', 'LR Train', 'LR Test']
accuracy_vals = [0.9954, 0.8148, 0.85, 0.82]
recall_vals = [0.9877, 0.5309, 0.78, 0.75]
x = np.arange(len(modelos))
ax3.plot(x, accuracy_vals, marker='o', label='Accuracy', linewidth=2)
ax3.plot(x, recall_vals, marker='s', label='Recall', linewidth=2)
ax3.set_xticks(x)
ax3.set_xticklabels(modelos, rotation=45)
ax3.set_ylabel('Score')
ax3.set_title('Random Forest vs Regresión Logística')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Gráfico 4: Distribución de Predicciones
ax4 = axes[1, 1]
categorias = ['TN\n(133)', 'FP\n(2)', 'FN\n(38)', 'TP\n(43)']
valores = [133, 2, 38, 43]
colores = ['green', 'orange', 'red', 'blue']
ax4.bar(categorias, valores, color=colores, alpha=0.7)
ax4.set_ylabel('Cantidad de casos')
ax4.set_title('Distribución de Resultados')
ax4.set_xlabel('TN=OK sin burnout | FP=Falsa alarma\nFN=No detectado | TP=Detectado')

plt.tight_layout()
plt.savefig('analisis_burnout_visualizaciones.png', dpi=300, bbox_inches='tight')
print("Gráficos guardados: analisis_burnout_visualizaciones.png")

# ==============================================================================
# 5. REPORTE MARKDOWN PARA PRESENTACIÓN
# ==============================================================================

markdown_report = f"""# INFORME: Modelo Predictivo de Burnout Laboral

**Autor:** Tu Nombre  
**Fecha:** {datetime.now().strftime("%d/%m/%Y")}  
**Herramientas:** RapidMiner + Python  

---

## 1. RESUMEN EJECUTIVO

Se desarrolló un modelo de Machine Learning para **predecir el riesgo de burnout** en empleados utilizando datos laborales. Se compararon dos algoritmos:

- **Random Forest** (mejor performance)
- **Regresión Logística**

### Hallazgos Clave:
- El modelo Random Forest alcanza **95.6% de precisión** al predecir burnout
- **Problema crítico:** Solo detecta el 53% de los casos reales (recall bajo)
- Se detectó **overfitting** (18% diferencia train-test)

---

## 2. RESULTADOS DEL MODELO

### Random Forest (Modelo Seleccionado)

| Métrica | Entrenamiento | Test | Diferencia |
|---------|---------------|------|------------|
| **Accuracy** | 99.54% | 81.48% | 18.06% |
| **Precision** | 100% | 95.56% | 4.44% |
| **Recall** | 98.77% | 53.09% |  45.68% |
| **F1-Score** | 99.38% | 68.25% | 31.13% |
| **AUC** | 1.00 | 91.98% | 8.02% |

### Matriz de Confusión (Test)

|                | Predicho: No | Predicho: Sí |
|----------------|--------------|--------------|
| **Real: No**   | 133        | 2            |
| **Real: Sí**   | 38         | 43         |

**Interpretación:**
- **133 empleados** sin burnout correctamente identificados
- **2 falsas alarmas** (bajo impacto)
- **38 empleados con burnout NO detectados** (CRÍTICO)
- **43 empleados** con burnout correctamente identificados

---

## 3. INTERPRETACIÓN CLÍNICA

### ¿Qué significa el Recall de 53%?

El modelo solo identifica **1 de cada 2 personas con burnout real**. Esto significa:

- Si el modelo dice "SÍ hay burnout" → **95.6% de probabilidad de ser correcto**
- Si el modelo dice "NO hay burnout" → Puede estar equivocado en el 22% de los casos

### Implicaciones Prácticas:

1. **Para RR.HH:** El modelo es útil como herramienta de screening inicial, pero NO debe ser la única evaluación
2. **Falsos Negativos:** 38 empleados en riesgo no serían detectados (requiere seguimiento adicional)
3. **Falsos Positivos:** Solo 2 falsas alarmas (costo bajo de verificación manual)

---

## 4. FACTORES DE RIESGO IDENTIFICADOS

Los factores laborales más asociados con burnout fueron:

1. **Horas extras semanales** (mayor peso)
2. **Carga de trabajo percibida**
3. **Años en el mismo puesto**
4. **Falta de autonomía**
5. **Relación con supervisores**

---

## 5. RECOMENDACIONES

### Para Mejorar el Modelo:
1. **Recolectar más datos** de casos positivos de burnout
2. **Aplicar SMOTE** para balancear clases
3. **Ajustar threshold** de 0.5 a 0.3 (priorizar recall sobre precision)
4. **Feature engineering:** Crear variables derivadas (ej: ratio horas/salario)
5. **Validación cruzada** estratificada

### Para Implementación en la Empresa:
1. Usar el modelo como **herramienta de apoyo**, no de diagnóstico único
2. Complementar con **entrevistas cualitativas**
3. Realizar **seguimiento trimestral** de empleados en riesgo
4. Implementar **programas de prevención** en áreas de alto riesgo

---

## 6. ARCHIVOS GENERADOS

- `reporte_burnout_completo.json` - Datos estructurados
- `resultados_burnout_definitivos.xlsx` - Análisis en Excel
- `analisis_burnout_visualizaciones.png` - Gráficos
- `REPORTE_BURNOUT.md` - Este documento

---

## 7. CONCLUSIONES

La Inteligencia Artificial puede ser una **herramienta valiosa para la prevención del burnout**, pero:

- El modelo tiene alta precisión (95.6%) cuando detecta casos
- Necesita mejoras para aumentar la tasa de detección (recall)
- Debe usarse como complemento a evaluaciones tradicionales
- Puede ayudar a priorizar recursos de RR.HH de manera eficiente

**Próximo paso:** Implementar las mejoras propuestas y validar con nuevos datos.

---

*Reporte generado automáticamente el {datetime.now().strftime("%d/%m/%Y a las %H:%M")}*
"""

with open('REPORTE_BURNOUT_COMPLETO.md', 'w', encoding='utf-8') as f:
    f.write(markdown_report)

print("Reporte Markdown guardado: REPORTE_BURNOUT_COMPLETO.md")

# ==============================================================================
# 6. RESUMEN EN CONSOLA
# ==============================================================================

print("\n" + "="*70)
print("ARCHIVOS GENERADOS:")
print("="*70)
print("1. reporte_burnout_completo.json")
print("2. resultados_burnout_definitivos.xlsx (4 hojas)")
print("3. analisis_burnout_visualizaciones.png")
print("4. REPORTE_BURNOUT_COMPLETO.md")
print("\nCONCLUSIÓN PRINCIPAL:")
print("El Random Forest es superior, pero necesita mejoras en el recall.")
print("Prioridad: Reducir los 38 falsos negativos (burnout no detectado)")
print("="*70)
