# INFORME: Modelo Predictivo de Burnout Laboral

**Autor:** Tu Nombre  
**Fecha:** 28/11/2025  
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

*Reporte generado automáticamente el 28/11/2025 a las 17:32*
