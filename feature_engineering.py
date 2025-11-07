"""
SCRIPT PARA ALTAIR AI STUDIO
Operador: Execute Python - Feature Engineering
Input: rm_main (ExampleSet de Altair)
Output: rm_main (ExampleSet enriquecido)
"""

import pandas as pd
import numpy as np

# ============================================
# RECIBIR DATOS DESDE ALTAIR AI STUDIO
# ============================================
# Altair AI Studio pasa los datos como 'rm_main'
# Convertir a DataFrame de pandas
df = rm_main

print("="*60)
print("FEATURE ENGINEERING EN ALTAIR AI STUDIO")
print("="*60)
print(f"Dataset recibido: {len(df)} registros, {len(df.columns)} columnas")

# ============================================
# 1. VARIABLES TEMPORALES/ESTACIONALES
# ============================================
print("\n🔧 Creando variables temporales...")

# Obtener columnas de mes y día
if 'Month of absence' in df.columns:
    df['Mes'] = df['Month of absence']
else:
    print("⚠️ Columna 'Month of absence' no encontrada, usando valores aleatorios")
    df['Mes'] = np.random.randint(1, 13, size=len(df))

if 'Day of the week' in df.columns:
    df['Dia_Semana'] = df['Day of the week']
else:
    df['Dia_Semana'] = np.random.randint(2, 7, size=len(df))

# Estación del año
estaciones = {
    1: 0, 2: 0, 3: 1,      # Invierno=0, Primavera=1
    4: 1, 5: 1, 6: 2,      # Verano=2
    7: 2, 8: 2, 9: 3,      # Otoño=3
    10: 3, 11: 3, 12: 0
}
df['Estacion'] = df['Mes'].map(estaciones)

# Proximidad a vacaciones (1=cerca, 0=lejos)
meses_prevacaciones = [6, 7, 11, 12]
df['Cerca_Vacaciones'] = df['Mes'].isin(meses_prevacaciones).astype(int)

# Día de la semana específico
df['Es_Lunes'] = (df['Dia_Semana'] == 2).astype(int)
df['Es_Viernes'] = (df['Dia_Semana'] == 6).astype(int)

# Trimestre y cierre trimestral
df['Trimestre'] = pd.cut(df['Mes'], bins=[0, 3, 6, 9, 12], labels=[1, 2, 3, 4])
df['Trimestre'] = df['Trimestre'].astype(int)
df['Cierre_Trimestre'] = df['Mes'].isin([3, 6, 9, 12]).astype(int)

print(f"   ✓ Variables temporales creadas: 7 nuevas variables")

# ============================================
# 2. VARIABLES DE CARGA LABORAL
# ============================================
print("\n🔧 Creando variables de carga laboral...")

# Ratio de carga laboral
if 'Work load Average/day ' in df.columns:
    avg_workload = df['Work load Average/day '].mean()
    df['Sobrecarga'] = (df['Work load Average/day '] > avg_workload * 1.25).astype(int)
    print(f"   ✓ Variable 'Sobrecarga' creada (umbral: {avg_workload * 1.25:.2f})")

# Ausencias acumuladas por empleado
if 'ID' in df.columns and 'Absenteeism time in hours' in df.columns:
    df = df.sort_values(['ID', 'Mes'])
    df['Ausencias_Acumuladas'] = df.groupby('ID')['Absenteeism time in hours'].cumsum()
    print(f"   ✓ Variable 'Ausencias_Acumuladas' creada")

# ============================================
# 3. VARIABLES DEMOGRÁFICAS
# ============================================
print("\n🔧 Creando variables demográficas...")

# Grupos de edad
if 'Age' in df.columns:
    df['Grupo_Edad'] = pd.cut(df['Age'], 
                               bins=[0, 30, 40, 50, 100], 
                               labels=[1, 2, 3, 4])  # 1=Joven, 2=Adulto, 3=Maduro, 4=Senior
    df['Grupo_Edad'] = df['Grupo_Edad'].astype(int)
    print(f"   ✓ Variable 'Grupo_Edad' creada")

# Nivel de experiencia
if 'Service time' in df.columns:
    df['Nivel_Experiencia'] = pd.cut(df['Service time'], 
                                      bins=[0, 5, 10, 15, 100], 
                                      labels=[1, 2, 3, 4])  # 1=Junior, 2=Mid, 3=Senior, 4=Expert
    df['Nivel_Experiencia'] = df['Nivel_Experiencia'].astype(int)
    print(f"   ✓ Variable 'Nivel_Experiencia' creada")

# Commute largo
if 'Distance from Residence to Work' in df.columns:
    threshold_commute = df['Distance from Residence to Work'].quantile(0.75)
    df['Commute_Largo'] = (df['Distance from Residence to Work'] > threshold_commute).astype(int)
    print(f"   ✓ Variable 'Commute_Largo' creada (umbral: {threshold_commute:.2f})")

# ============================================
# 4. VARIABLES DE SALUD
# ============================================
print("\n🔧 Creando variables de salud...")

# Ausencia médica seria
if 'Reason for absence' in df.columns:
    razones_medicas_serias = [23, 22, 21, 11, 13, 19, 27, 28]
    df['Ausencia_Medica_Seria'] = df['Reason for absence'].isin(razones_medicas_serias).astype(int)
    print(f"   ✓ Variable 'Ausencia_Medica_Seria' creada")

# ============================================
# 5. INTERACCIONES ENTRE VARIABLES
# ============================================
print("\n🔧 Creando interacciones entre variables...")

if 'Age' in df.columns and 'Service time' in df.columns:
    df['Edad_X_Experiencia'] = df['Age'] * df['Service time']
    print(f"   ✓ Interacción 'Edad_X_Experiencia' creada")

if 'Distance from Residence to Work' in df.columns and 'Es_Lunes' in df.columns:
    df['Distancia_X_Lunes'] = df['Distance from Residence to Work'] * df['Es_Lunes']
    print(f"   ✓ Interacción 'Distancia_X_Lunes' creada")

# ============================================
# 6. CREAR VARIABLE OBJETIVO (BURNOUT_RISK)
# ============================================
print("\n🎯 Creando variable objetivo: Burnout_Risk...")

if 'Absenteeism time in hours' in df.columns:
    # Usar el percentil 75 como umbral de riesgo
    threshold_burnout = df['Absenteeism time in hours'].quantile(0.75)
    df['Burnout_Risk'] = (df['Absenteeism time in hours'] > threshold_burnout).astype(int)
    
    print(f"   ✓ Variable objetivo creada")
    print(f"   ✓ Umbral de burnout: {threshold_burnout} horas")
    print(f"   ✓ Distribución:")
    print(f"      - Bajo riesgo (0): {(df['Burnout_Risk'] == 0).sum()} ({(df['Burnout_Risk'] == 0).sum()/len(df)*100:.1f}%)")
    print(f"      - Alto riesgo (1): {(df['Burnout_Risk'] == 1).sum()} ({(df['Burnout_Risk'] == 1).sum()/len(df)*100:.1f}%)")

# ============================================
# 7. LIMPIAR VALORES NULOS
# ============================================
print("\n🧹 Limpiando valores nulos...")

nulos_antes = df.isnull().sum().sum()
if nulos_antes > 0:
    # Rellenar nulos numéricos con la mediana
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    print(f"   ✓ {nulos_antes} valores nulos rellenados con la mediana")
else:
    print(f"   ✓ No hay valores nulos")

# ============================================
# 8. RESUMEN FINAL
# ============================================
print("\n" + "="*60)
print("✅ FEATURE ENGINEERING COMPLETADO")
print("="*60)
print(f"Variables originales: {len(rm_main.columns)}")
print(f"Variables nuevas: {len(df.columns) - len(rm_main.columns)}")
print(f"Variables totales: {len(df.columns)}")
print(f"Registros: {len(df)}")

# ============================================
# DEVOLVER DATOS A ALTAIR AI STUDIO
# ============================================
# El DataFrame 'df' se devuelve automáticamente como 'rm_main'
rm_main = df
