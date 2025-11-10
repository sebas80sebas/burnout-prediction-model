import pandas as pd
import numpy as np

def rm_main(df):
    """
    Feature Engineering para Detección de Burnout    
    """
    
    print("="*60)
    print("FEATURE ENGINEERING - Detección de Burnout")
    print("="*60)
    print(f"Dataset recibido: {len(df)} registros, {len(df.columns)} columnas")
    
    # ============================================
    # 1. VARIABLES TEMPORALES/ESTACIONALES
    # ============================================
    print("\n[1/7] Creando variables temporales...")
    
    # Obtener mes y día de la semana
    if 'Month of absence' in df.columns:
        df['Mes'] = df['Month of absence'].fillna(6)  # Rellenar NaN con valor neutral (junio)
    else:
        df['Mes'] = 6
    
    if 'Day of the week' in df.columns:
        df['Dia_Semana'] = df['Day of the week'].fillna(4)  # Miércoles por defecto
    else:
        df['Dia_Semana'] = 4
    
    # Estación del año (0=Invierno, 1=Primavera, 2=Verano, 3=Otoño)
    estaciones_map = {
        1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2,
        7: 2, 8: 2, 9: 3, 10: 3, 11: 3, 12: 0
    }
    df['Estacion'] = df['Mes'].map(estaciones_map).fillna(2)  # Verano por defecto
    
    # Proximidad a vacaciones
    df['Cerca_Vacaciones'] = df['Mes'].isin([6, 7, 11, 12]).astype(int)
    
    # Días específicos de la semana
    df['Es_Lunes'] = (df['Dia_Semana'] == 2).astype(int)
    df['Es_Viernes'] = (df['Dia_Semana'] == 6).astype(int)
    df['Inicio_Fin_Semana'] = ((df['Dia_Semana'] == 2) | (df['Dia_Semana'] == 6)).astype(int)
    
    # Trimestre (1, 2, 3, 4) - SIN .astype(int) que causaba el error
    df['Trimestre'] = pd.cut(df['Mes'], bins=[0, 3, 6, 9, 12], labels=[1, 2, 3, 4])
    # Convertir a numérico de forma segura
    df['Trimestre'] = pd.to_numeric(df['Trimestre'], errors='coerce').fillna(2)
    
    df['Cierre_Trimestre'] = df['Mes'].isin([3, 6, 9, 12]).astype(int)
    
    print(f"   ✓ 9 variables temporales creadas")
    
    # ============================================
    # 2. VARIABLES DE CARGA LABORAL
    # ============================================
    print("\n[2/7] Creando variables de carga laboral...")
    
    # Sobrecarga laboral
    if 'Work load Average/day ' in df.columns:
        # Rellenar NaN antes de calcular quantile
        workload_clean = df['Work load Average/day '].fillna(df['Work load Average/day '].median())
        threshold = workload_clean.quantile(0.75)
        df['Sobrecarga'] = (workload_clean > threshold).astype(int)
        print(f"   ✓ Sobrecarga creada (umbral: {threshold:.2f})")
    else:
        df['Sobrecarga'] = 0
    
    # Ausencias acumuladas por empleado
    if 'ID' in df.columns and 'Absenteeism time in hours' in df.columns:
        df = df.sort_values(['ID', 'Mes'])
        # Rellenar NaN en horas de ausencia
        df['Absenteeism time in hours'] = df['Absenteeism time in hours'].fillna(0)
        df['Ausencias_Acumuladas'] = df.groupby('ID')['Absenteeism time in hours'].cumsum()
        print(f"   ✓ Ausencias acumuladas creadas")
    else:
        df['Ausencias_Acumuladas'] = 0
    
    # ============================================
    # 3. VARIABLES DEMOGRÁFICAS
    # ============================================
    print("\n[3/7] Creando variables demográficas...")
    
    # Grupos de edad - SIN .astype(int) directo
    if 'Age' in df.columns:
        age_clean = df['Age'].fillna(df['Age'].median())
        df['Grupo_Edad'] = pd.cut(age_clean, 
                                   bins=[0, 30, 40, 50, 100],
                                   labels=[1, 2, 3, 4])
        # Convertir de forma segura
        df['Grupo_Edad'] = pd.to_numeric(df['Grupo_Edad'], errors='coerce').fillna(2)
        print(f"   ✓ Grupo_Edad creado")
    else:
        df['Grupo_Edad'] = 2
    
    # Nivel de experiencia - SIN .astype(int) directo
    if 'Service time' in df.columns:
        service_clean = df['Service time'].fillna(df['Service time'].median())
        df['Nivel_Experiencia'] = pd.cut(service_clean,
                                          bins=[0, 5, 10, 15, 100],
                                          labels=[1, 2, 3, 4])
        # Convertir de forma segura
        df['Nivel_Experiencia'] = pd.to_numeric(df['Nivel_Experiencia'], errors='coerce').fillna(2)
        print(f"   ✓ Nivel_Experiencia creado")
    else:
        df['Nivel_Experiencia'] = 2
    
    # Commute largo
    if 'Distance from Residence to Work' in df.columns:
        distance_clean = df['Distance from Residence to Work'].fillna(df['Distance from Residence to Work'].median())
        threshold_dist = distance_clean.quantile(0.75)
        df['Commute_Largo'] = (distance_clean > threshold_dist).astype(int)
        print(f"   ✓ Commute_Largo creado (umbral: {threshold_dist:.2f})")
    else:
        df['Commute_Largo'] = 0
    
    # ============================================
    # 4. VARIABLES DE SALUD
    # ============================================
    print("\n[4/7] Creando variables de salud...")
    
    # Ausencias médicas serias
    if 'Reason for absence' in df.columns:
        razones_serias = [23, 22, 21, 11, 13, 19, 27, 28]
        df['Ausencia_Medica_Seria'] = df['Reason for absence'].isin(razones_serias).astype(int)
        
        # Frecuencia de ausencias médicas por empleado
        if 'ID' in df.columns:
            df['Freq_Ausencias_Medicas'] = df.groupby('ID')['Ausencia_Medica_Seria'].transform('sum')
            print(f"   ✓ Variables de salud creadas")
    else:
        df['Ausencia_Medica_Seria'] = 0
        df['Freq_Ausencias_Medicas'] = 0
    
    # ============================================
    # 5. INTERACCIONES ENTRE VARIABLES
    # ============================================
    print("\n[5/7] Creando interacciones entre variables...")
    
    if 'Age' in df.columns and 'Service time' in df.columns:
        age_clean = df['Age'].fillna(df['Age'].median())
        service_clean = df['Service time'].fillna(df['Service time'].median())
        df['Edad_X_Experiencia'] = age_clean * service_clean
        print(f"   ✓ Edad_X_Experiencia creada")
    else:
        df['Edad_X_Experiencia'] = 0
    
    if 'Distance from Residence to Work' in df.columns:
        distance_clean = df['Distance from Residence to Work'].fillna(df['Distance from Residence to Work'].median())
        df['Distancia_X_Lunes'] = distance_clean * df['Es_Lunes']
        print(f"   ✓ Distancia_X_Lunes creada")
    else:
        df['Distancia_X_Lunes'] = 0
    
    # ============================================
    # 6. VARIABLE OBJETIVO: BURNOUT_RISK
    # ============================================
    print("\n[6/7] Creando variable objetivo: Burnout_Risk...")
    
    if 'Absenteeism time in hours' in df.columns:
        # Rellenar NaN antes de calcular quantile
        hours_clean = df['Absenteeism time in hours'].fillna(0)
        threshold_burnout = hours_clean.quantile(0.75)
        df['Burnout_Risk'] = (hours_clean > threshold_burnout).astype(int)
        
        n_bajo = (df['Burnout_Risk'] == 0).sum()
        n_alto = (df['Burnout_Risk'] == 1).sum()
        
        print(f"   ✓ Burnout_Risk creada")
        print(f"   ✓ Umbral: {threshold_burnout:.1f} horas")
        print(f"   ✓ Distribución:")
        print(f"      - Bajo riesgo (0): {n_bajo} ({n_bajo/len(df)*100:.1f}%)")
        print(f"      - Alto riesgo (1): {n_alto} ({n_alto/len(df)*100:.1f}%)")
    else:
        df['Burnout_Risk'] = 0
    
    # ============================================
    # 7. LIMPIEZA FINAL DE VALORES NULOS
    # ============================================
    print("\n[7/7] Limpieza final de valores nulos...")
    
    # Rellenar cualquier NaN restante en columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    nulos_finales = df.isnull().sum().sum()
    print(f"   ✓ Valores nulos restantes: {nulos_finales}")
    
    # ============================================
    # RESUMEN FINAL
    # ============================================
    print("\n" + "="*60)
    print("✅ FEATURE ENGINEERING COMPLETADO")
    print("="*60)
    print(f"Variables totales: {len(df.columns)}")
    print(f"Registros: {len(df)}")
    print("="*60)
    
    return df
