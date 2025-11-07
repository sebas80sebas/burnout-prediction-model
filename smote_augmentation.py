"""
SCRIPT PARA ALTAIR AI STUDIO
Operador: Execute Python - SMOTE Data Augmentation
Input: rm_main (ExampleSet enriquecido)
Output: rm_main (ExampleSet balanceado)
"""

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

# ============================================
# RECIBIR DATOS DESDE ALTAIR AI STUDIO
# ============================================
df = rm_main

print("="*60)
print("AUMENTO DE DATOS CON SMOTE EN ALTAIR AI STUDIO")
print("="*60)
print(f"Dataset recibido: {len(df)} registros, {len(df.columns)} columnas")

# ============================================
# 1. VERIFICAR VARIABLE OBJETIVO
# ============================================
if 'Burnout_Risk' not in df.columns:
    print("\n⚠️ ERROR: No se encuentra la variable 'Burnout_Risk'")
    print("   Asegúrate de ejecutar primero el script de Feature Engineering")
    rm_main = df  # Devolver sin cambios
else:
    print(f"\n📊 Distribución de clases ANTES de SMOTE:")
    print(df['Burnout_Risk'].value_counts())
    
    n_class_0 = (df['Burnout_Risk'] == 0).sum()
    n_class_1 = (df['Burnout_Risk'] == 1).sum()
    ratio_original = n_class_1 / n_class_0 if n_class_0 > 0 else 0
    
    print(f"   Clase 0 (bajo riesgo): {n_class_0} ({n_class_0/len(df)*100:.1f}%)")
    print(f"   Clase 1 (alto riesgo): {n_class_1} ({n_class_1/len(df)*100:.1f}%)")
    print(f"   Ratio minoritaria/mayoritaria: {ratio_original:.3f}")
    
    # ============================================
    # 2. PREPARAR DATOS PARA SMOTE
    # ============================================
    print("\n🔧 Preparando datos para SMOTE...")
    
    # Separar features y target
    X = df.drop(['Burnout_Risk'], axis=1, errors='ignore')
    y = df['Burnout_Risk']
    
    # Seleccionar solo columnas numéricas
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Excluir columnas problemáticas
    exclude_cols = ['ID', 'Unnamed: 0']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    X_numeric = X[feature_cols].copy()
    
    print(f"   ✓ Features seleccionadas: {len(feature_cols)}")
    
    # Manejar valores nulos
    if X_numeric.isnull().sum().sum() > 0:
        print(f"   ⚠️ Rellenando {X_numeric.isnull().sum().sum()} valores nulos...")
        X_numeric = X_numeric.fillna(X_numeric.median())
    
    # ============================================
    # 3. APLICAR SMOTE
    # ============================================
    print("\n🚀 Aplicando SMOTE...")
    
    try:
        # Configurar SMOTE
        # sampling_strategy: ratio deseado de clase minoritaria / clase mayoritaria
        # 0.6 = generar hasta que la clase minoritaria sea 60% de la mayoritaria
        smote = SMOTE(
            sampling_strategy=0.6,  # Balance moderado (no perfecto 50-50)
            random_state=42,
            k_neighbors=5
        )
        
        # Aplicar SMOTE
        X_resampled, y_resampled = smote.fit_resample(X_numeric, y)
        
        print(f"\n✅ SMOTE aplicado exitosamente!")
        
        # Estadísticas después de SMOTE
        n_class_0_after = (y_resampled == 0).sum()
        n_class_1_after = (y_resampled == 1).sum()
        ratio_after = n_class_1_after / n_class_0_after if n_class_0_after > 0 else 0
        
        print(f"\n📊 Distribución de clases DESPUÉS de SMOTE:")
        print(f"   Clase 0 (bajo riesgo): {n_class_0_after} ({n_class_0_after/len(y_resampled)*100:.1f}%)")
        print(f"   Clase 1 (alto riesgo): {n_class_1_after} ({n_class_1_after/len(y_resampled)*100:.1f}%)")
        print(f"   Ratio minoritaria/mayoritaria: {ratio_after:.3f}")
        print(f"\n📈 Incremento:")
        print(f"   Registros originales: {len(df)}")
        print(f"   Registros con SMOTE: {len(y_resampled)}")
        print(f"   Nuevos registros sintéticos: {len(y_resampled) - len(df)} (+{(len(y_resampled)/len(df) - 1)*100:.1f}%)")
        
        # ============================================
        # 4. RECONSTRUIR DATASET COMPLETO
        # ============================================
        print("\n🔧 Reconstruyendo dataset completo...")
        
        # Crear DataFrame con datos aumentados
        df_augmented = pd.DataFrame(X_resampled, columns=feature_cols)
        df_augmented['Burnout_Risk'] = y_resampled
        
        # Añadir columnas no numéricas de los registros originales
        non_numeric_cols = [col for col in df.columns if col not in feature_cols and col != 'Burnout_Risk']
        
        # Para los registros originales, mantener las columnas no numéricas
        # Para los sintéticos, usar valores por defecto o más comunes
        n_original = len(df)
        n_synthetic = len(df_augmented) - n_original
        
        for col in non_numeric_cols:
            if col in df.columns:
                # Registros originales: valores originales
                original_values = df[col].values
                # Registros sintéticos: usar el valor más común (moda)
                most_common = df[col].mode()[0] if len(df[col].mode()) > 0 else df[col].iloc[0]
                synthetic_values = [most_common] * n_synthetic
                
                # Combinar
                df_augmented[col] = list(original_values) + synthetic_values
        
        # Añadir flag de identificación
        df_augmented['Es_Sintetico'] = 0
        df_augmented['Es_Sintetico'].iloc[n_original:] = 1
        
        print(f"   ✓ Dataset reconstruido con {len(df_augmented.columns)} columnas")
        
        # ============================================
        # 5. VALIDACIÓN DE CALIDAD
        # ============================================
        print("\n🔍 Validación de calidad de datos sintéticos:")
        
        # Comparar estadísticas de variables clave
        key_vars = ['Age', 'Service time', 'Distance from Residence to Work']
        existing_key_vars = [v for v in key_vars if v in df_augmented.columns]
        
        for var in existing_key_vars[:3]:  # Solo 3 para no saturar el output
            orig_mean = df_augmented[df_augmented['Es_Sintetico'] == 0][var].mean()
            synt_mean = df_augmented[df_augmented['Es_Sintetico'] == 1][var].mean()
            diff_pct = abs(orig_mean - synt_mean) / orig_mean * 100 if orig_mean != 0 else 0
            
            print(f"   {var}:")
            print(f"      Original: μ={orig_mean:.2f}")
            print(f"      Sintético: μ={synt_mean:.2f}")
            print(f"      Diferencia: {diff_pct:.1f}%")
        
        # ============================================
        # 6. PREPARAR OUTPUT
        # ============================================
        print("\n" + "="*60)
        print("✅ AUMENTO DE DATOS COMPLETADO")
        print("="*60)
        print(f"Dataset final: {len(df_augmented)} registros, {len(df_augmented.columns)} columnas")
        print(f"   - Registros originales: {n_original}")
        print(f"   - Registros sintéticos: {n_synthetic}")
        
        # Devolver dataset aumentado
        rm_main = df_augmented
        
    except Exception as e:
        print(f"\n❌ ERROR al aplicar SMOTE: {str(e)}")
        print("   Devolviendo dataset original sin cambios...")
        rm_main = df
