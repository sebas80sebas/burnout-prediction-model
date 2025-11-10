import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

def rm_main(df):
    """
    Aumento de Datos con SMOTE para Balanceo de Clases
    Input: DataFrame con variable Burnout_Risk
    Output: DataFrame balanceado con datos sintéticos
    """
    
    print("="*60)
    print("AUMENTO DE DATOS CON SMOTE")
    print("="*60)
    print(f"Dataset recibido: {len(df)} registros, {len(df.columns)} columnas")
    
    # ============================================
    # 1. VERIFICAR VARIABLE OBJETIVO
    # ============================================
    if 'Burnout_Risk' not in df.columns:
        print("\n❌ ERROR: Variable 'Burnout_Risk' no encontrada")
        print("   Asegúrate de ejecutar primero Feature Engineering")
        return df
    
    print(f"\n📊 Distribución ANTES de SMOTE:")
    n_class_0 = (df['Burnout_Risk'] == 0).sum()
    n_class_1 = (df['Burnout_Risk'] == 1).sum()
    ratio_orig = n_class_1 / n_class_0 if n_class_0 > 0 else 0
    
    print(f"   Clase 0 (bajo riesgo): {n_class_0} ({n_class_0/len(df)*100:.1f}%)")
    print(f"   Clase 1 (alto riesgo): {n_class_1} ({n_class_1/len(df)*100:.1f}%)")
    print(f"   Ratio: {ratio_orig:.3f}")
    
    # ============================================
    # 2. PREPARAR DATOS
    # ============================================
    print("\n🔧 Preparando datos para SMOTE...")
    
    # Separar features y target
    y = df['Burnout_Risk'].copy()
    X = df.drop(['Burnout_Risk'], axis=1, errors='ignore')
    
    # Seleccionar solo columnas numéricas
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Excluir columnas problemáticas
    exclude_cols = ['ID', 'Unnamed: 0']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    X_numeric = X[feature_cols].copy()
    
    print(f"   ✓ Features seleccionadas: {len(feature_cols)}")
    
    # Manejar valores nulos
    nulos = X_numeric.isnull().sum().sum()
    if nulos > 0:
        print(f"   ⚠️ Rellenando {nulos} valores nulos con mediana...")
        X_numeric = X_numeric.fillna(X_numeric.median())
    
    # ============================================
    # 3. APLICAR SMOTE
    # ============================================
    print("\n🚀 Aplicando SMOTE...")
    
    try:
        # Configurar SMOTE
        # sampling_strategy=0.6 significa que la clase minoritaria
        # será el 60% de la clase mayoritaria
        smote = SMOTE(
            sampling_strategy=0.6,
            random_state=42,
            k_neighbors=5
        )
        
        # Aplicar SMOTE
        X_resampled, y_resampled = smote.fit_resample(X_numeric, y)
        
        # Estadísticas después de SMOTE
        n_class_0_after = (y_resampled == 0).sum()
        n_class_1_after = (y_resampled == 1).sum()
        ratio_after = n_class_1_after / n_class_0_after if n_class_0_after > 0 else 0
        
        print(f"\n✅ SMOTE aplicado exitosamente!")
        print(f"\n📊 Distribución DESPUÉS de SMOTE:")
        print(f"   Clase 0 (bajo riesgo): {n_class_0_after} ({n_class_0_after/len(y_resampled)*100:.1f}%)")
        print(f"   Clase 1 (alto riesgo): {n_class_1_after} ({n_class_1_after/len(y_resampled)*100:.1f}%)")
        print(f"   Ratio: {ratio_after:.3f}")
        print(f"\n📈 Incremento:")
        print(f"   Registros originales: {len(df)}")
        print(f"   Registros con SMOTE: {len(y_resampled)}")
        print(f"   Nuevos sintéticos: {len(y_resampled) - len(df)} (+{(len(y_resampled)/len(df) - 1)*100:.1f}%)")
        
        # ============================================
        # 4. RECONSTRUIR DATASET
        # ============================================
        print("\n🔧 Reconstruyendo dataset completo...")
        
        # Crear DataFrame con datos aumentados
        df_augmented = pd.DataFrame(X_resampled, columns=feature_cols)
        df_augmented['Burnout_Risk'] = y_resampled
        
        # Añadir columnas no numéricas
        n_original = len(df)
        n_synthetic = len(df_augmented) - n_original
        
        non_numeric_cols = [col for col in df.columns 
                           if col not in feature_cols and col != 'Burnout_Risk']
        
        for col in non_numeric_cols:
            if col in df.columns:
                # Valores originales
                orig_vals = df[col].values.tolist()
                
                # Para sintéticos, usar el valor más común
                if len(df[col].mode()) > 0:
                    most_common = df[col].mode()[0]
                else:
                    most_common = df[col].iloc[0] if len(df) > 0 else 0
                
                synt_vals = [most_common] * n_synthetic
                
                # Combinar
                df_augmented[col] = orig_vals + synt_vals
        
        # Flag de identificación
        df_augmented['Es_Sintetico'] = 0
        df_augmented.loc[n_original:, 'Es_Sintetico'] = 1
        
        print(f"   ✓ Dataset reconstruido: {len(df_augmented.columns)} columnas")
        
        # ============================================
        # 5. VALIDACIÓN RÁPIDA
        # ============================================
        print("\n🔍 Validación de calidad:")
        
        # Comparar medias de variables clave
        key_vars = ['Age', 'Service time', 'Distance from Residence to Work']
        existing_vars = [v for v in key_vars if v in df_augmented.columns]
        
        for var in existing_vars[:2]:  # Solo 2 para no saturar
            orig_mean = df_augmented[df_augmented['Es_Sintetico'] == 0][var].mean()
            synt_mean = df_augmented[df_augmented['Es_Sintetico'] == 1][var].mean()
            diff_pct = abs(orig_mean - synt_mean) / orig_mean * 100 if orig_mean != 0 else 0
            
            print(f"   {var}: Original={orig_mean:.1f}, Sintético={synt_mean:.1f}, Diff={diff_pct:.1f}%")
        
        print("\n" + "="*60)
        print("✅ AUMENTO DE DATOS COMPLETADO")
        print("="*60)
        print(f"Dataset final: {len(df_augmented)} registros")
        print("="*60)
        
        return df_augmented
        
    except Exception as e:
        print(f"\n❌ ERROR al aplicar SMOTE: {str(e)}")
        print("   Devolviendo dataset original sin cambios...")
        return df
