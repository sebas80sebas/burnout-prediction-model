import pandas as pd
import numpy as np

def rm_main(df):
    """
    Aumento de Datos con Oversampling Inteligente (alternativa a SMOTE)
    Crea muestras sintéticas interpolando entre casos reales
    """
    
    print("="*60)
    print("AUMENTO DE DATOS - Oversampling Inteligente")
    print("="*60)
    print("Dataset recibido: {} registros".format(len(df)))

    # ===============================================================
    # 🔹 CONVERTIR ETIQUETAS NOMINALES 'Bajo'/'Alto' A NUMÉRICAS 0/1
    # ===============================================================
    if 'Burnout_Risk' in df.columns:
        if df['Burnout_Risk'].dtype.name in ['category', 'object']:
            print("\nConvirtiendo etiquetas 'Bajo'/'Alto' a valores 0/1 para entrenamiento...")
            df['Burnout_Risk'] = df['Burnout_Risk'].map({'Bajo': 0, 'Alto': 1})
    
    # Convertir categóricas a numéricas (como ya haces)
    for col in df.columns:
        if df[col].dtype.name == 'category':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Verificar variable objetivo
    if 'Burnout_Risk' not in df.columns:
        print("ERROR: Burnout_Risk no encontrada")
        return df
    
    # Estadísticas iniciales
    n_class_0 = (df['Burnout_Risk'] == 0).sum()
    n_class_1 = (df['Burnout_Risk'] == 1).sum()
    
    print("\nDistribución ANTES:")
    print("  Clase 0 (bajo riesgo): {} ({:.1f}%)".format(n_class_0, n_class_0/len(df)*100))
    print("  Clase 1 (alto riesgo): {} ({:.1f}%)".format(n_class_1, n_class_1/len(df)*100))
    
    
    # Separar clases
    df_majority = df[df['Burnout_Risk'] == 0].copy()
    df_minority = df[df['Burnout_Risk'] == 1].copy()
    
    # Calcular cuantas muestras sinteticas crear
    target_minority = int(len(df_majority) * 0.6)
    n_synthetic = target_minority - len(df_minority)
    
    if n_synthetic <= 0:
        print("\nNo se requiere oversampling")
        return df
    
    print("\nCreando {} muestras sinteticas...".format(n_synthetic))
    
    # Seleccionar columnas numericas (excluir ID y Burnout_Risk)
    exclude_cols = ['ID', 'Unnamed: 0', 'Burnout_Risk']
    numeric_cols = [c for c in df_minority.select_dtypes(include=[np.number]).columns 
                    if c not in exclude_cols]
    
    # Crear muestras sinteticas interpolando
    synthetic_samples = []
    np.random.seed(42)
    
    for i in range(n_synthetic):
        # Seleccionar dos casos aleatorios de la clase minoritaria
        idx1, idx2 = np.random.choice(len(df_minority), 2, replace=True)
        sample1 = df_minority.iloc[idx1]
        sample2 = df_minority.iloc[idx2]
        
        # Crear muestra sintetica
        synthetic = {}
        
        # Interpolar valores numericos
        alpha = np.random.uniform(0, 1)
        for col in numeric_cols:
            val1 = sample1[col] if pd.notna(sample1[col]) else 0
            val2 = sample2[col] if pd.notna(sample2[col]) else 0
            synthetic[col] = val1 + alpha * (val2 - val1)
        
        # Copiar valores categoricos del caso mas cercano
        for col in df_minority.columns:
            if col not in numeric_cols and col not in exclude_cols:
                synthetic[col] = sample1[col]
        
        # Asignar clase
        synthetic['Burnout_Risk'] = 1
        
        synthetic_samples.append(synthetic)
    
    # Crear DataFrame con muestras sinteticas
    df_synthetic = pd.DataFrame(synthetic_samples)
    
    # Combinar: mayoria + minoria original + sinteticos
    df_balanced = pd.concat([df_majority, df_minority, df_synthetic], ignore_index=True)
    
    # Mezclar aleatoriamente
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Marcar muestras sinteticas
    df_balanced['Es_Sintetico'] = 0
    df_balanced.iloc[len(df_majority) + len(df_minority):, 
                     df_balanced.columns.get_loc('Es_Sintetico')] = 1
    
    # Estadisticas finales
    n_class_0_after = (df_balanced['Burnout_Risk'] == 0).sum()
    n_class_1_after = (df_balanced['Burnout_Risk'] == 1).sum()
    
    # ===============================================================
    # 🔹 Forzar que Burnout_Risk sea binominal (2 clases exactas)
    # ===============================================================
    df_balanced['Burnout_Risk'] = df_balanced['Burnout_Risk'].astype(str)
    df_balanced['Burnout_Risk'] = df_balanced['Burnout_Risk'].replace({'0': 'No', '1': 'Sí'})
    df_balanced['Burnout_Risk'] = pd.Categorical(df_balanced['Burnout_Risk'], categories=['No', 'Sí'])

    
    print("\nDistribucion DESPUES:")
    print("  Clase 0 (bajo riesgo): {} ({:.1f}%)".format(n_class_0_after, n_class_0_after/len(df_balanced)*100))
    print("  Clase 1 (alto riesgo): {} ({:.1f}%)".format(n_class_1_after, n_class_1_after/len(df_balanced)*100))
    print("\nIncremento:")
    print("  Registros originales: {}".format(len(df)))
    print("  Registros balanceados: {}".format(len(df_balanced)))
    print("  Nuevos sinteticos: {} (+{:.1f}%)".format(n_synthetic, (len(df_balanced)/len(df) - 1)*100))
    
    
    print("\n" + "="*60)
    print("AUMENTO DE DATOS COMPLETADO")
    print("="*60)
    
    return df_balanced
