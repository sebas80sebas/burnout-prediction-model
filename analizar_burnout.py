import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# =========================================
# 1. CARGAR CSV
# =========================================
print("Cargando métricas del modelo...")

df = pd.read_csv("burnout_prediction.csv", sep=";", header=0)
df["Criterion"] = df["Criterion"].str.replace('"', '').str.strip()
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

# =========================================
# 2. SEPARAR TRAIN Y TEST
# =========================================
if len(df) != 12:
    print(f"Advertencia: se esperaban 12 métricas (6 train + 6 test), se encontraron {len(df)}")
    exit()

df_train = df.iloc[:6].set_index("Criterion")
df_test = df.iloc[6:].set_index("Criterion")

print("\n=== Métricas Train ===")
print(df_train)
print("\n=== Métricas Test ===")
print(df_test)

# =========================================
# 3. COMPARACIÓN METRICAS
# =========================================
comparacion = pd.DataFrame({
    "Train": df_train["Value"],
    "Test": df_test["Value"],
    "Gap": df_train["Value"] - df_test["Value"]
})
print("\n=== Comparación Train vs Test ===")
print(comparacion)

# =========================================
# 4. MATRIZ DE CONFUSIÓN ESTIMADA
# =========================================
# Asumimos un total de 100 casos para simular
total = 100
recall = df_test.loc["recall", "Value"]
precision = df_test.loc["precision", "Value"]

TP = int(total * recall)
FN = total - TP
FP = int(TP * (1/precision - 1)) if precision > 0 else 0
TN = total - TP - FP - FN
if TN < 0: TN = 0  # prevenir negativo

cm_dict = {"TN": TN, "FP": FP, "FN": FN, "TP": TP}
print("\n=== Matriz de Confusión Estimada ===")
print(cm_dict)

# =========================================
# 5. GRÁFICOS
# =========================================
sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Análisis del Modelo de Predicción de Burnout', fontsize=16, fontweight='bold')

# --- 5.1 Comparación Train vs Test ---
ax1 = axes[0,0]
metrics = ["accuracy", "precision", "recall", "f_measure", "AUC"]
train_vals = df_train.loc[metrics, "Value"]
test_vals = df_test.loc[metrics, "Value"]

x = np.arange(len(metrics))
width = 0.35
ax1.bar(x - width/2, train_vals, width, label='Train', color='lightgreen')
ax1.bar(x + width/2, test_vals, width, label='Test', color='coral')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, rotation=45)
ax1.set_ylim(0,1.05)
ax1.set_ylabel("Score")
ax1.set_title("Train vs Test Metrics")
ax1.legend()
ax1.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)

# --- 5.2 Radar de métricas Test ---
ax2 = axes[0,1]
labels = metrics
values = df_test.loc[metrics,"Value"].values
values = np.append(values, values[0])  # cerrar el radar
angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
angles = np.append(angles, angles[0])
ax2 = plt.subplot(2,2,2, polar=True)
ax2.plot(angles, values, 'o-', linewidth=2)
ax2.fill(angles, values, alpha=0.25)
ax2.set_thetagrids(angles[:-1]*180/np.pi, labels)
ax2.set_ylim(0,1)
ax2.set_title("Radar de Métricas (Test)", y=1.1)

# --- 5.3 Distribución de resultados (TP, FP, FN, TN) ---
ax3 = axes[1,0]
categories = ['TN','FP','FN','TP']
values_cm = [TN, FP, FN, TP]
colors = ['green','orange','red','blue']
ax3.bar(categories, values_cm, color=colors, alpha=0.7, edgecolor='black')
ax3.set_ylabel("Cantidad de casos")
ax3.set_title("Distribución de Resultados Estimada")
for i,v in enumerate(values_cm):
    ax3.text(i, v + 0.5, str(v), ha='center', fontweight='bold')

# --- 5.4 Gap Train-Test ---
ax4 = axes[1,1]
metrics_gap = comparacion.index.tolist()   
gaps = comparacion["Gap"].values
ax4.bar(metrics_gap, gaps, color='purple', alpha=0.7)
ax4.set_ylabel("Train - Test")
ax4.set_title("Diferencia Train vs Test (Gap)")
ax4.axhline(0, color='gray', linestyle='--')
for i,v in enumerate(gaps):
    ax4.text(i, v + 0.01, f"{v:.2f}", ha='center')


plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig("analisis_burnout_completo.png", dpi=300)
print("\n✓ Gráficos guardados: analisis_burnout_completo.png")
plt.show()
