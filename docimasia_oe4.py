"""
Docimasia de hipótesis - Objetivo Específico 4
Prueba t de una muestra, unilateral derecha, sobre Faithfulness en dominio (Tabla 14)
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Valores de Faithfulness en dominio (Tabla 14, N=14)
faithfulness_dominio = np.array([
    0.43, 0.78, 0.50, 0.22, 0.36, 0.80, 0.53,
    0.29, 0.80, 0.89, 0.54, 0.56, 0.62, 0.73
])

n = len(faithfulness_dominio)
media = np.mean(faithfulness_dominio)
desv_std = np.std(faithfulness_dominio, ddof=1)  # muestral
mu_0 = 0  # condición de Ausencia (H0)
alpha = 0.05
df = n - 1

# Estadístico t
error_estandar = desv_std / np.sqrt(n)
t_calculado = (media - mu_0) / error_estandar
t_critico = stats.t.ppf(1 - alpha, df)
p_valor = 1 - stats.t.cdf(t_calculado, df)

print("=" * 60)
print("DOCIMASIA DE HIPÓTESIS - OBJETIVO ESPECÍFICO 4")
print("=" * 60)
print(f"N (consultas en dominio)   : {n}")
print(f"Media (Faithfulness)       : {media:.4f}")
print(f"Desviación estándar        : {desv_std:.4f}")
print(f"Grados de libertad (df)    : {df}")
print(f"Nivel de significancia (α) : {alpha}")
print(f"t crítico (t_0.05, {df})     : {t_critico:.4f}")
print(f"t calculado                : {t_calculado:.4f}")
print(f"p-valor                    : {p_valor:.6f}")
print("-" * 60)
if t_calculado > t_critico:
    print("DECISIÓN: Se rechaza H0. Se acepta H1 (HG1 respaldada).")
else:
    print("DECISIÓN: No se rechaza H0.")
print("=" * 60)

# --- Gráfico de la zona de rechazo ---
x = np.linspace(-4.5, 12, 1000)
y = stats.t.pdf(x, df)

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(x, y, color="#1f4e79", linewidth=2, label=f"Distribución t (df={df})")

# Zona de rechazo (sombreada)
x_rechazo = np.linspace(t_critico, 12, 200)
y_rechazo = stats.t.pdf(x_rechazo, df)
ax.fill_between(x_rechazo, y_rechazo, color="#c0504d", alpha=0.5,
                 label=f"Zona de rechazo (t > {t_critico:.3f})")

# Línea del t crítico
ax.axvline(t_critico, color="#c0504d", linestyle="--", linewidth=1.5)
ax.text(t_critico, max(y) * 0.55, f"  t crítico = {t_critico:.3f}",
        color="#c0504d", fontsize=10, fontweight="bold")

# Línea del t calculado
ax.axvline(t_calculado, color="#2e7d32", linestyle="-", linewidth=2)
ax.text(t_calculado, max(y) * 0.85, f"  t calculado = {t_calculado:.2f}",
        color="#2e7d32", fontsize=10, fontweight="bold")

ax.set_xlabel("Valor de t", fontsize=11)
ax.set_ylabel("Densidad de probabilidad", fontsize=11)
ax.set_title("Prueba t de una muestra (unilateral derecha) — Faithfulness en dominio (OE4)",
             fontsize=12, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.set_facecolor("white")
fig.patch.set_facecolor("white")
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig("figura19_zona_rechazo.png", dpi=200, facecolor="white")
print("\nGráfico guardado en figura19_zona_rechazo.png")
