"""
Prueba de normalidad Shapiro-Wilk - Objetivo Específico 4
Verificación del supuesto de normalidad sobre Faithfulness en dominio (Tabla 14)
previo a la aplicación de la prueba t de una muestra.
"""
import numpy as np
from scipy import stats

# Valores de Faithfulness en dominio (Tabla 14, N=14)
faithfulness_dominio = np.array([
    0.43, 0.78, 0.50, 0.22, 0.36, 0.80, 0.53,
    0.29, 0.80, 0.89, 0.54, 0.56, 0.62, 0.73
])

alpha = 0.05

# Prueba de Shapiro-Wilk
estadistico_w, p_valor = stats.shapiro(faithfulness_dominio)

print("=" * 60)
print("PRUEBA DE NORMALIDAD SHAPIRO-WILK - OBJETIVO ESPECÍFICO 4")
print("=" * 60)
print(f"N (consultas en dominio)   : {len(faithfulness_dominio)}")
print(f"Nivel de significancia (α) : {alpha}")
print(f"Estadístico W              : {estadistico_w:.4f}")
print(f"p-valor                    : {p_valor:.4f}")
print("-" * 60)
if p_valor > alpha:
    print("DECISIÓN: No se rechaza H0 de normalidad.")
    print("Los datos son consistentes con una distribución normal.")
    print("-> La prueba t de Student es estadísticamente apropiada.")
else:
    print("DECISIÓN: Se rechaza H0 de normalidad.")
    print("Los datos NO son consistentes con una distribución normal.")
    print("-> Corresponde usar una prueba no paramétrica (Wilcoxon).")
print("=" * 60)