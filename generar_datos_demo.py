"""
generar_datos_demo.py
------------------------
Este script NO es parte del flujo real del proyecto. Se incluye
únicamente para poder probar que entrenamiento.py y modelo.py funcionan
de punta a punta (por ejemplo, en una computadora sin cámara, o antes de
ir a capturar datos reales).

Genera un archivo data/datos_senas.csv con vectores de características
INVENTADOS matemáticamente (no vienen de una mano real) para 5 "señas"
de ejemplo. Sirve solo para validar que el código de entrenamiento
corre sin errores y produce un modelo cargable.

*** Para el proyecto real deben usar captura_datos.py con su cámara,
    siguiendo el diccionario de LESSA como referencia. ***
"""

import os
import numpy as np
import pandas as pd

from utils import nombres_columnas_caracteristicas

SEÑAS_DEMO = ["hola", "gracias", "por_favor", "bien", "adios"]
MUESTRAS_POR_SEÑA = 60
RUTA_CSV = os.path.join("data", "datos_senas.csv")


def main():
    np.random.seed(42)
    columnas = nombres_columnas_caracteristicas()
    n_features = len(columnas)

    filas = []
    for i, seña in enumerate(SEÑAS_DEMO):
        # Cada seña de ejemplo tiene un "centro" distinto en el espacio de
        # características, con algo de ruido alrededor para simular la
        # variación natural entre repeticiones de una misma seña.
        centro = np.random.RandomState(i).uniform(-1, 1, size=n_features)
        for _ in range(MUESTRAS_POR_SEÑA):
            ruido = np.random.normal(0, 0.05, size=n_features)
            filas.append([seña] + list(centro + ruido))

    df = pd.DataFrame(filas, columns=["etiqueta"] + columnas)
    os.makedirs("data", exist_ok=True)
    df.to_csv(RUTA_CSV, index=False)
    print(f"Datos de DEMOSTRACIÓN (sintéticos, no reales) guardados en {RUTA_CSV}")
    print(f"Señas: {SEÑAS_DEMO}  |  {MUESTRAS_POR_SEÑA} muestras cada una")
    print("\nAhora pueden correr: python entrenamiento.py")


if __name__ == "__main__":
    main()
