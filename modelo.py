"""
modelo.py
----------
Envoltorio (wrapper) alrededor del modelo entrenado. Aísla al resto de
la aplicación (main.py) de los detalles de scikit-learn/joblib: main.py
solo necesita llamar a predecir(vector) y recibir una etiqueta de texto
más un porcentaje de confianza.
"""

import os
import joblib

RUTA_MODELO = os.path.join("modelos", "modelo_lessa.pkl")


class ClasificadorSenas:
    def __init__(self, ruta_modelo=RUTA_MODELO):
        self.modelo = None
        self.codificador = None
        self.ruta_modelo = ruta_modelo
        self.cargar()

    def cargar(self):
        """Intenta cargar el modelo desde disco. Si no existe todavía
        (por ejemplo, la primera vez que se usa el proyecto y aún no se
        ha corrido entrenamiento.py), self.modelo queda en None y
        disponible() devuelve False."""
        if os.path.exists(self.ruta_modelo):
            datos = joblib.load(self.ruta_modelo)
            self.modelo = datos["modelo"]
            self.codificador = datos["codificador"]
            return True
        return False

    def disponible(self):
        return self.modelo is not None

    def predecir(self, vector_caracteristicas):
        """
        Recibe un vector de características (ver utils.extraer_caracteristicas)
        y devuelve una tupla (etiqueta_texto, confianza_0_a_1).

        Si el modelo no está cargado, devuelve (None, 0.0).
        """
        if not self.disponible():
            return None, 0.0

        vector = vector_caracteristicas.reshape(1, -1)
        probabilidades = self.modelo.predict_proba(vector)[0]
        indice_mejor = probabilidades.argmax()
        confianza = float(probabilidades[indice_mejor])
        etiqueta = self.codificador.inverse_transform([indice_mejor])[0]
        return etiqueta, confianza
