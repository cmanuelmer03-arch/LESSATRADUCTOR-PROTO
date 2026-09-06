"""
entrenamiento.py
------------------
Entrena el modelo de clasificación de señas a partir del CSV generado
por captura_datos.py (data/datos_senas.csv) y guarda el modelo entrenado
en modelos/modelo_lessa.pkl.

Se usa un RandomForestClassifier de scikit-learn. Se eligió sobre una
red neuronal profunda / TensorFlow porque:
  - Con pocos datos por clase (decenas o cientos de muestras, no miles)
    un RandomForest generaliza muy bien y es difícil que "sobreajuste".
  - Entrena en segundos en cualquier laptop, sin GPU.
  - Es fácil de interpretar y depurar para un proyecto STEM de bachillerato/
    universidad.
Si en el futuro se recolectan muchos más datos (miles de muestras por
seña) se puede migrar a una red neuronal con TensorFlow/Keras siguiendo
la misma idea: el vector de utils.extraer_caracteristicas() como entrada.

Ejecutar simplemente con:
    python entrenamiento.py
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

RUTA_CSV = os.path.join("data", "datos_senas.csv")
RUTA_MODELO = os.path.join("modelos", "modelo_lessa.pkl")
MINIMO_MUESTRAS_POR_CLASE = 10


def cargar_datos():
    if not os.path.exists(RUTA_CSV):
        raise FileNotFoundError(
            f"No se encontró {RUTA_CSV}. Primero deben capturar datos con "
            f"'python captura_datos.py' (o generar datos de demostración con "
            f"'python generar_datos_demo.py')."
        )
    df = pd.read_csv(RUTA_CSV)
    return df


def main():
    df = cargar_datos()

    conteo = df["etiqueta"].value_counts()
    print("Muestras por seña:")
    print(conteo, "\n")

    clases_insuficientes = conteo[conteo < MINIMO_MUESTRAS_POR_CLASE]
    if len(clases_insuficientes) > 0:
        print(f"Aviso: estas señas tienen menos de {MINIMO_MUESTRAS_POR_CLASE} muestras, "
              f"el modelo podría reconocerlas mal:")
        print(clases_insuficientes, "\n")

    X = df.drop(columns=["etiqueta"]).values
    y_texto = df["etiqueta"].values

    # El modelo trabaja con números, no con texto, así que convertimos
    # cada nombre de seña ("hola", "gracias", ...) a un número con
    # LabelEncoder. Guardamos también el encoder para poder traducir la
    # predicción de vuelta a texto en main.py.
    codificador = LabelEncoder()
    y = codificador.fit_transform(y_texto)

    # Si hay muy pocas muestras en alguna clase, train_test_split con
    # estratificación podría fallar; en ese caso entrenamos con todo.
    try:
        X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        print("Muy pocas muestras para separar en train/test de forma estratificada; "
              "se entrenará con el 100% de los datos y no se mostrará evaluación.")
        X_entrenamiento, y_entrenamiento = X, y
        X_prueba, y_prueba = None, None

    modelo = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        class_weight="balanced",  # ayuda si algunas señas tienen más muestras que otras
    )
    modelo.fit(X_entrenamiento, y_entrenamiento)

    if X_prueba is not None:
        y_predicho = modelo.predict(X_prueba)
        print(f"Precisión (accuracy) en datos de prueba: {accuracy_score(y_prueba, y_predicho):.2%}\n")
        print("Reporte por clase:")
        print(classification_report(
            y_prueba, y_predicho,
            labels=range(len(codificador.classes_)),
            target_names=codificador.classes_,
            zero_division=0,
        ))

    os.makedirs("modelos", exist_ok=True)
    joblib.dump({"modelo": modelo, "codificador": codificador}, RUTA_MODELO)
    print(f"Modelo guardado en {RUTA_MODELO}")


if __name__ == "__main__":
    main()
